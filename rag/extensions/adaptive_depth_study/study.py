"""Resumable, controlled practice/development study; no pretrained model is changed."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from policies import adaptive_k, guarded_k

OUT = ROOT / "results/adaptive_depth_study"
SYSTEM = "Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."
SEED = 20261105
FIXED = [1, 3, 5, 10, 20]
THRESHOLDS = [0.0, .1, .2, .3, .4, .5, .7, 1.0, 1e9]
POLICY = dict(buffer=5, min_k=1, max_k=20, search_fraction=.9)
GUARD_POLICY = {**POLICY, "buffer": 0}
PROTOCOL = {
    "version": "adaptive-depth-v2", "seed": SEED, "practice_queries": 50,
    "tuning_queries": 30, "validation_queries": 20, "dev_queries": 50,
    "fixed_counts": FIXED, "candidate_pool": 100, "policy": POLICY, "guard_policy": GUARD_POLICY,
    "guard_threshold_grid": THRESHOLDS, "quality_tolerance": .02,
    "generation_temperature": .1, "judge_temperature": 0.0,
    "generation_repeats": 1, "judge_repeats": 1,
    "retrieval_timed_repeats": 10, "policy_timed_repeats": 100,
    "paper": "https://aclanthology.org/2025.emnlp-main.1017/",
    "adaptation": "Top-100 course candidates; gap search within first 90; B=5; output clamped to 1..20. Not original benchmark reproduction.",
    "selection": "Best fixed: highest tuning mean quality, then fewer tokens. Guard: minimum tuning mean prompt tokens within .02 of best-fixed tuning quality; quality breaks token ties.",
    "uncertainty": "Paired bootstrap across queries, conditional on one saved generation/judgment; not judge or generation reliability intervals.",
    "timing": "Warm exact retrieval median + policy median + context build + client API time. ASR, embedding, model loading and judging excluded.",
    "budgets": "At most 550 distinct answer prompts before retries, plus one judge call per answer. No monetary/GPU-energy inference.",
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def progress(phase, **extra):
    print(json.dumps({"phase": phase, **extra}), flush=True)
    save(OUT / "progress.json", {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})


def mean(values):
    return statistics.mean(values)


def make_protocol():
    from src.data import CourseData
    data = CourseData(ROOT / "data/rag_corpus")
    pilot = {r["query_id"] for r in read_rows(ROOT / "results/fixed_k_pilot/inputs.jsonl")}
    practice = sorted((r for r in data.audio_items("practice") if r["query_id"] not in pilot), key=lambda r:r["query_id"])
    dev = sorted(data.audio_items("dev"), key=lambda r:r["query_id"])
    chosen = random.Random(SEED).sample(practice, 50)
    heldout = random.Random(SEED+1).sample(dev, 50)
    for i, row in enumerate(chosen):
        row["study_group"] = "tune" if i < 30 else "validation"
    for row in heldout:
        row["study_group"] = "dev"
    value = {**PROTOCOL, "pilot_ids_excluded": sorted(pilot), "queries": chosen + heldout}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == value, "Protocol changed: use a fresh study directory"
    else:
        save(protocol_path, value)
    # A hash makes changes after sampling visible; no secrets are included.
    save(OUT / "source_hashes.json", {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    return value, data


def prepare(protocol, data):
    from src.asr_adapter import ASRAdapter
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    import numpy as np
    queries = protocol["queries"]
    transcript_path = OUT / "transcripts.jsonl"
    transcripts = {r["query_id"]:r for r in read_rows(transcript_path)}
    missing = [q for q in queries if q["query_id"] not in transcripts]
    if missing:
        asr = ASRAdapter(cuda_groups="none")
        t = time.perf_counter()
        asr.prepare()
        save(OUT / "asr_load.json", {"load_seconds":time.perf_counter()-t,"excluded_from_downstream":True})
        try:
            for i, q in enumerate(missing, 1):
                t = time.perf_counter()
                text = asr.transcribe(data.data_dir / q["audio_file"])
                row = {**q,"transcript":text,"asr_ms":(time.perf_counter()-t)*1000,"asr_cuda":"none"}
                append(transcript_path, row)
                transcripts[q["query_id"]] = row
                progress("transcribing", completed=len(transcripts), total=len(queries), query_id=q["query_id"])
        finally:
            asr.close()
    candidate_path = OUT / "candidates.jsonl"
    candidates = {r["query_id"]:r for r in read_rows(candidate_path)}
    missing = [q for q in queries if q["query_id"] not in candidates]
    if missing:
        encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
        retriever = ExactDenseRetriever(data.data_dir / "embeddings/corpus_enc_a.npy", data.data_dir / "embeddings/corpus_enc_a_ids.json")
        for _ in range(2):
            encoder.encode(transcripts[missing[0]["query_id"]]["transcript"])
        for q in missing:
            row = transcripts[q["query_id"]]
            t = time.perf_counter()
            vector = encoder.encode(row["transcript"])
            embedding_ms = (time.perf_counter()-t)*1000
            for _ in range(2):
                ret = retriever.retrieve(vector, k=100)
            timings=[]
            for _ in range(10):
                ret = retriever.retrieve(vector, k=100)
                timings.append(ret["latency_ms"])
            entry = {**row, "record_ids":ret["record_ids"],"scores":ret["scores"],
                     "embedding_ms":embedding_ms,"retrieval_ms":statistics.median(timings),
                     "retrieval_repeats_ms":timings,"query_vector":vector.tolist()}
            append(candidate_path, entry)
            candidates[q["query_id"]] = entry
            progress("retrieving", completed=len(candidates), total=len(queries), query_id=q["query_id"])
    return candidates


def call(client, model, messages, *, temperature, json_mode=False):
    error = None
    for attempt in range(3):
        try:
            args = dict(model=model, messages=messages, temperature=temperature)
            if json_mode:
                args["response_format"] = {"type":"json_object"}
            t = time.perf_counter()
            response = client.chat.completions.create(**args)
            return response, (time.perf_counter()-t)*1000
        except Exception as exc:
            error = exc
            append(OUT / "request_errors.jsonl", {"type":type(exc).__name__,"attempt":attempt+1,"model":model})
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"API request failed after retries: {type(error).__name__}") from error


def make_clients():
    from src.generator import ELMGenerator
    from src.evaluation.answer_judge import Judge
    gen = ELMGenerator()
    judge_model = os.getenv("ELM_JUDGE_MODEL") or gen.model
    judge = Judge(judge_model)
    gen.client.max_retries = 1
    judge.client.max_retries = 1
    return gen, judge


def evaluate_pairs(pairs, candidates, data, gen, judge):
    """Cache identical prompts so policies selecting the same evidence share answers."""
    from src.context_builder import SimpleContextBuilder
    from src.evaluation.answer_judge import JUDGE_SYSTEM_PROMPT, build_public_gold, RUBRIC_VERSION
    builder = SimpleContextBuilder()
    answers_path, judgments_path = OUT / "answers.jsonl", OUT / "judgments.jsonl"
    answers = {r["request_hash"]:r for r in read_rows(answers_path)}
    judgments = {r["judge_hash"]:r for r in read_rows(judgments_path)}
    pairs = sorted(set(pairs))
    random.Random(SEED+len(pairs)).shuffle(pairs)
    # Access dev judgments only after frozen_config.json has been saved.
    splits = {candidates[q]["split"] for q,k in pairs}
    if "dev" in splits:
        assert (OUT / "frozen_config.json").exists()
    gold = {split:build_public_gold(data.data_dir, split) for split in splits}
    evaluated = {}
    for index, (qid,k) in enumerate(pairs, 1):
        row = candidates[qid]
        t = time.perf_counter()
        built = builder.build(row["transcript"], row["record_ids"][:k], data)
        context_ms = (time.perf_counter()-t)*1000
        key = digest({"system":SYSTEM,"prompt":built["prompt"],"model":gen.model,"temperature":.1,"endpoint":str(gen.client.base_url)})
        if key not in answers:
            if len(answers) >= 550:
                raise RuntimeError("Preregistered 550-prompt generation budget reached")
            response, elapsed = call(gen.client, gen.model, [{"role":"system","content":SYSTEM},{"role":"user","content":built["prompt"]}], temperature=.1)
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise RuntimeError(f"Incomplete answer: finish_reason={choice.finish_reason}")
            usage = response.usage
            if usage is None or usage.prompt_tokens is None:
                raise RuntimeError("ELM did not report prompt tokens; cannot tune a token-saving policy")
            answer = {"request_hash":key,"query_id":qid,"split":row["split"],"k":k,"transcript":row["transcript"],
                      "answer":choice.message.content or "","prompt":built["prompt"],"retrieved_record_ids":row["record_ids"][:k],
                      "prompt_tokens":usage.prompt_tokens,"completion_tokens":usage.completion_tokens,"generation_ms":elapsed,
                      "context_ms":context_ms,"model":gen.model,"response_model":response.model,"response_id":response.id,
                      "finish_reason":choice.finish_reason,"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            append(answers_path, answer)
            answers[key] = answer
        answer = answers[key]
        docs = gold[row["split"]][qid]["gold_documents"]
        jkey = digest({"model":judge.model,"endpoint":str(judge.client.base_url),"system":JUDGE_SYSTEM_PROMPT,
                       "question":row["transcript"],"answer":answer["answer"],"docs":docs,"json_mode":True,"rubric":RUBRIC_VERSION})
        if jkey not in judgments:
            payload = {"question":row["transcript"],"candidate_answer":answer["answer"],
                       "gold_evidence":[{"record_id":d["record_id"],"text":d["text"]} for d in docs]}
            prompt = 'Return JSON exactly as {"correctness":0,"grounding":0,"completeness":0,"explanation":"brief reason"}.\n\nINPUT:\n' + json.dumps(payload, ensure_ascii=False)
            parsed = None
            for attempt in range(3):
                response, elapsed = call(judge.client, judge.model,[{"role":"system","content":JUDGE_SYSTEM_PROMPT},{"role":"user","content":prompt}],temperature=0.0,json_mode=True)
                try:
                    parsed = judge.parse(response.choices[0].message.content)
                    break
                except (ValueError, KeyError, TypeError):
                    append(OUT / "judge_format_errors.jsonl", {"query_id":qid,"k":k,"attempt":attempt+1,"raw":response.choices[0].message.content})
            if parsed is None:
                raise RuntimeError(f"Invalid judge output for {qid}, k={k}; no zero score substituted")
            judgment = {"judge_hash":jkey,"request_hash":key,"query_id":qid,"k":k,"model":judge.model,
                        **parsed,"normalised_answer_score":sum(parsed[d] for d in ("correctness","grounding","completeness"))/12}
            append(judgments_path, judgment)
            judgments[jkey] = judgment
        evaluated[(qid,k)] = {**answer,**judgments[jkey]}
        progress("evaluating_"+row["study_group"], current=index, pairs=len(pairs), unique_answers=len(answers), unique_judgments=len(judgments), query_id=qid, k=k)
    return evaluated


def metrics(qid, k, candidates, evaluated, data):
    row = candidates[qid]
    answer = evaluated[(qid,k)]
    if not hasattr(data, "study_qrels"):
        data.study_qrels = {}
    if row["split"] not in data.study_qrels:
        data.study_qrels[row["split"]] = data.load_qrels(row["split"])
    rel = data.study_qrels[row["split"]][qid]
    coverage = len(set(row["record_ids"][:k]) & set(rel))/len(rel)
    return {"query_id":qid,"k":k,"score":answer["normalised_answer_score"],"prompt_tokens":answer["prompt_tokens"],
            "completion_tokens":answer["completion_tokens"],"recall":coverage,"generation_ms":answer["generation_ms"],
            "retrieval_ms":row["retrieval_ms"],"context_ms":answer["context_ms"]}


def summarise(rows):
    keys=["k","score","prompt_tokens","completion_tokens","recall","generation_ms","retrieval_ms","selection_ms","downstream_ms"]
    return {"queries":len(rows),**{"mean_"+k:mean([r[k] for r in rows]) for k in keys},
            "median_downstream_ms":statistics.median(r["downstream_ms"] for r in rows),
            "p95_downstream_ms":sorted(r["downstream_ms"] for r in rows)[math.ceil(.95*len(rows))-1]}


def policy_record(name, qid, *, candidates, evaluated, data, fixed_k=None, threshold=None):
    scores=candidates[qid]["scores"]
    def select():
        if name.startswith("fixed_"):
            return {"k":fixed_k}
        if name=="adaptive":
            return adaptive_k(scores,**POLICY)
        if name=="gap_only":
            return adaptive_k(scores,**GUARD_POLICY)
        return guarded_k(scores,threshold=threshold,fixed_k=fixed_k,**GUARD_POLICY)
    result=select()
    timings=[]
    for _ in range(100):
        t=time.perf_counter_ns(); select(); timings.append((time.perf_counter_ns()-t)/1e6)
    row=metrics(qid,result["k"],candidates,evaluated,data)
    selection_ms=statistics.median(timings)
    return {**row,"policy":name,"decision":result,"selection_ms":selection_ms,
            "downstream_ms":row["retrieval_ms"]+selection_ms+row["context_ms"]+row["generation_ms"]}


def paired_interval(rows, reference):
    import numpy as np
    base={r["query_id"]:r for r in reference}
    difference=np.asarray([r["score"]-base[r["query_id"]]["score"] for r in rows])
    rng=np.random.default_rng(SEED)
    samples=rng.choice(difference,size=(5000,len(difference)),replace=True).mean(axis=1)
    return {"mean_difference":float(difference.mean()),"ci95":np.quantile(samples,[.025,.975]).tolist(),
            "wins":int((difference>0).sum()),"ties":int((difference==0).sum()),"losses":int((difference<0).sum())}


def write_analysis(groups, candidates, evaluated, data, config):
    report={"protocol":{k:v for k,v in PROTOCOL.items() if k!="queries"},"frozen_config":config,"groups":{}}
    allrows=[]
    for group,qids in groups.items():
        options={f"fixed_{k}":{"fixed_k":k} for k in (FIXED if group!="dev" else sorted(set([5,config["best_fixed_k"]])))}
        options["adaptive"]={}
        options["gap_only"]={}
        options["guarded"]={"fixed_k":config["best_fixed_k"],"threshold":config["threshold"]}
        rows={name:[policy_record(name,qid,candidates=candidates,evaluated=evaluated,data=data,**args) for qid in qids] for name,args in options.items()}
        base=rows["fixed_5"]
        report["groups"][group]={name:{**summarise(values),"paired_vs_fixed5":paired_interval(values,base)} for name,values in rows.items()}
        allrows += [r for values in rows.values() for r in values]
    save(OUT/"analysis.json",report)
    (OUT/"policy_results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in allrows),encoding="utf-8")
    diagnostics=[]
    for group,qids in groups.items():
        rels=data.load_qrels("dev" if group=="dev" else "practice")
        for qid in qids:
            ranks=[i+1 for i,rid in enumerate(candidates[qid]["record_ids"]) if rid in rels[qid]]
            diagnostics.append({"group":group,"query_id":qid,"relevant_ranks_top100":ranks,"known_relevant_count":len(rels[qid]),
                                "adaptive":adaptive_k(candidates[qid]["scores"],**POLICY),
                                "scores_by_k":{str(k):evaluated[(qid,k)]["normalised_answer_score"] for q,k in evaluated if q==qid}})
    save(OUT/"diagnostics.json",diagnostics)
    lines=["# Adaptive retrieval depth: measured study", "", "## Shared question", "",
           "To what extent can ASR execution optimisations and query-adaptive retrieval depth, individually and combined, reduce the response time and resource usage of an audio-RAG system without lowering answer quality, compared with the supplied baseline?", "",
           "This study evaluates the retrieval contribution only. ASR optimisation and combined full-pipeline measurements remain separate work.", "",
           "## Results", "", "| Group | Policy | n | Mean score | Mean input tokens | Mean k | Mean downstream s | Mean recall |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group, policies in report["groups"].items():
        for name,s in policies.items():
            lines.append(f"| {group} | {name} | {s['queries']} | {s['mean_score']:.4f} | {s['mean_prompt_tokens']:.1f} | {s['mean_k']:.2f} | {s['mean_downstream_ms']/1000:.3f} | {s['mean_recall']:.4f} |")
    lines += ["", "## Interpretation limits", "",
              "The 50 practice queries exclude the original five pilot queries and are split before evaluation into 30 tuning and 20 validation queries. The separately sampled 50 dev queries represent 50/125 coverage; they are not the full split. The guard threshold and best fixed count use only tuning judgments; validation and dev do not change configuration.", "",
              "The adaptive baseline is a bounded top-100 adaptation of the published algorithm with B=5, not reproduction of its original benchmarks. Bare gap selection uses B=0; the guard accepts this bare cutoff only for an unclipped sufficiently prominent gap, otherwise falling back to the tuning-selected fixed count. Reporting the bare rule separately distinguishes removing the buffer from adding the safeguard. This is a simple tested modification, not claimed as novel. Selection uses only retrieval scores, never qrels or judged answers at inference.", "",
              "Each distinct prompt has one generation and one JSON-mode judgment. Identical prompts selected by different policies share the same cached answer and judgment; the corresponding generation latency is one shared observation, not an independent rerun. Local retrieval uses ten warm timing repeats; selection uses 100. Downstream time excludes ASR, embedding, loading and judging; it combines these warm local measurements with the single observed API response time.", "",
              "Paired bootstrap intervals resample queries; they do not quantify generation or judge uncertainty. The same model generates and judges, which can bias scores. Failure cases and per-query evidence coverage must accompany averages. Missing/invalid API judgments are never scored as zero. A .02 tuning tolerance is an operational screening criterion, not proof of statistical non-inferiority.", "",
              "Tokens are measured resource proxies, not billed monetary cost or server GPU/energy measurements. Findings may be negative. Evidence coverage from public qrels is not the same as answer sufficiency."]
    (OUT/"STUDY_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def run(stage):
    from dotenv import load_dotenv
    load_dotenv(ROOT/".env")
    OUT.mkdir(parents=True,exist_ok=True)
    protocol,data=make_protocol()
    metadata={"python":sys.version,"platform":platform.platform(),"slurm_job_id":os.getenv("SLURM_JOB_ID"),"hostname":platform.node(),
              "packages":{}}
    for name in ["torch","numpy","sentence-transformers","transformers","openai"]:
        try: metadata["packages"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    try: metadata["gpu"]=subprocess.check_output(["nvidia-smi","-L"],text=True).strip()
    except Exception: metadata["gpu"]="unavailable"
    save(OUT/"environment.json",metadata)
    candidates=prepare(protocol,data)
    progress("inputs_ready",queries=len(candidates))
    if stage=="prepare": return
    gen,judge=make_clients()
    save(OUT/"models.json",{"generator":gen.model,"judge":judge.model,"endpoint":str(gen.client.base_url),"json_mode":True})
    groups={g:[q["query_id"] for q in protocol["queries"] if q["study_group"]==g] for g in ["tune","validation","dev"]}
    practice=groups["tune"]+groups["validation"]
    pairs=[(qid,k) for qid in practice for k in FIXED]
    pairs += [(qid,adaptive_k(candidates[qid]["scores"],**POLICY)["k"]) for qid in practice]
    pairs += [(qid,adaptive_k(candidates[qid]["scores"],**GUARD_POLICY)["k"]) for qid in practice]
    evaluated=evaluate_pairs(pairs,candidates,data,gen,judge)
    tune=groups["tune"]
    fixed_stats={k:{"score":mean(evaluated[(q,k)]["normalised_answer_score"] for q in tune),
                    "tokens":mean(evaluated[(q,k)]["prompt_tokens"] for q in tune)} for k in FIXED}
    best=min(FIXED,key=lambda k:(-fixed_stats[k]["score"],fixed_stats[k]["tokens"],k))
    grid=[]
    for threshold in THRESHOLDS:
        counts={q:guarded_k(candidates[q]["scores"],threshold=threshold,fixed_k=best,**GUARD_POLICY)["k"] for q in tune}
        quality=mean(evaluated[(q,k)]["normalised_answer_score"] for q,k in counts.items())
        tokens=mean(evaluated[(q,k)]["prompt_tokens"] for q,k in counts.items())
        grid.append({"threshold":threshold,"quality":quality,"tokens":tokens,
                     "adaptive_choices":sum(k!=best for k in counts.values())})
    feasible=[r for r in grid if r["quality"] >= fixed_stats[best]["score"]-.02-1e-12]
    chosen=min(feasible,key=lambda r:(r["tokens"],-r["quality"],r["threshold"]))
    config={"best_fixed_k":best,"threshold":chosen["threshold"],"quality_tolerance":.02,"fixed_tuning":fixed_stats,"guard_grid":grid,
            "selection_uses":"30 practice-tuning queries only","protocol_hash":digest(protocol)}
    frozen=OUT/"frozen_config.json"
    if frozen.exists():
        assert json.loads(frozen.read_text())==json.loads(json.dumps(config)), "Frozen settings would change"
    else:
        save(frozen,config)
    progress("configuration_frozen",best_fixed_k=best,guard_threshold=chosen["threshold"])
    write_analysis({g:groups[g] for g in ["tune","validation"]},candidates,evaluated,data,config)
    if stage=="practice":
        progress("practice_complete")
        return
    pairs=[]
    for qid in groups["dev"]:
        pairs += [(qid,k) for k in [5,best,adaptive_k(candidates[qid]["scores"],**POLICY)["k"],
                                    adaptive_k(candidates[qid]["scores"],**GUARD_POLICY)["k"],
                                    guarded_k(candidates[qid]["scores"],threshold=chosen["threshold"],fixed_k=best,**GUARD_POLICY)["k"]]]
    evaluated.update(evaluate_pairs(pairs,candidates,data,gen,judge))
    report=write_analysis(groups,candidates,evaluated,data,config)
    progress("complete",queries=100,unique_answers=len(read_rows(OUT/"answers.jsonl")),groups=list(report["groups"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["prepare","practice","all"],default="all")
    args=parser.parse_args()
    try: run(args.stage)
    except Exception as exc:
        progress("failed",error_type=type(exc).__name__,message=str(exc)[:400])
        raise
