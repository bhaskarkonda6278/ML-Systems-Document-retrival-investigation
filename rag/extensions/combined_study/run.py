"""Four-system live pipeline evaluation with repeat timing and fixed judging questions."""
import faulthandler
import signal
faulthandler.enable()
if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1, all_threads=True)
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RAG = HERE.parents[1]
ROOT = RAG.parent
sys.path.insert(0, str(RAG))
sys.path.insert(0, str(HERE.parent / 'adaptive_depth_study'))
from policies import adaptive_k

SYSTEMS = {'B0': ('baseline', 'fixed5'), 'A': ('kv_trim', 'fixed5'),
           'B': ('baseline', 'gap0'), 'C': ('kv_trim', 'gap0')}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')
        f.flush()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Worker:
    def __init__(self, out):
        self.log = (out / 'asr_worker.log').open('a')
        self.p = subprocess.Popen([os.environ['RAG_ASR_PYTHON'], '-u', str(HERE/'asr_worker.py'),
            '--asr-root', str(ROOT/'hw1-asr')], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)
        self.info = self.read()
        if not self.info.get('ready'):
            raise RuntimeError(self.info)

    def read(self):
        line = self.p.stdout.readline()
        if not line:
            code = self.p.wait(timeout=10)
            raise RuntimeError(f'ASR worker exited with return code {code}; inspect asr_worker.log')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    def transcribe(self, audio_path, mode):
        self.p.stdin.write(json.dumps({'audio_path':str(audio_path), 'mode':mode})+'\n')
        self.p.stdin.flush()
        return self.read()

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            self.p.wait(timeout=15)
        self.log.close()


def api_call(gen, prompt, out, trial):
    for attempt in range(3):
        start = time.perf_counter()
        try:
            response = gen.client.chat.completions.create(model=gen.model, temperature=0.1,
                messages=[{'role':'system','content':"Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."},
                          {'role':'user','content':prompt}])
            if not response.usage or not response.choices[0].message.content:
                raise ValueError('Missing answer or token usage')
            return response, (time.perf_counter()-start)*1000, attempt
        except Exception as exc:
            append(out/'api_errors.jsonl', {'trial':trial,'attempt':attempt,'error_type':type(exc).__name__})
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def analyse(out):
    import numpy as np
    trials = rows(out/'trials.jsonl')
    scores = {r['judge_key']: r['score'] for r in rows(out/'judgments.jsonl')}
    report = {}
    for group in ['all_dev', 'previously_unseen75']:
        subset = [r for r in trials if group == 'all_dev' or not r['seen_in_previous_study']]
        if not subset:
            continue
        report[group] = {}
        baseline = {}
        for r in subset:
            if r['system'] == 'B0':
                baseline.setdefault(r['query_id'], []).append(scores[r['judge_key']])
        for system in SYSTEMS:
            values = [r for r in subset if r['system'] == system]
            if not values:
                continue
            per_query = {}
            for r in values:
                per_query.setdefault(r['query_id'], []).append(scores[r['judge_key']])
            differences = [statistics.mean(v)-statistics.mean(baseline[q]) for q,v in per_query.items() if q in baseline]
            rng = np.random.default_rng(20261109)
            ci = np.quantile(rng.choice(differences,(5000,len(differences)),replace=True).mean(axis=1),[.025,.975]).tolist()
            report[group][system] = {'queries':len(per_query),'trials':len(values),
                'mean_score':statistics.mean(statistics.mean(v) for v in per_query.values()),
                'quality_difference_vs_B0':statistics.mean(differences),'paired_query_ci95':ci,
                'retry_trials':sum(r['api_retries']>0 for r in values),
                **{f'mean_{key}':statistics.mean(r[key] for r in values) for key in
                   ['asr_ms','asr_inference_ms','embedding_ms','retrieval_ms','selection_ms','context_ms',
                    'generation_ms','end_to_end_ms','prompt_tokens','completion_tokens','k','recall','peak_gpu_gib']},
                'median_end_to_end_ms':statistics.median(r['end_to_end_ms'] for r in values),
                'p95_end_to_end_ms':float(np.quantile([r['end_to_end_ms'] for r in values],.95))}
    save(out/'analysis.json', report)
    print(json.dumps(report,indent=2),flush=True)


def run(args):
    from dotenv import load_dotenv
    load_dotenv(RAG/'.env')
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    from src.context_builder import SimpleContextBuilder
    from src.generator import ELMGenerator
    from src.evaluation.answer_judge import Judge, build_public_gold, RUBRIC_VERSION, JUDGE_SYSTEM_PROMPT
    data = CourseData(RAG/'data/rag_corpus')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.analyse_only:
        analyse(out)
        return
    old = json.loads((RAG/'results/adaptive_depth_study/protocol.json').read_text())
    seen = {q['query_id'] for q in old['queries'] if q['study_group']=='dev'}
    items = sorted(data.audio_items(args.split),key=lambda r:r['query_id'])
    if args.limit:
        items = random.Random(20261109).sample(items,args.limit)
        items.sort(key=lambda r:r['query_id'])
    gen = ELMGenerator() if not args.no_api else None
    judge = Judge(os.getenv('ELM_JUDGE_MODEL') or gen.model) if gen else None
    if gen:
        gen.client.max_retries = 0
    sources = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [HERE/'run.py',HERE/'asr_worker.py',HERE/'partner_asr/kv_cache.py',
                  HERE/'partner_asr/encoder_trim.py',HERE.parent/'adaptive_depth_study/policies.py',
                  ROOT/'hw1-asr/glm_asr_cuda/model.py',ROOT/'hw1-asr/glm_asr_cuda/ops.py']}
    protocol = {'version':'combined-v1','split':args.split,'queries':[q['query_id'] for q in items],
        'repeats':args.repeats,'systems':SYSTEMS,'seed':20261109,'source_hashes':sources,
        'generator_model':gen.model if gen else None,'judge_model':judge.model if judge else None,
        'endpoint':str(gen.client.base_url) if gen else None,'rubric':RUBRIC_VERSION,
        'question_mode':'fixed baseline full-audio transcript, prepared before generation',
        'no_api':args.no_api,'timing':'live warm persistent pipeline; model loading and judging excluded; IPC and retries included in wall time',
        'retrieval':'fixed5 uses supplied top5; gap0 retrieves top100 then frozen B=0 largest-gap policy',
        'generation_temperature':0.1,'judge_temperature':0,'max_asr_tokens':100,
        'prior_dev_exposure':'50 dev questions previously inspected; remaining75 reported separately',
        'max_successful_generations':len(items)*len(SYSTEMS)*args.repeats}
    protocol = json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==protocol, 'Protocol changed: use a new output directory'
    else:
        save(out/'protocol.json',protocol)
    builder = SimpleContextBuilder()
    print('PHASE: starting ASR worker', flush=True)
    worker = Worker(out)
    save(out/'asr_environment.json',worker.info)
    print('ASR ready:',json.dumps(worker.info),flush=True)
    try:
        # Warm all paths without hosted requests; exclude from trial metrics.
        print('PHASE: ASR warm-up', flush=True)
        for mode in ['baseline','kv_trim']:
            for _ in range(2):
                worker.transcribe(data.data_dir/items[0]['audio_file'],mode)
        print('PHASE: loading GPU query encoder', flush=True)
        encoder = LocalEncoder(RAG/'models/enc_a',device='cuda')
        print('PHASE: query encoder ready', flush=True)
        retriever = ExactDenseRetriever(data.data_dir/'embeddings/corpus_enc_a.npy',data.data_dir/'embeddings/corpus_enc_a_ids.json')
        for _ in range(2):
            vec = encoder.encode('warm up the question encoder')
            retriever.retrieve(vec,k=100)
        print('PHASE: preparing fixed evaluation questions', flush=True)
        references = {r['query_id']:r for r in rows(out/'reference_questions.jsonl')}
        for q in items:
            if q['query_id'] not in references:
                r = worker.transcribe(data.data_dir/q['audio_file'],'baseline')
                value = {'query_id':q['query_id'],'transcript':r['transcript'],'tokens':r['tokens']}
                append(out/'reference_questions.jsonl',value)
                references[q['query_id']]=value
        faulthandler.cancel_dump_traceback_later()
        print("PHASE: initialization complete", flush=True)
        gold = build_public_gold(data.data_dir,args.split)
        qrels = data.load_qrels(args.split)
        judgments = {r['judge_key']:r for r in rows(out/'judgments.jsonl')}
        complete = {r['trial_id']:r for r in rows(out/'trials.jsonl')}
        def judge_trial(row):
            key=row['judge_key']
            if key not in judgments:
                qid=row['query_id']
                score=judge.evaluate(references[qid]['transcript'],row['answer'],gold[qid]['gold_documents'])
                result={'judge_key':key,'query_id':qid,'model':judge.model,**score,
                        'score':sum(score[k] for k in ['correctness','grounding','completeness'])/12}
                append(out/'judgments.jsonl',result);judgments[key]=result
        # Resume outstanding judgments before issuing more generation requests.
        if judge:
            for row in complete.values():
                judge_trial(row)
        rng = random.Random(20261109)
        for repeat in range(1,args.repeats+1):
            order = list(items);rng.shuffle(order)
            for q in order:
                configs=list(SYSTEMS);rng.shuffle(configs)
                for system in configs:
                    trial=f"{q['query_id']}:{repeat}:{system}"
                    if trial in complete:
                        continue
                    mode,policy=SYSTEMS[system]
                    start=time.perf_counter()
                    t=time.perf_counter();asr=worker.transcribe(data.data_dir/q['audio_file'],mode)
                    asr_ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter();vec=encoder.encode(asr['transcript']);embedding_ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter();ret=retriever.retrieve(vec,k=5 if policy=='fixed5' else 100);retrieval_ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter();decision={'k':5} if policy=='fixed5' else adaptive_k(ret['scores'],buffer=0,min_k=1,max_k=20,search_fraction=.9)
                    selection_ms=(time.perf_counter()-t)*1000
                    chosen=ret['record_ids'][:decision['k']]
                    t=time.perf_counter();context=builder.build(asr['transcript'],chosen,data);context_ms=(time.perf_counter()-t)*1000
                    if args.no_api:
                        response=None;elapsed=0;retries=0;answer=''
                    else:
                        response,elapsed,retries=api_call(gen,context['prompt'],out,trial)
                        answer=response.choices[0].message.content
                    end_to_end=(time.perf_counter()-start)*1000
                    key=digest({'model':judge.model if judge else None,'endpoint':str(judge.client.base_url) if judge else None,
                        'question':references[q['query_id']]['transcript'],'answer':answer,
                        'gold':gold[q['query_id']]['gold_documents'],'rubric':RUBRIC_VERSION,'system_prompt':JUDGE_SYSTEM_PROMPT})
                    row={'trial_id':trial,'query_id':q['query_id'],'system':system,'repeat':repeat,
                        'seen_in_previous_study':q['query_id'] in seen,'transcript':asr['transcript'],
                        'same_tokens_as_reference':asr['tokens']==references[q['query_id']]['tokens'],
                        'hit_max_asr_tokens':asr['hit_max_tokens'],'frames':asr['frames'],
                        'asr_ms':asr_ms,'asr_inference_ms':asr['inference_ms'],'peak_gpu_gib':asr['peak_gpu_gib'],
                        'embedding_ms':embedding_ms,'retrieval_ms':retrieval_ms,'selection_ms':selection_ms,
                        'context_ms':context_ms,'generation_ms':elapsed,'end_to_end_ms':end_to_end,
                        'answer':answer,'judge_key':key,'k':decision['k'],'decision':decision,
                        'record_ids':chosen,'prompt':context['prompt'],'prompt_hash':digest(context['prompt']),
                        'prompt_tokens':response.usage.prompt_tokens if response else 0,
                        'completion_tokens':response.usage.completion_tokens if response else 0,
                        'finish_reason':response.choices[0].finish_reason if response else 'no_api',
                        'response_model':response.model if response else None,'api_retries':retries,
                        'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        'recall':len(set(chosen)&set(qrels[q['query_id']]))/len(qrels[q['query_id']])}
                    append(out/'trials.jsonl',row);complete[trial]=row
                    if judge:
                        judge_trial(row)
                    progress={'phase':'running','completed':len(complete),'total':len(items)*4*args.repeats,'latest':trial}
                    save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if not args.no_api:
            analyse(out)
        save(out/'progress.json',{'phase':'complete','trials':len(complete),'queries':len(items),'no_api':args.no_api})
    finally:
        worker.close()


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--split',choices=['practice','dev'],default='dev')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--no-api',action='store_true')
    ap.add_argument('--analyse-only',action='store_true')
    args=ap.parse_args()
    if args.repeats<1 or args.limit<0:
        ap.error('repeats must be positive; limit nonnegative')
    run(args)
