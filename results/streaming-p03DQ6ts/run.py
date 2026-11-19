import hashlib
import json
import os
import random
import re
import socket
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from generator import ELMGenerator

root = Path.cwd()
out = Path(__file__).resolve().parent
seed = 3735060
rng = random.Random(seed)
records = []
clock = time.perf_counter

def save(name, obj):
    (out / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def log(row):
    with (out / "calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    records.append(row)

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def failure(exc):
    return {"type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)}

def main():
    gen = ELMGenerator()
    api = gen.client.chat.completions

    # Use the actual generator to construct each request. Intercept the SDK
    # call before transmission so both modes receive the same arguments.
    class Captured(Exception):
        pass

    def request_for(prompt):
        with patch.object(api, "create", side_effect=Captured) as observed:
            try:
                gen.generate(prompt)
            except Captured:
                pass
        if observed.call_count != 1 or observed.call_args.args:
            raise RuntimeError("Unexpected generator call signature")
        return dict(observed.call_args.kwargs)

    def call(prompt, mode, **labels):
        req = request_for(prompt)
        if mode == "stream":
            req["stream"] = True
        row = dict(labels, mode=mode, prompt_sha256=digest(prompt),
                   model=req.get("model"), temperature=req.get("temperature"),
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   answer="", first_text_ms=None, completion_ms=None,
                   finish_reason=None, ok=False)
        start = clock()
        try:
            if mode == "stream":
                with api.create(**req) as chunks:
                    for chunk in chunks:
                        arrived = clock()
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        piece = choice.delta.content or ""
                        if piece:
                            if row["first_text_ms"] is None:
                                row["first_text_ms"] = (arrived - start) * 1000
                            row["answer"] += piece
                        if choice.finish_reason:
                            row["finish_reason"] = choice.finish_reason
                    done = clock()
            else:
                result = api.create(**req)
                done = clock()
                choice = result.choices[0]
                row["answer"] = choice.message.content or ""
                row["finish_reason"] = choice.finish_reason
                # The non-streaming client receives the whole answer at return.
                if row["answer"]:
                    row["first_text_ms"] = (done - start) * 1000
            if not row["answer"].strip() or not row["finish_reason"]:
                raise ValueError("Missing answer or terminal finish reason")
            row["completion_ms"] = (done - start) * 1000
            row["ok"] = True
        except Exception as exc:
            row["failure"] = failure(exc)
            row["elapsed_until_failure_ms"] = (clock() - start) * 1000
        log(row)
        return row

    smoke = "Answer in one short sentence: what is the capital of France?"
    base = request_for(smoke)
    settings = {
        "host": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "seed": seed, "repetitions": 3, "maximum_questions": 2,
        "model": base.get("model"), "temperature": base.get("temperature"),
        "endpoint_host": getattr(gen.client.base_url, "host", None),
        "sdk_max_retries": gen.client.max_retries,
        "generator_sha256": digest((root / "rag/src/generator.py").read_text()),
        "timing_boundary": "SDK request start to content arrival or completion",
    }
    save("settings.json", settings)
    for mode in ("baseline", "stream"):
        row = call(smoke, mode, phase="smoke")
        print("Smoke", mode, {
            k: row.get(k) for k in ("ok", "first_text_ms", "completion_ms", "failure")
        }, flush=True)
        if not row["ok"]:
            return

    # Read project outputs only. Skip previous runs of this test.
    candidates = []
    for folder in (root / "results", root / "rag"):
        for directory, dirs, names in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith((".", "streaming-", "stream-smoke-"))
                       and d not in {"__pycache__", "node_modules", "venv", "models"}]
            for name in sorted(names):
                p = Path(directory) / name
                if p.suffix.lower() in {".json", ".jsonl"}:
                    candidates.append(p)

    # Keep each file or explicit k=3/k=5 filename pair as its own experiment.
    # Conflicting prompt versions disqualify a question within that source.
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    inventory = []
    depth_pattern = re.compile(r"(?i)(?<![a-z])(?:top[_-]?k|top|k)[=_-]?([35])(?!\d)")

    def visit(obj, path, group, inherited_k=None):
        if isinstance(obj, list):
            for item in obj:
                visit(item, path, group, inherited_k)
            return
        if not isinstance(obj, dict):
            return
        depth = next((obj[key] for key in ("k", "top_k", "retrieval_k")
                      if key in obj), inherited_k)
        qid = next((obj[key] for key in ("question_id", "query_id", "qid")
                    if key in obj), None)
        prompt = obj.get("prompt")
        if (str(depth) in {"3", "5"} and isinstance(qid, (str, int))
                and isinstance(prompt, str) and prompt.strip()):
            groups[group][str(qid)][int(depth)].add((prompt, str(path)))
        for key, value in obj.items():
            if isinstance(value, (list, dict)):
                visit(value, path, group, depth)

    for p in sorted(candidates):
        rel = p.relative_to(root)
        note = {"file": str(rel)}
        inventory.append(note)
        if p.stat().st_size > 64 * 1024 * 1024:
            note["status"] = "over 64 MiB; inspect separately"
            continue
        marks = set(depth_pattern.findall(str(rel)))
        fallback = next(iter(marks)) if len(marks) == 1 else None
        group = depth_pattern.sub("kDEPTH", str(rel))
        try:
            with p.open(encoding="utf-8-sig") as f:
                items = (json.loads(line) for line in f if line.strip()) \
                    if p.suffix == ".jsonl" else [json.load(f)]
                for item in items:
                    if "fields" not in note:
                        sample = item[0] if isinstance(item, list) and item else item
                        if isinstance(sample, dict):
                            note["fields"] = list(sample)[:25]
                    visit(item, rel, group, fallback)
        except (OSError, ValueError, RecursionError) as exc:
            note["read_error"] = type(exc).__name__
            groups.pop(group, None)
    save("inventory.json", inventory)

    eligible = {}
    for group, questions in groups.items():
        valid = {}
        for qid, depths in questions.items():
            if any(len(depths.get(k, set())) != 1 for k in (3, 5)):
                continue
            pair = {k: next(iter(depths[k])) for k in (3, 5)}
            if pair[3][0] != pair[5][0]:
                valid[qid] = pair
        if valid:
            eligible[group] = valid

    if len(eligible) != 1:
        print("Comparison paused. Eligible experiment sources:", list(eligible), flush=True)
        for item in inventory:
            if re.search(r"prompt|retriev|fixed|top.?k|generat|experiment", item["file"], re.I):
                print(item["file"], item.get("fields", []))
        print("Full file and field inventory:", out / "inventory.json")
        return

    group, valid = next(iter(eligible.items()))
    ids = rng.sample(sorted(valid), min(2, len(valid)))
    prompts = []
    for qid in ids:
        for k in (3, 5):
            prompt, source = valid[qid][k]
            req = request_for(prompt)
            prompts.append({"qid": qid, "k": k, "source": source, "prompt": prompt,
                            "messages": req["messages"], "sha256": digest(prompt)})
    save("prompts.json", prompts)

    pairs = [(qid, k, rep) for qid in ids for k in (3, 5) for rep in range(1, 4)]
    rng.shuffle(pairs)
    schedule = []
    for qid, k, rep in pairs:
        modes = ["baseline", "stream"]
        rng.shuffle(modes)
        for mode in modes:
            schedule.append(dict(qid=qid, k=k, repetition=rep, mode=mode))
    save("schedule.json", schedule)
    print("Source:", group, "| questions:", ids, "| fresh calls:", len(schedule), flush=True)
    for order, entry in enumerate(schedule, 1):
        prompt = valid[entry["qid"]][entry["k"]][0]
        row = call(prompt, phase="comparison", order=order, **entry)
        print(f'{order}/{len(schedule)} k={entry["k"]} {entry["mode"]} '
              f'{"OK" if row["ok"] else row["failure"]}', flush=True)

    print("Paired medians in milliseconds (baseline minus stream).")
    for k in (3, 5):
        deltas = defaultdict(list)
        for qid in ids:
            for rep in range(1, 4):
                pair = {r["mode"]: r for r in records if r["ok"]
                        and r.get("qid") == qid and r.get("k") == k
                        and r.get("repetition") == rep}
                if len(pair) == 2:
                    for metric in ("first_text_ms", "completion_ms"):
                        deltas[metric].append(pair["baseline"][metric] - pair["stream"][metric])
        print("k =", k, "| complete pairs =", len(deltas["first_text_ms"]),
              {m: round(statistics.median(v), 1) for m, v in deltas.items() if v})

try:
    main()
except Exception as exc:
    save("run_failure.json", failure(exc))
    print("Run stopped:", failure(exc))
finally:
    print("Saved run directory:", out, flush=True)
