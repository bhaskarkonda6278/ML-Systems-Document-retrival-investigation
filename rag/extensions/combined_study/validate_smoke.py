"""Gate larger runs on a completed two-query factorial smoke test."""
import json
import sys
from pathlib import Path
p=Path(sys.argv[1])
progress=json.loads((p/"progress.json").read_text())
assert progress["phase"]=="complete", "Smoke test incomplete"
rows=[json.loads(x) for x in (p/"trials.jsonl").read_text().splitlines() if x.strip()]
assert len(rows)==8 and len({r["query_id"] for r in rows})==2
assert all(sum(r["system"]==s for r in rows)==2 for s in ["B0","A","B","C"])
assert all(r["transcript"].strip() and not r["hit_max_asr_tokens"] for r in rows)
assert all(r["same_tokens_as_reference"] for r in rows if r["system"] in ["B0","B"])
assert all(r["k"]==5 for r in rows if r["system"] in ["B0","A"])
assert all(1<=r["k"]<=20 for r in rows)
if not progress["no_api"]:
 scores={r["judge_key"] for r in [json.loads(x) for x in (p/"judgments.jsonl").read_text().splitlines() if x.strip()]}
 assert all(r["answer"].strip() and r["prompt_tokens"]>0 and r["judge_key"] in scores for r in rows)
print("SMOKE PASSED:",p,flush=True)
