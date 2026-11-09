"""Recreate cutoff and buffer diagnostics without model loads or API calls."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[2] / "results/adaptive_depth_study"
dev = json.loads((root / "analysis.json").read_text())["groups"]["dev"]
rows = []
for baseline in ["fixed_5", "fixed_10"]:
    base = dev[baseline]
    for policy in ["fixed_5", "fixed_10", "adaptive", "gap_only"]:
        r = dev[policy]
        rows.append(dict(baseline=baseline, policy=policy, mean_k=r["mean_k"],
            answer_score=r["mean_score"], quality_difference=r["mean_score"]-base["mean_score"],
            input_tokens=r["mean_prompt_tokens"], output_tokens=r["mean_completion_tokens"],
            input_token_saving_pct=100*(1-r["mean_prompt_tokens"]/base["mean_prompt_tokens"]),
            downstream_ms=r["mean_downstream_ms"],
            observed_time_saving_pct=100*(1-r["mean_downstream_ms"]/base["mean_downstream_ms"]),
            recall=r["mean_recall"]))
with (root / "retrieval_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
queries = [r for r in json.loads((root / "diagnostics.json").read_text()) if r["group"]=="dev"]
policies = {"fixed_5":lambda r:5, "fixed_10":lambda r:10,
            "adaptive":lambda r:r["adaptive"]["k"],
            "gap_only":lambda r:min(20,max(1,r["adaptive"]["raw_k"]))}
cutoffs = {}
for policy, choose in policies.items():
    cutoffs[policy] = dict(questions=len(queries), mean_k=mean(choose(r) for r in queries),
        zero_known_relevant_in_context=sum(not any(rank<=choose(r) for rank in r["relevant_ranks_top100"]) for r in queries),
        known_relevant_between_cutoff_and_rank20=sum(any(choose(r)<rank<=20 for rank in r["relevant_ranks_top100"]) for r in queries))
(root / "cutoff_diagnostics.json").write_text(json.dumps(cutoffs, indent=2))
counts = dict(questions=len(queries),buffer_adds_known_relevant_evidence=0,
    buffer_rescues_zero_hit_context=0,buffer_score_higher=0,buffer_score_equal=0,buffer_score_lower=0)
rescued=[]
for r in queries:
    bare=policies["gap_only"](r); buffered=policies["adaptive"](r)
    ranks=r["relevant_ranks_top100"]
    adds=any(bare<rank<=buffered for rank in ranks)
    rescue=not any(rank<=bare for rank in ranks) and any(rank<=buffered for rank in ranks)
    counts["buffer_adds_known_relevant_evidence"]+=adds
    counts["buffer_rescues_zero_hit_context"]+=rescue
    difference=r["scores_by_k"][str(buffered)]-r["scores_by_k"][str(bare)]
    label="higher" if difference>1e-9 else "lower" if difference< -1e-9 else "equal"
    counts["buffer_score_"+label]+=1
    if rescue:
        rescued.append(dict(query_id=r["query_id"],bare_k=bare,buffered_k=buffered,
                            relevant_ranks_top100=ranks,score_difference=difference))
(root / "buffer_effect.json").write_text(json.dumps(dict(summary=counts,rescued_queries=rescued),indent=2))
print("Saved comparison, cutoff diagnostics and buffer effects under", root)
