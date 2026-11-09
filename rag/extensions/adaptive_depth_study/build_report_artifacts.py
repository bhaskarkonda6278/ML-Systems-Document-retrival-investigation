"""Create tables, plots and a review note exclusively from saved study outputs."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE=Path(__file__).resolve().parents[2]/"results/adaptive_depth_study"
OUT=SOURCE/"report_artifacts"
OUT.mkdir(parents=True,exist_ok=True)
analysis=json.loads((SOURCE/"analysis.json").read_text(encoding="utf-8"))
rows=[json.loads(x) for x in (SOURCE/"policy_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
protocol=json.loads((SOURCE/"protocol.json").read_text(encoding="utf-8"))
groups={q["query_id"]:q["study_group"] for q in protocol["queries"]}
config=analysis["frozen_config"]
best=config["best_fixed_k"]
labels={"adaptive":"Adaptive-k (buffer 5)","gap_only":"Largest gap (no buffer)","guarded":"Gap + fixed fallback",**{f"fixed_{k}":f"Fixed k={k}" for k in [1,3,5,10,20]}}
colors={"adaptive":"#7253a0","gap_only":"#d78c37","guarded":"#168578",**{f"fixed_{k}":"#527aa1" for k in [1,3,5,10,20]}}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.titleweight":"bold"})

flat=[]
for group,policies in analysis["groups"].items():
    for policy,summary in policies.items():
        flat.append({"group":group,"policy":policy,**{k:v for k,v in summary.items() if not isinstance(v,dict)}})
with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]))
    writer.writeheader(); writer.writerows(flat)

fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
for ax,group in zip(axes,["validation","dev"]):
    policies=analysis["groups"].get(group,{})
    for name,s in policies.items():
        ax.scatter(s["mean_prompt_tokens"],s["mean_score"],s=65,c=colors[name],marker="s" if name.startswith("fixed") else "o",label=labels[name],edgecolors="white",linewidths=.6)
    ax.set(title=f"{group.capitalize()}: quality and input tokens",xlabel="Mean ELM input tokens per query",ylabel="Mean normalized answer score")
    ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    if group=="dev": ax.legend(loc="best",fontsize=8)
fig.savefig(OUT/"quality_tokens.png",dpi=180)
plt.close(fig)

dev=analysis["groups"].get("dev",{})
names=list(dev)
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
y=np.arange(len(names))
for ax,key,title,xlabel in [(axes[0],"mean_downstream_ms","Observed downstream time","Seconds per query"),(axes[1],"mean_recall","Known relevant evidence coverage","Mean Recall@selected-k")]:
    values=[dev[n][key]/1000 if key.endswith("ms") else dev[n][key] for n in names]
    ax.barh(y,values,color=[colors[n] for n in names],height=.62)
    ax.set_yticks(y,[labels[n] for n in names]); ax.invert_yaxis()
    ax.set(title=title,xlabel=xlabel); ax.grid(axis="x",alpha=.15)
fig.savefig(OUT/"latency_coverage.png",dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.1),layout="constrained")
for name in ["adaptive","gap_only","guarded"]:
    counts=[r["k"] for r in rows if groups[r["query_id"]]=="dev" and r["policy"]==name]
    ax.hist(counts,bins=np.arange(.5,21.5),histtype="step",linewidth=2,label=labels[name],color=colors[name])
ax.set(xlabel="Selected document count",ylabel="Number of dev queries",title="Selected document counts on the held-out subset",xticks=[1,3,5,10,15,20])
ax.legend(fontsize=8); ax.grid(axis="y",alpha=.15)
fig.savefig(OUT/"selected_counts.png",dpi=180)
plt.close(fig)

devrows={n:{r["query_id"]:r for r in rows if groups[r["query_id"]]=="dev" and r["policy"]==n} for n in names}
reference=devrows[f"fixed_{best}"]
comparisons={}
for name,values in devrows.items():
    differences=np.array([r["score"]-reference[q]["score"] for q,r in sorted(values.items())])
    rng=np.random.default_rng(20261105)
    bootstrap=rng.choice(differences,size=(5000,len(differences)),replace=True).mean(axis=1)
    comparisons[name]={"quality_difference":float(differences.mean()),"ci95":np.quantile(bootstrap,[.025,.975]).tolist(),
                       "input_token_change_percent":100*(dev[name]["mean_prompt_tokens"]/dev[f"fixed_{best}"]["mean_prompt_tokens"]-1),
                       "downstream_change_percent":100*(dev[name]["mean_downstream_ms"]/dev[f"fixed_{best}"]["mean_downstream_ms"]-1),
                       "fallback_queries":sum(bool(r["decision"].get("fallback")) for r in values.values())}
(OUT/"paired_vs_tuned_fixed.json").write_text(json.dumps(comparisons,indent=2)+"\n",encoding="utf-8")

note=["# Five-step retrieval investigation: completion record", "",
      "## Work completed", "",
      "1. Implemented and unit-tested the published largest-gap + five-document buffer selection mechanism, explicitly adapted to a top-100 candidate pool and output range 1..20. The bare largest-gap rule is reported separately.",
      "2. Selected 50 new practice queries excluding the original five; preregistered a 30/20 tuning-validation partition and a separate 50-of-125 dev sample.",
      "3. Compared fixed k values and adaptive policies using shared cached transcripts, vectors, generator prompt/settings and judge rubric/JSON mode.",
      "4. Saved per-query quality, evidence coverage, relevant ranks, token usage, stage timings and judge explanations for failure analysis.",
      "5. Tuned a simple gap-prominence fallback only on 30 tuning queries, froze it before validation/dev analysis, then measured it on held-out subsets.", "",
      f"Frozen best fixed count: **k={best}**. Guard threshold: **{config['threshold']}**. The .02 tuning score tolerance is an operational choice, not a proof of non-inferiority.", "",
      "## Held-out dev comparison", "",
      "| Policy | n | Mean score | Input tokens | Mean k | Downstream s | Evidence recall |", "|---|---:|---:|---:|---:|---:|---:|"]
for name,s in dev.items():
    note.append(f"| {labels[name]} | {s['queries']} | {s['mean_score']:.4f} | {s['mean_prompt_tokens']:.1f} | {s['mean_k']:.2f} | {s['mean_downstream_ms']/1000:.3f} | {s['mean_recall']:.4f} |")
note += ["",f"## Paired quality differences versus tuning-selected fixed k={best}",""]
for name,s in comparisons.items():
    note.append(f"- {labels[name]}: mean difference {s['quality_difference']:+.4f}; paired query-bootstrap 95% interval [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]; input-token change {s['input_token_change_percent']:+.1f}%; observed downstream-time change {s['downstream_change_percent']:+.1f}%.")
note += ["", "## Interpretation", "",
         "These differences are conditional on one generated answer and one judgment per distinct prompt. Query bootstrap intervals do not measure judge or generation reliability. Hosted API variability, output length and shared prompt observations limit latency claims. Count repeated local retrieval timings separately from single API observations. No ASR speed-up, combined-system speed-up, monetary cost or ELM GPU-energy reduction is established by this retrieval-only study.", "",
         "The buffer-five policy cannot select fewer than six documents. It is a published comparison baseline, not a mechanism capable of reducing prefix-context tokens below fixed k=5. The guarded bare rule can choose fewer documents, but tuning quality constraints can select a constant fixed fallback; report such collapse honestly.", "",
         "## Largest dev losses for the guarded rule", ""]
losses=sorted(devrows["guarded"].items(),key=lambda item:item[1]["score"]-reference[item[0]]["score"])
for qid,row in losses[:5]:
    other=reference[qid]
    note.append(f"- {qid}: guard k={row['k']}, score={row['score']:.3f}, recall={row['recall']:.3f}; fixed k={best}, score={other['score']:.3f}, recall={other['recall']:.3f}; gap prominence={row['decision'].get('prominence',0):.4f}, fallback={row['decision'].get('fallback')}.")
(OUT/"INVESTIGATION_SUMMARY.md").write_text("\n".join(note)+"\n",encoding="utf-8")
print(json.dumps({"output":str(OUT),"best_fixed_k":best,"threshold":config["threshold"],"dev":dev,"paired":comparisons},indent=2))
