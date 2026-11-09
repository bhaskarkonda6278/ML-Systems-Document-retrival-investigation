import json
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path("rag").resolve()))
from src.evaluation.answer_judge import Judge, build_public_gold

root = Path("rag/results/adaptive_depth_study")
out = root / "judge_audit"
out.mkdir(exist_ok=True)

def read(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

dev_ids = sorted({
    row["query_id"]
    for row in read(root / "transcripts.jsonl")
    if row["study_group"] == "dev"
})
selected = sorted(random.Random(20261107).sample(dev_ids, 10))
manifest = {"seed": 20261107, "query_ids": selected, "repeats": 3}

manifest_path = out / "manifest.json"
if manifest_path.exists():
    read_manifest = json.loads(manifest_path.read_text())
    assert read_manifest == manifest, "Existing audit uses different settings"
else:
    manifest_path.write_text(json.dumps(manifest, indent=2))

answers = {
    row["request_hash"]: row
    for row in read(root / "answers.jsonl")
    if row["query_id"] in selected
}
original = {
    row["request_hash"]: row
    for row in read(root / "judgments.jsonl")
}

missing = set(answers) - set(original)
if missing:
    raise RuntimeError(f"{len(missing)} answers lack original judgments")

gold = build_public_gold(Path("rag/data/rag_corpus"), "dev")
results_path = out / "repeated_judgments.jsonl"
results = read(results_path) if results_path.exists() else []
done = {(row["request_hash"], row["repeat"]) for row in results}

jobs = [
    (key, repeat)
    for key in sorted(answers)
    for repeat in range(1, 4)
]
random.Random(20261108).shuffle(jobs)
print(f"Selected 10 questions; {len(jobs) - len(done)} judging requests remain.")

judges = {}
for key, repeat in jobs:
    if (key, repeat) in done:
        continue

    answer = answers[key]
    model = original[key]["model"]
    if model not in judges:
        judges[model] = Judge(model)

    score = judges[model].evaluate(
        answer["transcript"],
        answer["answer"],
        gold[answer["query_id"]]["gold_documents"],
    )
    row = {
        "request_hash": key,
        "query_id": answer["query_id"],
        "k": answer["k"],
        "repeat": repeat,
        "model": model,
        **score,
        "score": sum(
            score[name]
            for name in ["correctness", "grounding", "completeness"]
        ) / 12,
    }
    with results_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")

    results.append(row)
    done.add((key, repeat))
    print(
        f"{len(done)}/{len(jobs)} | {row['query_id']} | "
        f"k={row['k']} | repeat={repeat} | score={row['score']:.3f}",
        flush=True,
    )

summary = []
for key, answer in sorted(answers.items()):
    repeats = [
        row["score"] for row in results
        if row["request_hash"] == key
    ]
    first = original[key]["normalised_answer_score"]
    summary.append({
        "query_id": answer["query_id"],
        "k": answer["k"],
        "original_score": first,
        "repeat_scores": repeats,
        "repeat_mean": mean(repeats),
        "range_including_original": (
            max([first] + repeats) - min([first] + repeats)
        ),
    })

(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("Finished:", out / "summary.json")
