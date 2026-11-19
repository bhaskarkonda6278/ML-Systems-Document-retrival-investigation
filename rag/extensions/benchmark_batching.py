"""Controlled offline retrieval comparison with cached ASR transcripts.

Query encoding is performed once and excluded from search timing. All variants
use the same enc_a vectors, k, queries and two-thread BLAS limit. Timings cover
validation, normalization, similarity, top-k and result formatting. Batch-fill
waiting and answer generation are not measured by this offline experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import tracemalloc

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.batched_retriever import BatchedExactDenseRetriever
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retriever import ExactDenseRetriever


def quality(results, qids, qrels):
    rows = []
    for result, qid in zip(results, qids):
        ranked, relevance = result["record_ids"], qrels[qid]
        rows.append({"query_id": qid, **{f"Recall@{k}": recall_at_k(ranked, relevance, k)
                     for k in (5, 10, 20, 100)},
                     "MRR@10": mrr_at_k(ranked, relevance, 10),
                     "NDCG@10": ndcg_at_k(ranked, relevance, 10)})
    return {key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key != "query_id"}


def execute(retriever, queries, batch_size, k):
    results, batches = [], []
    start = time.perf_counter()
    for offset in range(0, len(queries), batch_size):
        t0 = time.perf_counter()
        if batch_size == 1:
            results.append(retriever.retrieve(queries[offset], k=k))
        else:
            results.extend(retriever.retrieve_many(queries[offset:offset+batch_size], k=k)["results"])
        batches.append((time.perf_counter() - t0) * 1000)
    return results, (time.perf_counter() - start) * 1000, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], required=True)
    parser.add_argument("--transcript-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = CourseData(ROOT / "data/rag_corpus")
    items = data.audio_items(args.split)
    qids = [item["query_id"] for item in items]
    cached = {row["query_id"]: row["transcript"] for row in
              (json.loads(line) for line in args.transcript_cache.read_text().splitlines() if line.strip())}
    if any(qid not in cached for qid in qids):
        raise ValueError("The transcript cache must cover the complete selected split")
    encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
    # Use the supplied baseline's single-query encoding path, then freeze these
    # exact vectors. The experiment changes search batching, not encoder batching.
    queries = np.stack([encoder.encode(cached[qid]) for qid in qids])
    np.save(args.out_dir / "query_vectors.npy", queries)
    (args.out_dir / "query_ids.json").write_text(json.dumps(qids))
    embeddings = ROOT / "data/rag_corpus/embeddings"
    retriever = BatchedExactDenseRetriever(embeddings / "corpus_enc_a.npy", embeddings / "corpus_enc_a_ids.json")
    qrels = data.load_qrels(args.split)
    baseline, _, _ = execute(retriever, queries, 1, 100)
    baseline_quality = quality(baseline, qids, qrels)
    sizes = [1, 4, 16, 64, 128]
    timings = {size: [] for size in sizes}
    service = {size: [] for size in sizes}
    for size in sizes:
        execute(retriever, queries, size, 100)
    randomizer = random.Random(2026)
    for repeat in range(args.repeats):
        order = sizes.copy()
        randomizer.shuffle(order)
        for size in order:
            _, elapsed, batch_times = execute(retriever, queries, size, 100)
            timings[size].append(elapsed)
            service[size].extend(batch_times)
        print(f"Timing round {repeat+1}/{args.repeats}", flush=True)
    rows = []
    for size in sizes:
        results, _, _ = execute(retriever, queries, size, 100)
        metrics = quality(results, qids, qrels)
        identical = sum(a["record_ids"] == b["record_ids"] for a, b in zip(baseline, results))
        overlap = float(np.mean([len(set(a["record_ids"]) & set(b["record_ids"])) / 100
                                 for a, b in zip(baseline, results)]))
        tracemalloc.start()
        execute(retriever, queries, size, 100)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = statistics.median(timings[size])
        rows.append({"batch_size": size, "quality": metrics,
                     "identical_rankings": identical, "top100_overlap": overlap,
                     "median_split_ms": elapsed, "split_ms_repeats": timings[size],
                     "amortized_ms_per_query": elapsed / len(queries),
                     "queries_per_second": len(queries) * 1000 / elapsed,
                     "batch_service_ms_p50": float(np.percentile(service[size], 50)),
                     "batch_service_ms_p95": float(np.percentile(service[size], 95)),
                     "traced_peak_extra_mib": peak / 2**20,
                     "score_buffer_mib": min(size, len(queries)) * len(retriever.record_ids) * 4 / 2**20})
    from threadpoolctl import threadpool_info
    summary = {"split": args.split, "queries": len(queries), "k": 100,
               "corpus_shape": list(retriever.embeddings.shape),
               "baseline_quality": baseline_quality, "repeats": args.repeats,
               "blas_threadpools": threadpool_info(), "variants": rows,
               "limitations": ["Offline batch-fill waiting is excluded", "Encoding and ASR excluded from retrieval timing",
                               "Traced allocations exclude untracked BLAS workspaces", "ELM answer generation/judging require a key"]}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
