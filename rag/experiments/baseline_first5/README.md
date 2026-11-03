# Initial audio-RAG baseline: five practice queries

Archived on 2026-11-03 from `/home/s3021085/DIY/edin-mls-26-autumn`.
This is a diagnostic run, not a full-split benchmark or final investigation result.

## Source and settings

- Code commit: `0762a4dda0bf4ce16e9250fdd92c72057ae8c030`.
- Cluster working tree was clean when the files were collected.
- Split: practice, first 5 of 350 queries; retrieval depth: default `k=5`.
- Query encoder: supplied `enc_a`, requested device: CPU.
- Retriever: supplied exact dense retriever; context: supplied simple context builder.
- ELM model configured at archive time: `utter-project/EuroLLM-22B-Instruct-2512`.
- Judge model recorded in results: the same EuroLLM model; one judgment per answer.
- The shell used the RAG environment and a GPU allocation on `saxa`, as shown in the run transcript. Exact dependency versions, ASR_CUDA setting, GPU allocation details and warm-up state were not captured in these result files. Do not infer them from later sessions.

## Commands used

From the repository root, in the prepared RAG environment and GPU allocation:

```bash
python rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output rag/results/baseline_first5.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/baseline_first5.jsonl --allow-subset --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/baseline_first5.jsonl --allow-subset --out-dir rag/results/baseline_first5_judge
```

For a rerun, choose new output paths to preserve existing results. ELM generation and judging require a private local API key. Do not commit `.env`.

## Archived files

- `baseline_first5.jsonl`: original transcripts, generated answers, selected document IDs and timings.
- `answer_judge_per_query.jsonl`: original three-dimensional scores and explanations.
- `answer_judge_summary.json`: original aggregate evaluation and coverage.
- `judge_cache.jsonl`: original cached judgments, retained for provenance.
- `manifest.json`: SHA-256 hashes of the four files copied from the cluster.

The original files under `rag/results/` remain unchanged. This experiment directory deliberately tracks a small selected result set while ordinary generated results remain ignored.

## Observed outcome and limitations

Mean normalized answer score: **0.3333333333**. Mean correctness: **1.6/4**;
grounding: **1.4/4**; completeness: **1.0/4**. Coverage: **5/350** practice queries.
This is a judge score, not an accuracy percentage or coursework mark.

The evaluator judges the ASR transcript against evidence reconstructed from public qrels. It does not independently assess audio transcription accuracy. The generator and judge use the same model; judgments can be mistaken or biased.

These files do not record input-token counts, retrieval scores, exact prompts, or separate compression/context-building timings. Recorded generation latency is client-observed API latency, not server GPU compute time. The initial timings are not a controlled repeated warm benchmark.

## Future comparisons

Reuse fixed transcripts, corpus, encoder, prompt template, generator and judge settings across variants. Tune fixed-k and adaptive policies on practice queries, then freeze settings before dev evaluation. Compare fixed document counts, adaptive counts, compression alone and their combination. Record token counts, stage timings, method overhead and per-query quality. This archive alone is insufficient to establish the best policy.
