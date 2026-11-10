# Completed combined ASR and retrieval evaluation

Job 3708351; 125 dev questions; three repetitions for each of four configurations.
All 1,500 recorded trials completed. This evaluates fixed5 and the frozen zero-buffer
rule, not all fixed-depth and buffered methods. Additional factorial methods are
planned, not completed in this archive.

| System | ASR | Retrieval | Mean quality | Mean total seconds | Mean input tokens |
|---|---|---|---:|---:|---:|
| B0 | supplied PyTorch | fixed5 | 0.3447 | 4.997 | 1209.0 |
| A | KV cache + trim | fixed5 | 0.3427 | 4.814 | 1208.6 |
| B | supplied PyTorch | zero-buffer | 0.3296 | 3.695 | 605.0 |
| C | KV cache + trim | zero-buffer | 0.3302 | 3.593 | 597.4 |

These are observed means, not proof of preserved quality. Paired query-bootstrap
intervals against B0 are in analysis.json and permit meaningful losses. GPU
embedding is shared across all configurations; these total timings cannot be
directly substituted for the old CPU-embedding-excluded retrieval-only timings.
The original study's inspected 50 dev questions are marked; analysis.json also
reports the remaining 75 separately. Quality is averaged within query before
aggregation. Identical answer/question/evidence combinations share judgments,
while each recorded generation timing is a fresh request.

## Files

- protocol.json: question IDs, frozen controls, source hashes and settings.
- analysis.json: all-dev and previously-unseen-75 results and uncertainty.
- progress.json: completion record.
- asr_environment.json: recorded GPU/ASR environment.
- reference_questions.jsonl: fixed baseline-ASR questions used for judging.
- trial_metrics.jsonl: all 1,500 per-trial metrics, answer scores, flags and judge keys.
- per_query_summary.jsonl: 500 query/system summaries over three repetitions.

Raw trials.jsonl (including prompts/answers), judgments.jsonl and diagnostic logs
remain in /home/s3021085/DIY/edin-mls-26-autumn/rag/results/combined_gpu_dev125_v3/.
They are not included in this compact code-review archive. No credentials or
model/corpus payloads are included.
