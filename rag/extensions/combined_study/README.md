# Combined ASR and retrieval evaluation

Four configurations: B0 = supplied PyTorch ASR + fixed5; A = partner's KV cache
and real-audio trim + fixed5; B = supplied ASR + frozen zero-buffer gap rule;
C = KV cache + trim + zero-buffer rule. Weights and submitted homework files
are unchanged. Partner helper files are preserved under `partner_asr/` with
archive/source hashes. Its model.py, ops.py and benchmark.py matched our checkout.

The frozen gap rule searches within the first 90 of 100 ranked candidates and
clamps depth to 1..20. Fixed5 uses the actual supplied top5 retrieval path;
policy overhead is included. Both use the same embedding model and corpus.

## Controls

- Full dev evaluation: all 125 questions, three fresh generation/timing trials
  for each of four configurations (1,500 successful requests plus bounded retries).
- The previously inspected 50 dev questions are marked. Report the remaining
  75 separately, acknowledging earlier exploratory exposure.
- Random query/configuration order in every repeat, fixed seed 20261109.
- Persistent ASR and GPU embedding models; two warm-ups; loading excluded.
- Every timed trial starts from audio and performs fresh ASR, embedding, retrieval,
  context construction and generation. No timing observations are shared across systems.
- Live end-to-end wall time includes preprocessing, worker IPC and generation
  retry/backoff delays; judge requests are outside the timer. Also record clean
  stage/API durations, retry counts, finish reasons and output length.
- Judge the same audio item against a fixed baseline ASR question, prepared
  before generation, and the same gold evidence across all configurations.
  This differs from the old study's per-answer ASR-question evaluation and
  makes ASR transcript changes assessable; it is not a human-transcript gold standard.
- Original rubric, same hosted model, generation temperature 0.1, JSON-mode judge
  temperature 0. Exact answer/question/evidence combinations share judgments;
  independently generated response timings are never reused.
- Average quality within each query before aggregating; paired query bootstrap
  (5,000 resamples) against B0. Intervals are not proof of non-inferiority and
  do not fully quantify systematic judge bias. No dev-driven policy changes.
- Tokens, GPU memory and times are resource proxies, not energy or billed cost.

## Run from repository root on the cluster

First run two practice questions without API calls:

```bash
sbatch --output=rag/results/combined_smoke_gpu/slurm-%j.log \
  rag/extensions/combined_study/run_combined.sbatch \
  --split practice --limit 2 --repeats 1 --no-api \
  --output rag/results/combined_smoke_gpu
```

Create output directories before submission. Inspect ASR transcripts, lengths,
token-cap flags and times before progressing to the hosted-API smoke test.
Use another output directory for an otherwise identical run without `--no-api`.

```bash
sbatch --output=rag/results/combined_gpu_dev125_v3/slurm-%j.log \
  rag/extensions/combined_study/run_combined.sbatch \
  --split dev --repeats 3 --output rag/results/combined_gpu_dev125_v3
```

Rerun the same command to resume; source hashes, model/settings and sample must
match the saved protocol. Never overwrite a completed output or change frozen
settings to improve dev scores. Inspect progress.json, trials.jsonl, judgments.jsonl,
asr_worker.log and analysis.json. Credentials stay in the private local rag/.env.

## Current verified setup (10 November 2026)

The successful runner loads `LocalEncoder(..., device="cuda")`; earlier CPU smoke
folders are diagnostic history. ASR uses the separate Python 3.11 torch 2.10/CUDA
12.8 environment, and the RAG GPU encoder uses Python 3.12 torch 2.14.1+cu130.
Both passed the isolated/integrated GPU checks. Periodic timed faulthandler dumps
were removed before the successful ELM smoke and full run; the earlier native
crash's root cause was not proved. Fatal-error diagnostics remain enabled.

Successful API smoke: `combined_gpu_api_check_v3`, job 3708271.
Completed full run: `combined_gpu_dev125_v3`, job 3708351, 125 questions,
three repetitions, four systems, 1,500 trials. The full job requested six hours.
Use `run_combined.sbatch` with a fresh output path for a new protocol; use the
identical code/output/settings to resume. Do not change settings to improve dev.
The obsolete hard-coded `pipeline_api.sbatch` and `gpu_pipeline.sbatch` wrappers
are not needed for reproduction.

Protocols and compact review outputs are committed under
`rag/experiments/combined_gpu_dev125_v3/`. Original full prompts, answer bodies,
judgments and logs remain under `rag/results/` on the cluster.
