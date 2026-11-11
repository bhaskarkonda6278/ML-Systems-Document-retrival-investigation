# Expanded all-methods study: code available, results not verified

Updated 11 November 2026. This file distinguishes implemented code from measured
results; it is not a completion record.

## Completed evidence already in this repository

- `adaptive_depth_study/`: 30 tuning, 20 validation and 50 dev questions.
- `combined_gpu_dev125_v3/`: completed 125-dev, four-system, three-repeat study
  (1,500 trials). It compares fixed5 and zero-buffer retrieval under original
  ASR and KV-cache-plus-trimming ASR.

## Expanded comparison

`rag/extensions/combined_study/run_all_methods.py` adds fixed1, fixed3, fixed10,
fixed20 and buffered Adaptive-k to the four-system runner. The matrix has two
ASR modes and seven retrieval policies, giving 14 configurations. Full scope is
125 dev questions x 14 configurations x three repeats = 5,250 fresh trials,
plus hosted judging requests. Prior 1,500-trial measurements remain a separate
replication; their latency observations are not silently pooled with later runs.

The new runner retains GPU query embedding, the fixed B=0/B=5 policies, original
generation/judge settings, fixed baseline-ASR judging questions and cached-output
resumption safeguards. It uses a new protocol version and hashes its own source.
All 125 dev questions have previously been evaluated: `outside_initial50` names
a historical subgroup, not an untouched holdout.

The runner and batch file were reconstructed locally from the exact preparation
code supplied in the conversation. The 11 November SSH connection timed out, so
byte-for-byte agreement with the latest cluster copy and any newer run results
could not be checked during this publication. No expanded run is claimed complete.

## Commands (repository root)

Offline tests:

```bash
python -m unittest discover -s rag/extensions/combined_study -p test_all_methods.py
```

On an already allocated H200 MIG GPU, after activating the existing environments
and setting `RAG_ASR_PYTHON`, start/resume the small check:

```bash
python -u rag/extensions/combined_study/run_all_methods.py \
  --split practice --limit 2 --repeats 1 \
  --output rag/results/all_methods_smoke_v1
```

This should finish 28 trials. The older `validate_smoke.py` assumes four systems
and eight trials; do not use it to validate this expanded check. Verify protocol,
completion, all 14 systems, nonempty answers and judgments before a full run.
If the existing protocol differs, stop and inspect it rather than overwrite it.

Batch alternative (do not submit a duplicate of an interactive run):

```bash
mkdir -p rag/results/all_methods_smoke_v1
sbatch --time=01:00:00 \
  --output=rag/results/all_methods_smoke_v1/slurm-%j.log \
  rag/extensions/combined_study/run_all_methods.sbatch \
  --split practice --limit 2 --repeats 1 \
  --output rag/results/all_methods_smoke_v1
```

Run the full matrix only after validation, using a fresh, clearly named results
directory and an allowed time limit. Do not retune policies based on dev scores.
