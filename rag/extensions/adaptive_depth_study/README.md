# Adaptive retrieval-depth study

This directory implements a bounded adaptation of Adaptive-k (Taguchi et al., EMNLP 2025, https://aclanthology.org/2025.emnlp-main.1017/), plus a simple confidence-based fallback. It is not a claim of a new algorithm or reproduction of the original benchmark results.

## Five work packages

1. Reproduce the selection mechanism: largest adjacent similarity gap within the first 90 of 100 ranked candidates, add five documents, clamp output to 1..20. Tests cover rank indexing, buffer, tail exclusion, bounds and fallback. The finite candidate pool and output bound are explicit course adaptations.
2. Expand evaluation: exclude the original five diagnostic practice queries, randomly select 50 new practice queries and split them into 30 tuning / 20 validation. Select a separate 50-query dev subset (50/125 coverage). Seeds and IDs are saved before inference.
3. Compare fixed k=1,3,5,10,20 and the adaptive policy on practice. The fixed count with highest tuning quality (tokens break ties) is the competitive fixed baseline.
4. Diagnose evidence loss and answer failure with saved qrels coverage, relevant ranks, score curves and judge explanations. These labels are never passed to an inference-time selector.
5. Evaluate a safeguard: separately report the bare B=0 gap rule; accept that bare cutoff only if the gap is sufficiently prominent and the count was not clipped, otherwise use the best tuning-fixed count. This permits counts below five while preserving the published B=5 baseline. Select the threshold on the 30 tuning queries by minimum input tokens within an operational .02 score tolerance. Freeze before inspecting validation/dev outcomes. A pure fixed fallback is permitted and must be reported as a negative result for adaptation if selected.

## Run

From the course repository root, on the head node:

```bash
mkdir -p rag/results/adaptive_depth_study
sbatch rag/extensions/adaptive_depth_study/run_study.sbatch --stage all
```

The batch script requests one H200 MIG 3g.71gb allocation, two CPU cores, 24 GB host memory and three hours. It uses the existing separate ASR/RAG environments and ASR_CUDA=none. No CUDA homework or supplied model file is changed.

To prepare cached inputs without API requests, use `--stage prepare`. To stop after practice and the frozen configuration, use `--stage practice`. Resume with the same command and protocol; successful generation and judgments are reused.

## Outputs

Everything is saved in `rag/results/adaptive_depth_study/`: protocol, sampled IDs, transcripts, top-100 rankings/vectors, answers, judgments, frozen parameters, diagnostics, paired bootstrap intervals and `STUDY_RESULTS.md`. Credentials are never stored in these outputs.

Generation retains the supplied prompt and temperature .1. Judging retains the supplied rubric with explicit JSON response mode and temperature zero. Each distinct prompt has one generation and judgment; policies producing identical prompts share the corresponding answer, judgment and API latency observation. At most 550 unique generation prompts are allowed, plus judging and bounded retries. Protocol v2 adds the bare-gap ablation before any API evaluation; the superseded v1 protocol, if created, is retained for audit.

Local retrieval timing uses two warmups and ten timed repetitions. Policy selection uses 100 timed repetitions. Reported downstream latency combines warm retrieval, policy selection, context construction and a single observed API generation time, excluding ASR, query embedding, loading and judging. It is not an independently repeated end-to-end benchmark. Hardware, packages and model names are recorded. Token counts do not establish monetary charges, ELM GPU usage or energy.

Paired 95% bootstrap intervals resample queries. They do not measure judge/generation reliability or establish non-inferiority. This study intentionally reports modest coverage and single model outputs rather than implying robust conclusions from replication that was not performed.

Validation and dev judgments cannot change policy settings. Dev relevance/quality evaluation is gated on the saved frozen configuration. The separately cached dev transcripts and vectors contain inputs, not outcome labels.
