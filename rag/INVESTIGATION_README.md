# Query-adaptive retrieval depth: investigation and results

Bhaskar Konda — s3021085. Results from the completed November 2026 study.

## Research question

**Can query-adaptive retrieval depth reduce language-model input-token usage and downstream response time while preserving answer quality, compared with fixed-depth retrieval?**

This investigation changes how many ranked documents are sent to the answer model. It evaluates the retrieval-to-answer portion of the spoken-question RAG pipeline. ASR optimisation and the combined full-pipeline evaluation are separate work and are not claimed here.

## Reference and implementation

The main reference is Taguchi, Maekawa and Bhutani, [Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k](https://aclanthology.org/2025.emnlp-main.1017/), EMNLP 2025. [Official implementation](https://github.com/megagonlabs/adaptive-k-retrieval).

Our bounded adaptation ranks 100 course documents, finds the largest adjacent cosine-score gap within the first 90 candidates, selects the earliest maximum after rank `r`, and keeps `min(20, max(1, r + B))` records.

- **Fixed controls:** k = 1, 3, 5, 10, 20 on practice. The supplied baseline is k=5; tuning selected k=10 as the stronger fixed comparator.
- **Buffered Adaptive-k:** B=5, following the paper's buffer setting.
- **Zero-buffer ablation:** B=0, isolating the effect of removing that buffer.

This adapts the method to a different dataset, model and bounded candidate pool; it is not an exact reproduction of the paper's experiments. B=5 implies at least six documents, so it cannot reduce prefix-context tokens against k=5. An exploratory guarded fallback was also tested, but is our extension and is excluded from the paper-based primary comparison below.

Evidence diagnostics are inspired by [Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://arxiv.org/abs/2411.06037). They are not a replication of its evaluator: known-relevant-document coverage does not prove answer sufficiency.

## Experimental design

The course provides 24,000 corpus records, 350 practice questions and 125 dev questions. An initial five-question fixed-k pilot preceded the larger study and is excluded from its practice sample.

| Subset | Questions | Purpose |
|---|---:|---|
| Practice tuning | 30 | Select fixed comparator and exploratory settings |
| Practice validation | 20 | Check generalisation without retuning |
| Dev sample | 50 | Evaluate frozen choices |

Sampling seed: 20261105. The 100-question scope was an API/runtime budget choice, not a power-based sample-size justification. The completed study saved 508 generated answers across distinct prompts, including exploratory policies. It does not cover all 125 dev questions.

Policies share transcripts, embeddings, corpus, ranked candidates, prompt construction and generation settings. ASR preparation used `ASR_CUDA=none` on an H200 MIG 3g.71gb allocation. Generation and judging use hosted `utter-project/EuroLLM-22B-Instruct-2512`, at temperatures 0.1 and 0 respectively, with JSON-mode judgments. Each distinct prompt has one generation and one judgment; identical contexts reuse cached outputs.

Answer quality is the sum of correctness, grounding and completeness (each 0–4), divided by 12. It is not classification accuracy or a coursework mark. Input/output tokens come from API usage.

Downstream time combines warm retrieval median (10 repeats), policy-selection median (100 repeats), context construction and a single observed generation API time. It excludes ASR, embedding, model loading and judging. The generation measurement includes remote-service/network effects; it is not server GPU time.

## Main dev results

Means over the same 50 dev questions, using original judgments:

| Policy | Mean k | Answer score | Input tokens | Output tokens | Downstream seconds | Mean relevant-document recall |
|---|---:|---:|---:|---:|---:|---:|
| Fixed k=5 | 5.00 | 0.3633 | 1,156.36 | 267.20 | 4.374 | 0.4066 |
| Fixed k=10 | 10.00 | 0.3650 | 2,454.08 | 258.10 | 4.363 | 0.4501 |
| Adaptive-k, B=5 | 7.30 | 0.3250 | 1,798.14 | 268.18 | 4.420 | 0.4334 |
| Gap rule, B=0 | 2.30 | 0.3417 | 558.08 | 180.40 | 2.933 | 0.3112 |

Relative to supplied k=5, the zero-buffer rule saves **51.7% of input tokens** and has **32.9% lower observed mean downstream time**, with a score decrease of **0.0217**. Buffered Adaptive-k instead uses **55.5% more input tokens**, with a lower observed score and similar mean time.

Against k=10, buffered Adaptive-k saves 26.7% of input tokens; zero-buffer selection saves 77.3%. Fixed k=5 uses 52.9% fewer input tokens than k=10, with a nearly identical observed mean score. These comparisons do not establish quality equivalence.

## Why the trends matter

| Policy | Questions with at least one known relevant document |
|---|---:|
| Fixed k=5 | 29/50 |
| Fixed k=10 | 30/50 |
| Adaptive-k, B=5 | 30/50 |
| Gap rule, B=0 | 23/50 |

The bare rule often stops early: for 22 questions, a known relevant record exists after its cutoff but within rank 20. A similarity discontinuity does not certify that later documents are irrelevant.

Adding the five-document buffer supplies additional relevant evidence for 15 questions and rescues seven contexts with no known relevant record. Of those seven, three receive higher answer scores and four remain unchanged. Across all 50 questions, buffering improves scores for 10, ties for 28 and lowers scores for 12. Better evidence coverage therefore does not automatically yield a better judged answer. These are observations, not proof that extra context causes distraction.

Document count, token count and latency are related but distinct. Document lengths vary; response time also depends on output length, network and service behaviour. Zero-buffer outputs average 180 tokens versus 267 for k=5, which may contribute to lower latency. Its speed difference cannot be attributed solely to shorter input. Smaller k does not avoid the shared full-corpus scoring operation.

## Uncertainty and conclusion

Paired query-bootstrap 95% intervals for mean quality differences versus k=5 are:

- k=10: +0.0017, interval [-0.0583, +0.0600].
- Buffered Adaptive-k: -0.0383, interval [-0.1267, +0.0450].
- Zero-buffer rule: -0.0217, interval [-0.1067, +0.0567].

Intervals include zero but also allow meaningful losses; absence of a clear difference does not establish non-inferiority. They condition on single saved generations/judgments and omit their rerun variability. The generator and judge are the same model.

A separate audit of 10 randomly sampled dev questions rejudged 40 saved answers three times. Ten answers had varying scores. The audit is saved separately and paused; primary results retain their original scores. Repeated consistency would not by itself establish judging accuracy.

**Conclusion:** fixed k=5 is an attractive practical default in the observed quality–token comparison. Zero-buffer selection offers stronger efficiency savings with an unresolved quality trade-off. Buffered Adaptive-k does not improve the observed trade-off against k=5 in this setting. No tested adaptive policy has demonstrated preserved answer quality, and no combined audio-to-answer speedup is claimed.

## Code and saved outputs

From the repository root:

```bash
# Policy tests; no model/API calls
python -m unittest discover -s rag/extensions/adaptive_depth_study -p test_policies.py

# Rebuild comparisons and evidence diagnostics from saved outputs; no API calls
python rag/extensions/adaptive_depth_study/analyse_saved_results.py

# Build tables and plots; needs matplotlib/numpy and saved study outputs
python rag/extensions/adaptive_depth_study/build_report_artifacts.py
```

The original suite lives in `rag/extensions/adaptive_depth_study/`; the fixed-k pilot is `rag/extensions/fixed_k_pilot.py`. `rag/extensions/judge_audit.py` makes additional hosted judging requests if explicitly run; it is not required for the primary analysis.

Cluster outputs are under `/home/s3021085/DIY/edin-mls-26-autumn/rag/results/adaptive_depth_study/`: protocol, frozen configuration, transcripts, candidates, answers, judgments, per-policy metrics, `analysis.json`, `STUDY_RESULTS.md`, comparison CSV, cutoff diagnostics, buffer effects and figures. Audit files are under `judge_audit/`, with a separate archive. Ordinary generated results are ignored by Git and are not all included merely by cloning this code repository. Never commit `.env`, API keys, passwords, model weights or downloaded course assets.

Remaining work includes fuller dev coverage if resources permit, more robust timing, systematic per-query analyses and combined evaluation with optimised ASR. Policies must remain frozen during final evaluation; do not tune on already inspected dev outcomes.