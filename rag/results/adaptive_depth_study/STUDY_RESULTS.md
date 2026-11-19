# Adaptive retrieval depth: measured study

## Shared question

To what extent can ASR execution optimisations and query-adaptive retrieval depth, individually and combined, reduce the response time and resource usage of an audio-RAG system without lowering answer quality, compared with the supplied baseline?

This study evaluates the retrieval contribution only. ASR optimisation and combined full-pipeline measurements remain separate work.

## Results

| Group | Policy | n | Mean score | Mean input tokens | Mean k | Mean downstream s | Mean recall |
|---|---|---:|---:|---:|---:|---:|---:|
| tune | fixed_1 | 30 | 0.3194 | 275.8 | 1.00 | 2.092 | 0.2875 |
| tune | fixed_3 | 30 | 0.3139 | 774.9 | 3.00 | 3.005 | 0.5180 |
| tune | fixed_5 | 30 | 0.3667 | 1268.1 | 5.00 | 3.260 | 0.5574 |
| tune | fixed_10 | 30 | 0.3889 | 2449.1 | 10.00 | 4.079 | 0.6112 |
| tune | fixed_20 | 30 | 0.2833 | 4899.9 | 20.00 | 4.439 | 0.7056 |
| tune | adaptive | 30 | 0.3972 | 1770.6 | 7.00 | 3.382 | 0.5945 |
| tune | gap_only | 30 | 0.3667 | 540.9 | 2.00 | 2.490 | 0.4017 |
| tune | guarded | 30 | 0.4028 | 1135.1 | 4.97 | 3.130 | 0.5101 |
| validation | fixed_1 | 20 | 0.4250 | 332.3 | 1.00 | 3.217 | 0.2625 |
| validation | fixed_3 | 20 | 0.4125 | 802.6 | 3.00 | 4.181 | 0.3542 |
| validation | fixed_5 | 20 | 0.4375 | 1370.3 | 5.00 | 4.180 | 0.3833 |
| validation | fixed_10 | 20 | 0.4875 | 2516.6 | 10.00 | 4.705 | 0.4183 |
| validation | fixed_20 | 20 | 0.3625 | 4832.8 | 20.00 | 5.611 | 0.4725 |
| validation | adaptive | 20 | 0.3583 | 1795.7 | 6.65 | 4.884 | 0.4083 |
| validation | gap_only | 20 | 0.4208 | 541.3 | 1.65 | 3.283 | 0.2625 |
| validation | guarded | 20 | 0.4833 | 1312.0 | 4.90 | 3.500 | 0.3017 |
| dev | fixed_5 | 50 | 0.3633 | 1156.4 | 5.00 | 4.374 | 0.4066 |
| dev | fixed_10 | 50 | 0.3650 | 2454.1 | 10.00 | 4.363 | 0.4501 |
| dev | adaptive | 50 | 0.3250 | 1798.1 | 7.30 | 4.420 | 0.4334 |
| dev | gap_only | 50 | 0.3417 | 558.1 | 2.30 | 2.933 | 0.3112 |
| dev | guarded | 50 | 0.3467 | 1518.2 | 6.04 | 3.611 | 0.3794 |

## Interpretation limits

The 50 practice queries exclude the original five pilot queries and are split before evaluation into 30 tuning and 20 validation queries. The separately sampled 50 dev queries represent 50/125 coverage; they are not the full split. The guard threshold and best fixed count use only tuning judgments; validation and dev do not change configuration.

The adaptive baseline is a bounded top-100 adaptation of the published algorithm with B=5, not reproduction of its original benchmarks. Bare gap selection uses B=0; the guard accepts this bare cutoff only for an unclipped sufficiently prominent gap, otherwise falling back to the tuning-selected fixed count. Reporting the bare rule separately distinguishes removing the buffer from adding the safeguard. This is a simple tested modification, not claimed as novel. Selection uses only retrieval scores, never qrels or judged answers at inference.

Each distinct prompt has one generation and one JSON-mode judgment. Identical prompts selected by different policies share the same cached answer and judgment; the corresponding generation latency is one shared observation, not an independent rerun. Local retrieval uses ten warm timing repeats; selection uses 100. Downstream time excludes ASR, embedding, loading and judging; it combines these warm local measurements with the single observed API response time.

Paired bootstrap intervals resample queries; they do not quantify generation or judge uncertainty. The same model generates and judges, which can bias scores. Failure cases and per-query evidence coverage must accompany averages. Missing/invalid API judgments are never scored as zero. A .02 tuning tolerance is an operational screening criterion, not proof of statistical non-inferiority.

Tokens are measured resource proxies, not billed monetary cost or server GPU/energy measurements. Findings may be negative. Evidence coverage from public qrels is not the same as answer sufficiency.
