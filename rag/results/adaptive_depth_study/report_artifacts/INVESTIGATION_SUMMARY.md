# Five-step retrieval investigation: completion record

## Work completed

1. Implemented and unit-tested the published largest-gap + five-document buffer selection mechanism, explicitly adapted to a top-100 candidate pool and output range 1..20. The bare largest-gap rule is reported separately.
2. Selected 50 new practice queries excluding the original five; preregistered a 30/20 tuning-validation partition and a separate 50-of-125 dev sample.
3. Compared fixed k values and adaptive policies using shared cached transcripts, vectors, generator prompt/settings and judge rubric/JSON mode.
4. Saved per-query quality, evidence coverage, relevant ranks, token usage, stage timings and judge explanations for failure analysis.
5. Tuned a simple gap-prominence fallback only on 30 tuning queries, froze it before validation/dev analysis, then measured it on held-out subsets.

Frozen best fixed count: **k=10**. Guard threshold: **0.3**. The .02 tuning score tolerance is an operational choice, not a proof of non-inferiority.

## Held-out dev comparison

| Policy | n | Mean score | Input tokens | Mean k | Downstream s | Evidence recall |
|---|---:|---:|---:|---:|---:|---:|
| Fixed k=5 | 50 | 0.3633 | 1156.4 | 5.00 | 4.374 | 0.4066 |
| Fixed k=10 | 50 | 0.3650 | 2454.1 | 10.00 | 4.363 | 0.4501 |
| Adaptive-k (buffer 5) | 50 | 0.3250 | 1798.1 | 7.30 | 4.420 | 0.4334 |
| Largest gap (no buffer) | 50 | 0.3417 | 558.1 | 2.30 | 2.933 | 0.3112 |
| Gap + fixed fallback | 50 | 0.3467 | 1518.2 | 6.04 | 3.611 | 0.3794 |

## Paired quality differences versus tuning-selected fixed k=10

- Fixed k=5: mean difference -0.0017; paired query-bootstrap 95% interval [-0.0600, +0.0583]; input-token change -52.9%; observed downstream-time change +0.2%.
- Fixed k=10: mean difference +0.0000; paired query-bootstrap 95% interval [+0.0000, +0.0000]; input-token change +0.0%; observed downstream-time change +0.0%.
- Adaptive-k (buffer 5): mean difference -0.0400; paired query-bootstrap 95% interval [-0.1267, +0.0400]; input-token change -26.7%; observed downstream-time change +1.3%.
- Largest gap (no buffer): mean difference -0.0233; paired query-bootstrap 95% interval [-0.1183, +0.0650]; input-token change -77.3%; observed downstream-time change -32.8%.
- Gap + fixed fallback: mean difference -0.0183; paired query-bootstrap 95% interval [-0.0883, +0.0434]; input-token change -38.1%; observed downstream-time change -17.2%.

## Interpretation

These differences are conditional on one generated answer and one judgment per distinct prompt. Query bootstrap intervals do not measure judge or generation reliability. Hosted API variability, output length and shared prompt observations limit latency claims. Count repeated local retrieval timings separately from single API observations. No ASR speed-up, combined-system speed-up, monetary cost or ELM GPU-energy reduction is established by this retrieval-only study.

The buffer-five policy cannot select fewer than six documents. It is a published comparison baseline, not a mechanism capable of reducing prefix-context tokens below fixed k=5. The guarded bare rule can choose fewer documents, but tuning quality constraints can select a constant fixed fallback; report such collapse honestly.

## Largest dev losses for the guarded rule

- q_bba95a0c50bb2b65: guard k=1, score=0.000, recall=0.500; fixed k=10, score=1.000, recall=0.500; gap prominence=0.3558, fallback=False.
- q_0ab841636984af0c: guard k=1, score=0.000, recall=0.000; fixed k=10, score=0.583, recall=0.000; gap prominence=0.3880, fallback=False.
- q_3df64fbcb882bbee: guard k=1, score=0.000, recall=0.000; fixed k=10, score=0.583, recall=0.667; gap prominence=0.3233, fallback=False.
- q_236bb02e0972054e: guard k=1, score=0.417, recall=0.500; fixed k=10, score=1.000, recall=1.000; gap prominence=0.6294, fallback=False.
- q_929d9f9a0e015d65: guard k=1, score=0.500, recall=0.333; fixed k=10, score=0.583, recall=0.667; gap prominence=0.3795, fallback=False.
