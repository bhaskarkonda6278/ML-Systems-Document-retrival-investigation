"""Label-free selection policies. Qrels and judge scores never enter these functions."""
from __future__ import annotations

import math


def adaptive_k(scores, *, buffer=5, min_k=1, max_k=20, search_fraction=0.9):
    """Bounded course adaptation of Taguchi et al. (EMNLP 2025), section 3.3.

    Search adjacent gaps entirely within the top floor(0.9*M) candidates,
    take the earliest largest gap, add B=5 and clamp to [min_k,max_k].
    The paper's benchmark setting is NOT reproduced: this study uses M=100
    candidates from the course corpus and explicitly limits output to 20.
    """
    values = [float(x) for x in scores]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("finite nonempty sorted scores required")
    if any(a < b for a, b in zip(values, values[1:])):
        raise ValueError("scores must be sorted descending")
    if not (0 < search_fraction <= 1 and 1 <= min_k <= max_k and buffer >= 0):
        raise ValueError("invalid policy bounds")
    limit = min(len(values), max(2, math.floor(len(values) * search_fraction)))
    gaps = [values[i] - values[i + 1] for i in range(limit - 1)]
    raw_k = max(range(len(gaps)), key=gaps.__getitem__) + 1 if gaps else 1
    selected = min(len(values), max_k, max(min_k, raw_k + buffer))
    head = values[:min(max_k, len(values))]
    span = head[0] - head[-1]
    # A descriptive score-curve feature, NOT a probability of evidence sufficiency.
    prominence = (gaps[raw_k - 1] / span) if gaps and span > 0 else 0.0
    return {"k": selected, "raw_k": raw_k, "gap": gaps[raw_k-1] if gaps else 0.0,
            "prominence": prominence, "search_candidates": limit,
            "clipped": raw_k + buffer > min(max_k, len(values))}


def guarded_k(scores, *, threshold, fixed_k, **kwargs):
    """Return the adaptive choice only for an unclipped, sufficiently prominent gap.

    Otherwise use a fixed count selected on practice-tuning data. This simple
    safeguard is an evaluated modification, not claimed as a novel algorithm.
    """
    result = adaptive_k(scores, **kwargs)
    accept = not result["clipped"] and result["prominence"] >= threshold
    return {**result, "k": result["k"] if accept else min(fixed_k, len(scores)),
            "fallback": not accept}
