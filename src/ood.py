from __future__ import annotations

from typing import Any

import numpy as np


def normalized_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy normalized to [0, 1].

    A value near 0 means the classifier is highly confident.
    A value near 1 means the probability distribution is nearly uniform.
    """
    probs = np.asarray(probabilities, dtype=float)

    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D array")

    if len(probs) < 2:
        raise ValueError("at least two class probabilities are required")

    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")

    if np.any(probs < 0):
        raise ValueError("probabilities cannot be negative")

    total = probs.sum()

    if total <= 0:
        raise ValueError("probabilities must have positive sum")

    # Be tolerant of tiny floating-point deviations from sum=1.
    probs = probs / total

    entropy = -np.sum(
        probs * np.log(np.clip(probs, 1e-12, 1.0))
    )

    max_entropy = np.log(len(probs))

    return float(entropy / max_entropy)


def ood_decision(
    probabilities: np.ndarray,
    top_similarity: float,
    entropy_threshold: float,
    similarity_threshold: float,
) -> dict[str, Any]:
    """
    Decide whether a ticket should be treated as out-of-domain.

    The ticket is considered OOD when either:
      1. normalized entropy is above the configured threshold, or
      2. similarity to the nearest known ticket is below the threshold.
    """
    entropy = normalized_entropy(probabilities)

    low_confidence = entropy > entropy_threshold
    low_similarity = top_similarity < similarity_threshold

    reasons: list[str] = []

    if low_confidence:
        reasons.append("high prediction entropy")

    if low_similarity:
        reasons.append("low similarity to known tickets")

    return {
        "ood": bool(low_confidence or low_similarity),
        "entropy": float(entropy),
        "top_similarity": float(top_similarity),
        "ood_reasons": reasons,
    }