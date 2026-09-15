from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def response_vectors(original: Sequence[float], counterfactuals: Sequence[Sequence[float]]) -> np.ndarray:
    base = np.asarray(original, dtype=float)
    values = np.asarray(counterfactuals, dtype=float)
    if values.ndim != 2 or values.shape[1] != base.size or not np.isfinite(values).all() or not np.isfinite(base).all():
        raise ValueError("Original and counterfactual distributions must be finite and shape-compatible.")
    return values - base


def consensus(vectors: Sequence[Sequence[float]], method: str = "median") -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("At least one response vector is required.")
    if method == "median":
        return np.median(values, axis=0)
    if method == "mean":
        return values.mean(axis=0)
    if method == "geometric_median":
        result = np.median(values, axis=0)
        for _ in range(32):
            distances = np.linalg.norm(values - result, axis=1)
            weights = 1.0 / np.maximum(distances, 1e-8)
            result = (weights[:, None] * values).sum(axis=0) / weights.sum()
        return result
    raise ValueError(f"Unknown consensus method: {method}")


def consistency(vectors: Sequence[Sequence[float]], method: str = "geometric_median", distance: str = "l2", tau: float = 1.0) -> dict:
    values = np.asarray(vectors, dtype=float)
    centre = consensus(values, method)
    deltas = values - centre
    if distance == "l1":
        deviations = np.abs(deltas).sum(axis=1)
    elif distance == "l2":
        deviations = np.linalg.norm(deltas, axis=1)
    elif distance == "cosine":
        denominator = np.linalg.norm(values, axis=1) * max(np.linalg.norm(centre), 1e-8)
        deviations = 1.0 - np.divide(values @ centre, denominator, out=np.zeros(values.shape[0]), where=denominator > 0)
    else:
        raise ValueError(f"Unknown response distance: {distance}")
    reliability = np.exp(-deviations / max(float(tau), 1e-8))
    return {"consensus_method": method, "distance": distance, "tau": float(tau), "consensus": centre.tolist(), "deviation": deviations.tolist(), "reliability": reliability.tolist()}
