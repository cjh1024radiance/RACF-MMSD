from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np


def normalize_distribution(values: dict[str, float], labels: Sequence[str]) -> np.ndarray:
    """Validate and normalize a named categorical probability distribution."""
    if set(values) != set(labels):
        raise ValueError(f"Probability labels must be exactly {list(labels)}; received {sorted(values)}")
    vector = np.asarray([float(values[label]) for label in labels], dtype=float)
    total = float(vector.sum())
    if not np.isfinite(vector).all() or (vector < 0).any() or total <= 0:
        raise ValueError(f"Invalid probability distribution: {values}")
    return vector / total


def js_distance(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = (first + second) / 2.0

    def kl_divergence(distribution: np.ndarray) -> float:
        nonzero = distribution > 0
        return float(np.sum(distribution[nonzero] * np.log2(distribution[nonzero] / midpoint[nonzero])))

    return float(np.sqrt((kl_divergence(first) + kl_divergence(second)) / 2.0))


def l1_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first - second).sum())


def compute_effects(original: np.ndarray, counterfactuals: Sequence[np.ndarray]) -> list[dict[str, float]]:
    return [{"ce_js": js_distance(original, distribution), "ce_l1": l1_distance(original, distribution)} for distribution in counterfactuals]


def _validate_effects(effects: Sequence[float]) -> np.ndarray:
    values = np.asarray(effects, dtype=float)
    if values.size != 3:
        raise ValueError(f"CEC requires exactly three counterfactual effects; received {values.size}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"CEC effects must be finite and non-negative: {effects}")
    return values


def cec_cv_consistency(effects: Sequence[float], epsilon: float = 1e-8) -> dict:
    values = _validate_effects(effects)
    mean = float(values.mean())
    standard_deviation = float(values.std())
    coefficient_of_variation = standard_deviation / (mean + epsilon)
    return {
        "cec_definition": "cv_consistency",
        "cec_value": max(0.0, min(1.0, 1.0 - coefficient_of_variation)),
        "cec_raw_statistics": {"mean_ce": mean, "std_ce": standard_deviation, "cv": coefficient_of_variation, "epsilon": epsilon},
    }


def cec_std_based_consistency(effects: Sequence[float]) -> dict:
    values = _validate_effects(effects)
    standard_deviation = float(values.std())
    return {
        "cec_definition": "std_based_consistency",
        "cec_value": 1.0 / (1.0 + standard_deviation),
        "cec_raw_statistics": {"mean_ce": float(values.mean()), "std_ce": standard_deviation},
    }


def cec_pairwise_similarity(effects: Sequence[float]) -> dict:
    values = _validate_effects(effects)
    differences = [abs(float(values[first]) - float(values[second])) for first in range(3) for second in range(first + 1, 3)]
    mean_pairwise_difference = float(np.mean(differences))
    return {
        "cec_definition": "pairwise_similarity",
        "cec_value": 1.0 / (1.0 + mean_pairwise_difference),
        "cec_raw_statistics": {"mean_ce": float(values.mean()), "std_ce": float(values.std()), "mean_pairwise_difference": mean_pairwise_difference},
    }


CEC_METRICS: dict[str, Callable[..., dict]] = {
    "cv_consistency": cec_cv_consistency,
    "std_based_consistency": cec_std_based_consistency,
    "pairwise_similarity": cec_pairwise_similarity,
    "coefficient_of_variation": cec_cv_consistency,
}


def compute_cec(effects: Sequence[float], metric_name: str, epsilon: float = 1e-8) -> dict:
    try:
        metric = CEC_METRICS[metric_name]
    except KeyError as error:
        raise ValueError(f"Unknown CEC metric: {metric_name}. Available: {sorted(CEC_METRICS)}") from error
    return metric(effects, epsilon=epsilon) if metric_name in {"cv_consistency", "coefficient_of_variation"} else metric(effects)


def _normalized(scores: Sequence[float]) -> list[float]:
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Weights must be finite and non-negative: {scores}")
    total = float(values.sum())
    return (values / total).tolist() if total > 0 else np.full(values.size, 1.0 / values.size).tolist()


def counterfactual_weights(effects: Sequence[float], sample_id: str, seed: int, epsilon: float = 1e-8) -> dict[str, list[float]]:
    values = _validate_effects(effects)
    mean = float(values.mean())
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return {
        "uniform": np.full(3, 1.0 / 3.0).tolist(),
        "random": _normalized(generator.random(3)),
        "magnitude": _normalized(values),
        "reliability": _normalized(1.0 / (np.abs(values - mean) + epsilon)),
    }


def aggregate_counterfactual_distributions(distributions: Sequence[np.ndarray], weights: dict[str, Sequence[float]]) -> dict[str, list[float]]:
    stacked = np.asarray(distributions, dtype=float)
    if stacked.shape != (3, 3):
        raise ValueError(f"Expected three 3-class counterfactual distributions; received shape {stacked.shape}")
    return {method: (np.asarray(method_weights, dtype=float) @ stacked).tolist() for method, method_weights in weights.items()}
