from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _precision_recall_f1(target: np.ndarray, prediction: np.ndarray, labels: list[str]) -> tuple[float, float, float]:
    values = []
    for label in range(len(labels)):
        tp = ((target == label) & (prediction == label)).sum()
        fp = ((target != label) & (prediction == label)).sum()
        fn = ((target == label) & (prediction != label)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((float(precision), float(recall), float(f1), int((target == label).sum())))
    macro = np.asarray(values, dtype=float).mean(axis=0)
    total = max(1, len(target))
    weighted = np.average(np.asarray([row[2] for row in values]), weights=np.asarray([row[3] for row in values]))
    return float(macro[0]), float(macro[1]), float(weighted if len(labels) == 3 else macro[2])


def evaluate_classification(records: Sequence[dict], labels: Sequence[str]) -> dict:
    labels = list(labels)
    index = {name: position for position, name in enumerate(labels)}
    target, prediction = [], []
    for record in records:
        label = record.get("label_name", record.get("label"))
        predicted = record.get("prediction", record.get("predicted_label"))
        if isinstance(label, int):
            label = labels[label]
        if isinstance(predicted, int):
            predicted = labels[predicted]
        if label in index and predicted in index:
            target.append(index[label])
            prediction.append(index[predicted])
    if not target:
        return {"count": 0, "accuracy": None, "precision": None, "recall": None, "f1": None, "macro_f1": None, "weighted_f1": None}
    target_array, prediction_array = np.asarray(target), np.asarray(prediction)
    precision, recall, weighted_or_macro = _precision_recall_f1(target_array, prediction_array, labels)
    per_class_f1 = []
    for label in range(len(labels)):
        tp = ((target_array == label) & (prediction_array == label)).sum()
        fp = ((target_array != label) & (prediction_array == label)).sum()
        fn = ((target_array == label) & (prediction_array != label)).sum()
        p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        per_class_f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"count": len(target), "accuracy": float((target_array == prediction_array).mean()), "precision": precision, "recall": recall, "f1": float(np.mean(per_class_f1)), "macro_f1": float(np.mean(per_class_f1)), "weighted_f1": float(weighted_or_macro)}
