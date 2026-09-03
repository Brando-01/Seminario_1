import numpy as np


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(scores))]) if len(thresholds) else 0.5


def select_threshold_at_fp_budget(y_true: np.ndarray, probabilities: np.ndarray, max_false_positives_per_1000: float) -> float:
    """Maximize validation recall while respecting a false-positive budget."""
    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    limit = max_false_positives_per_1000 / 1000.0
    valid = np.flatnonzero(false_positive_rate <= limit + 1e-12)
    if not len(valid):
        return float("inf")
    best = max(valid, key=lambda index: (true_positive_rate[index], -false_positive_rate[index], thresholds[index]))
    return float(thresholds[best])


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    predictions = (probabilities >= threshold).astype(int)
    return {
        "auc_pr": float(average_precision_score(y_true, probabilities)),
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "false_positives_per_1000": float(((predictions == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1) * 1000),
    }
