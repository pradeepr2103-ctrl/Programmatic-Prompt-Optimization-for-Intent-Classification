"""
Classification Evaluation Metrics Engine.
Zero-dependency, pure-Python implementation of Accuracy, Precision, Recall,
Macro-F1, Weighted-F1, and Confusion Matrix calculation.
"""

from typing import Dict, List, Optional, Tuple, Union


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Computes comprehensive classification metrics.

    Returns:
        Dict containing:
        - accuracy: float
        - macro_precision: float
        - macro_recall: float
        - macro_f1: float
        - weighted_f1: float
        - per_class: Dict[label, {precision, recall, f1, support}]
        - confusion_matrix: Dict or 2D array representation
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")

    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "total_samples": 0,
        }

    # Determine unique labels
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    else:
        labels = list(labels)

    total_samples = len(y_true)
    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_count / total_samples

    # Build confusion counts
    # tp[c], fp[c], fn[c], support[c]
    tp = {c: 0 for c in labels}
    fp = {c: 0 for c in labels}
    fn = {c: 0 for c in labels}
    support = {c: 0 for c in labels}

    # 2D Confusion Matrix: cm[actual_idx][pred_idx]
    label_to_idx = {l: idx for idx, l in enumerate(labels)}
    cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

    for yt, yp in zip(y_true, y_pred):
        if yt in support:
            support[yt] += 1
        yt_idx = label_to_idx.get(yt)
        yp_idx = label_to_idx.get(yp)

        if yt_idx is not None and yp_idx is not None:
            cm[yt_idx][yp_idx] += 1

        if yt == yp:
            if yt in tp:
                tp[yt] += 1
        else:
            if yp in fp:
                fp[yp] += 1
            if yt in fn:
                fn[yt] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for c in labels:
        c_tp = tp[c]
        c_fp = fp[c]
        c_fn = fn[c]
        c_supp = support[c]

        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": c_supp,
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(labels) if labels else 0.0
    macro_recall = sum(recalls) / len(labels) if labels else 0.0
    macro_f1 = sum(f1s) / len(labels) if labels else 0.0

    # Weighted F1
    weighted_f1 = sum(per_class[c]["f1_score"] * per_class[c]["support"] for c in labels) / total_samples

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": cm,
        "total_samples": total_samples,
    }


def render_ascii_confusion_matrix(metrics_dict: dict) -> str:
    """Formats the confusion matrix as a clear text table."""
    labels = metrics_dict["labels"]
    cm = metrics_dict["confusion_matrix"]
    short_labels = [l[:12] for l in labels]
    col_width = 13

    lines = []
    header = "Actual \\ Pred".ljust(15) + "".join(lbl.rjust(col_width) for lbl in short_labels)
    lines.append(header)
    lines.append("-" * len(header))

    for idx, row in enumerate(cm):
        row_str = short_labels[idx].ljust(15) + "".join(str(val).rjust(col_width) for val in row)
        lines.append(row_str)

    return "\n".join(lines)
