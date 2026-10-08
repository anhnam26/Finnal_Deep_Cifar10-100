"""Multiclass metrics. Accuracy values in %, other scores in [0,1] unless noted."""

import math

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def wilson_interval(correct, total, z=1.959963984540054):
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Invalid binomial counts")
    p = correct / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return [100 * max(0, center - radius), 100 * min(1, center + radius)]


def classification_metrics(logits, labels, class_names, bins=15):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    classes = len(class_names)
    if (logits.ndim != 2 or logits.shape != (len(labels), classes) or len(labels) == 0
            or classes < 2 or labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() >= classes or bins <= 0 or not np.isfinite(logits).all()):
        raise ValueError("Invalid logits/labels/classes/bin count")
    if len(set(class_names)) != classes:
        raise ValueError("Class names must be unique")
    log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
    probabilities = np.exp(log_prob)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    order = np.argsort(-probabilities, axis=1, kind="stable")
    one_hot = np.eye(classes)[labels]
    matrix = confusion_matrix(labels, predictions, labels=np.arange(classes))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(classes), zero_division=0,
    )
    summary = {
        "samples": len(labels), "classes": classes, "top1_accuracy": float(100 * correct.mean()),
        "error_rate": float(100 * (1 - correct.mean())),
        "top5_accuracy": float(100 * np.any(order[:, :min(5, classes)] == labels[:, None], axis=1).mean()),
        "nll": float(-log_prob[np.arange(len(labels)), labels].mean()),
        "brier_sum_over_classes": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "balanced_accuracy": float(100 * per_recall[support > 0].mean()),
        "accuracy_wilson_95_percent": wilson_interval(int(correct.sum()), len(labels)),
    }
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=np.arange(classes), average=average, zero_division=0,
        )
        summary.update({f"precision_{average}": float(precision), f"recall_{average}": float(recall),
                        f"f1_{average}": float(f1)})
    calibration = []
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean_confidence = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        calibration.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                            "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    summary["ece_15" if bins == 15 else f"ece_{bins}"] = ece
    per_class, curves = [], {}
    grid = np.linspace(0, 1, 201)
    interpolated_roc = []
    aucs, aps, weights = [], [], []
    for index, name in enumerate(class_names):
        truth = one_hot[:, index]
        auc, ap = None, None
        if 0 < support[index] < len(labels):
            auc = float(roc_auc_score(truth, probabilities[:, index]))
            ap = float(average_precision_score(truth, probabilities[:, index]))
            fpr, tpr, _ = roc_curve(truth, probabilities[:, index])
            precision, recall, _ = precision_recall_curve(truth, probabilities[:, index])
            curves[str(index)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                  "precision": precision.tolist(), "recall": recall.tolist()}
            interpolated_roc.append(np.interp(grid, fpr, tpr))
            aucs.append(auc)
            aps.append(ap)
            weights.append(int(support[index]))
        per_class.append({"class_id": index, "class_name": name, "support": int(support[index]),
                          "precision": float(per_precision[index]), "recall": float(per_recall[index]),
                          "f1": float(per_f1[index]), "roc_auc_ovr": auc, "average_precision_ovr": ap})
    summary["auc_defined_classes"] = len(aucs)
    summary["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else None
    summary["roc_auc_ovr_weighted"] = float(np.average(aucs, weights=weights)) if aucs else None
    summary["average_precision_ovr_macro"] = float(np.mean(aps)) if aps else None
    summary["average_precision_ovr_weighted"] = float(np.average(aps, weights=weights)) if aps else None
    flat_truth, flat_probability = one_hot.ravel(), probabilities.ravel()
    summary["roc_auc_ovr_micro"] = float(roc_auc_score(flat_truth, flat_probability))
    summary["average_precision_ovr_micro"] = float(average_precision_score(flat_truth, flat_probability))
    fpr, tpr, _ = roc_curve(flat_truth, flat_probability)
    precision, recall, _ = precision_recall_curve(flat_truth, flat_probability)
    curves["micro"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                       "precision": precision.tolist(), "recall": recall.tolist()}
    if interpolated_roc:
        curves["macro_roc"] = {"fpr": grid.tolist(), "tpr": np.mean(interpolated_roc, axis=0).tolist()}
    return {
        "summary": summary, "per_class": per_class, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(), "calibration": calibration, "curves": curves,
        "classification_report": classification_report(labels, predictions, labels=np.arange(classes),
                                                         target_names=class_names, output_dict=True, zero_division=0),
        "units": "accuracy/error/top5/balanced/Wilson in percent; P/R/F1/AUC/AP/ECE fractions; Brier summed across classes; NLL nats",
        "notes": "Undefined OvR class AUC/AP are null; aggregates include defined classes only. Micro F1 equals top1/100 for single-label data. Wilson is image sampling uncertainty, not training seed variability.",
    }, probabilities


def paired_bootstrap(labels, predictions_a, predictions_b, repeats=2000, seed=2026):
    labels, a, b = map(np.asarray, (labels, predictions_a, predictions_b))
    if labels.ndim != 1 or a.shape != labels.shape or b.shape != labels.shape or not len(labels) or repeats <= 0:
        raise ValueError("Matched predictions and labels required")
    differences = (b == labels).astype(int) - (a == labels).astype(int)
    # Resample the {-1,0,1} paired differences with multinomial counts: exactly
    # equivalent to resampling indexed image pairs, without an N x repeats allocation.
    values, counts = np.unique(differences, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(labels), counts / len(labels), size=repeats)
    estimates = draws @ values / len(labels) * 100
    return {"delta_b_minus_a_percentage_points": float(100 * differences.mean()),
            "paired_bootstrap_95_percent": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "repeats": repeats, "seed": seed, "samples": len(labels),
            "a_correct_b_wrong": int((differences == -1).sum()),
            "a_wrong_b_correct": int((differences == 1).sum()),
            "note": "Image sampling uncertainty only; not training-seed variance or a multiplicity-adjusted significance test."}