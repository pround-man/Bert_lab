"""Set and routing metrics: undefined denominators remain explicit nulls."""

import math

import numpy as np


def rate(numerator, denominator):
    n, d = int(numerator), int(denominator)
    if not d:
        return {"value": None, "numerator": n, "denominator": d, "ci95": None}
    p, z = n / d, 1.959963984540054
    center = (p + z * z / (2 * d)) / (1 + z * z / d)
    radius = z * math.sqrt(p * (1 - p) / d + z * z / (4 * d * d)) / (1 + z * z / d)
    return {"value": p, "numerator": n, "denominator": d,
            "ci95": [max(0.0, center - radius), min(1.0, center + radius)]}


def predict_labels(probabilities, threshold):
    values = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(threshold, dtype=float)
    if (values.ndim != 2 or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
            or thresholds.ndim > 1 or not np.isfinite(thresholds).all()
            or ((thresholds <= 0) | (thresholds >= 1)).any()
            or (thresholds.ndim == 1 and thresholds.shape != (values.shape[1],))):
        raise ValueError("Invalid probabilities or thresholds")
    return (values >= thresholds).astype(np.int64)


def compute_metrics(truth, predictions, labels):
    y, p = np.asarray(truth), np.asarray(predictions)
    if (y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(labels)
            or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all()):
        raise ValueError("Expected equal binary [samples, labels] arrays")
    y, p = y.astype(bool), p.astype(bool)
    exact = (y == p).all(axis=1)
    yc, pc = y.sum(axis=1), p.sum(axis=1)
    simple, zero, multi, eligible = pc == 1, yc == 0, yc >= 2, yc == 1
    tp, fp, fn = (y & p).sum(axis=0), (~y & p).sum(axis=0), (y & ~p).sum(axis=0)
    per_label = []
    for i, label in enumerate(labels):
        a, b, c = int(tp[i]), int(fp[i]), int(fn[i])
        per_label.append({"label": label, "support": a + c, "tp": a, "fp": b, "fn": c,
                          "precision": a / (a + b) if a + b else None,
                          "recall": a / (a + c) if a + c else None,
                          "f1": 2 * a / (2 * a + b + c) if 2 * a + b + c else None})
    denominator = int(2 * tp.sum() + fp.sum() + fn.sum())
    defined = [x["f1"] for x in per_label if x["f1"] is not None]
    confusion = np.zeros((3, 3), dtype=int)
    for a, b in zip(np.minimum(yc, 2), np.minimum(pc, 2), strict=True):
        confusion[a, b] += 1
    return {
        "n": len(y), "exact_set_accuracy": rate(exact.sum(), len(y)),
        "micro_f1": float(2 * tp.sum() / denominator) if denominator else None,
        "macro_f1": float(np.mean(defined)) if defined else None,
        "macro_f1_defined_labels": len(defined), "per_label": per_label,
        "simple_error_rate": rate((simple & ~exact).sum(), simple.sum()),
        "simple_accuracy": rate((simple & exact).sum(), simple.sum()),
        "simple_coverage": rate(simple.sum(), len(y)),
        "correct_simple_coverage": rate((simple & exact).sum(), len(y)),
        "simple_request_recall": rate((eligible & exact).sum(), eligible.sum()),
        "zero_to_simple": rate((zero & simple).sum(), zero.sum()),
        "multi_to_simple": rate((multi & simple).sum(), multi.sum()),
        "cardinality_confusion": confusion.tolist(),
    }


def calibrate(truth, probabilities, labels):
    curve = []
    for hundredths in range(10, 91):
        threshold = hundredths / 100
        m = compute_metrics(truth, predict_labels(probabilities, threshold), labels)
        curve.append({"threshold": threshold, "exact_set_accuracy": m["exact_set_accuracy"]["value"],
                      "micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"],
                      "simple_error_rate": m["simple_error_rate"]["value"],
                      "simple_coverage": m["simple_coverage"]["value"]})
    if not len(truth):
        raise ValueError("Cannot calibrate an empty validation set")
    best = max(curve, key=lambda x: (x["exact_set_accuracy"],
                                    -abs(round(x["threshold"] * 100) - 50), -x["threshold"]))
    return {"threshold": best["threshold"], "selection": "validation_exact_set",
            "metrics": best, "curve": curve}


def stratified_metrics(rows, truth, predictions, labels):
    y, p = np.asarray(truth), np.asarray(predictions)
    result = {"overall": compute_metrics(y, p, labels)}
    for key in ("locale", "kind", "scenario"):
        result[key] = {}
        for value in sorted({row[key] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row[key] == value]
            result[key][value] = compute_metrics(y[indices], p[indices], labels)
    language_scores = {name: m["exact_set_accuracy"]["value"]
                       for name, m in result["locale"].items()}
    result["language_equal_mean_exact"] = float(np.mean(list(language_scores.values())))
    worst = min(language_scores, key=language_scores.get)
    result["worst_language_exact"] = {"locale": worst, "value": language_scores[worst]}
    result["locale_scenario"] = {}
    for locale in result["locale"]:
        result["locale_scenario"][locale] = {}
        for scenario in result["scenario"]:
            indices = [i for i, row in enumerate(rows)
                       if row["locale"] == locale and row["scenario"] == scenario]
            if indices:
                result["locale_scenario"][locale][scenario] = compute_metrics(y[indices], p[indices], labels)
    return result
