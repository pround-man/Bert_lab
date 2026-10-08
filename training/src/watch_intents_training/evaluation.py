"""Validation-only calibration and explicitly frozen test evaluation."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .data import FrozenData, ensure_disjoint
from .engine import infer, make_loader, run_path, runtime
from .io import digest, directory_hashes, read_json, verify_files, write_json
from .metrics import calibrate, predict_labels, stratified_metrics
from .model import load_bundle


def bundle_manifest(path):
    hashes = read_json(path / "checksums.json")
    verify_files(path, hashes)
    return digest(hashes)


def _prediction_inputs(config, run_id, split, *, calibrated=True):
    data = FrozenData(config)
    directory = run_path(config, run_id)
    run = read_json(directory / "run.json")
    if run["status"] != "complete":
        raise ValueError("Training must be complete before calibration/evaluation")
    path = directory / ("model" if calibrated else run["best_dir"])
    device, precision = runtime(config)
    model, tokenizer, meta = load_bundle(path, device)
    if meta["identity"]["dataset"] != data.identity or meta["labels"] != data.labels:
        raise ValueError("Model and evaluation dataset/labels differ")
    if meta["max_length"] != config.max_length:
        raise ValueError("Evaluation length differs from model contract")
    rows, y = data.read_split(split)
    train_rows, _ = data.read_split("train")
    if split == "test":
        val_rows, _ = data.read_split("validation")
        ensure_disjoint(train_rows, val_rows, rows)
    else:
        ensure_disjoint(train_rows, rows)
    probs = infer(model, make_loader(rows, y, tokenizer, config), device, precision)
    return data, rows, y, probs, meta, path


def calibrate_run(config, run_id):
    directory = run_path(config, run_id)
    if (directory / "calibration.json").exists():
        saved = read_json(directory / "calibration.json")
        if saved["model_digest"] != bundle_manifest(directory / "model"):
            raise ValueError("Calibrated model changed")
        if saved["dataset"] != FrozenData(config).identity:
            raise ValueError("Calibration dataset changed")
        return saved
    data, rows, y, probs, meta, best = _prediction_inputs(config, run_id, "validation", calibrated=False)
    selected = calibrate(y, probs, data.labels)
    target = directory / "model"
    # A freeze file is produced only after calibration.json; this is an incomplete copy.
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(best, target)
    meta.update(thresholds=[selected["threshold"]] * len(data.labels), calibrated=True)
    write_json(target / "bundle.json", meta)
    hashes = directory_hashes(target)
    hashes.pop("checksums.json", None)
    write_json(target / "checksums.json", hashes)
    selected.update(dataset=data.identity, model_digest=bundle_manifest(target))
    with (directory / "threshold-curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected["curve"][0]))
        writer.writeheader()
        writer.writerows(selected["curve"])
    metrics = stratified_metrics(rows, y, predict_labels(probs, selected["threshold"]), data.labels)
    write_report(directory, "validation", rows, y, probs, meta, metrics)
    # Completion marker must be last: interrupted report generation is safe to rerun.
    write_json(directory / "calibration.json", selected)
    return selected


def freeze_runs(config, run_ids):
    """Persist the exact set of candidates before any sealed-test predictions are generated."""
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("Freeze requires distinct run IDs")
    data = FrozenData(config)
    candidates = {}
    for run_id in run_ids:
        path = run_path(config, run_id)
        run = read_json(path / "run.json")
        calibration = read_json(path / "calibration.json")
        model_digest = bundle_manifest(path / "model")
        if (run["status"] != "complete" or calibration["dataset"] != data.identity
                or calibration["model_digest"] != model_digest):
            raise ValueError("Candidate is not fully calibrated against this data")
        candidates[run_id] = {"model_digest": model_digest,
                              "threshold": calibration["threshold"],
                              "validation_metrics": calibration["metrics"]}
    content = {"dataset": data.identity, "candidates": candidates,
               "policy": "No model/threshold selection using test results"}
    path = config.output_dir / f"freeze-{digest(content)[:16]}.json"
    write_json(path, content)
    return path


def evaluate_run(config, run_id, split, *, freeze=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if split == "test":
        if freeze is None:
            raise ValueError("Test evaluation requires --freeze from the freeze command")
        frozen = read_json(freeze)
        expected = frozen["candidates"].get(run_id)
        if (expected is None or frozen["dataset"] != FrozenData(config).identity
                or expected["model_digest"] != bundle_manifest(run_path(config, run_id) / "model")):
            raise ValueError("Candidate or dataset differs from freeze")
    data, rows, y, probs, meta, _ = _prediction_inputs(config, run_id, split)
    if not meta.get("calibrated"):
        raise ValueError("Model has not been calibrated")
    metrics = stratified_metrics(rows, y, predict_labels(probs, meta["thresholds"]), data.labels)
    report = write_report(run_path(config, run_id), split, rows, y, probs, meta, metrics)
    if freeze is not None:
        report["freeze"] = read_json(freeze)
        write_json(run_path(config, run_id) / f"{split}-metrics.json", report)
    return report


def write_report(directory, split, rows, y, probabilities, metadata, metrics):
    directory = Path(directory)
    labels = metadata["labels"]
    predictions = predict_labels(probabilities, metadata["thresholds"])
    report = {"split": split, "metrics": metrics, "model": metadata,
              "scope": "synthetic_only_not_production_acceptance",
              "uncertainty": "Wilson intervals assume independent samples; template dependence may invalidate this assumption.",
              "macro_policy": "Mean of defined label F1 values; undefined labels and count are reported."}
    write_json(directory / f"{split}-metrics.json", report)
    with (directory / f"{split}-predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, truth, pred, probs in zip(rows, y, predictions, probabilities, strict=True):
            value = {"sample_id": row["sample_id"], "text": row["text"], "locale": row["locale"],
                     "kind": row["kind"], "scenario": row["scenario"], "truth": row["labels"],
                     "prediction": [label for label, on in zip(labels, pred, strict=True) if on],
                     "probabilities": probs.tolist(), "exact": bool(np.array_equal(truth, pred))}
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    with (directory / f"{split}-per-label.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["locale", *metrics["overall"]["per_label"][0]])
        writer.writeheader()
        for locale, m in {"overall": metrics["overall"], **metrics["locale"]}.items():
            writer.writerows({"locale": locale, **row} for row in m["per_label"])
    lines = [f"# {split} 指标报告", "", "评估范围：合成文本；不代表生产验收或线上覆盖率。",
             "比例指标含 Wilson 95% 区间（见 JSON）；模板相关性可能使独立样本假设失效。",
             "", "|语言|样本数|集合完全正确率|Micro F1|Macro F1|简单链路错误率|简单链路覆盖率|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "不可评估" if value is None else f"{value:.4f}"
    for locale, m in {"overall": metrics["overall"], **metrics["locale"]}.items():
        values = [m["exact_set_accuracy"]["value"], m["micro_f1"], m["macro_f1"],
                  m["simple_error_rate"]["value"], m["simple_coverage"]["value"]]
        lines.append(f"|{locale}|{m['n']}|" + "|".join(fmt(v) for v in values) + "|")
    lines += ["", (f"语言等权集合正确率：{fmt(metrics['language_equal_mean_exact'])}；"
              f"最差语言：{metrics['worst_language_exact']['locale']}。"),
              "", "完整路由指标、0/1/≥2 混淆矩阵、场景及语言×场景切片见同名 JSON；",
              "逐标签支持数见 CSV；全部预测和错误案例见 predictions.jsonl。",
              "测试误例仅用于后续诊断；据此调优后不能继续把同一测试集称为独立盲测。", ""]
    (directory / f"{split}-report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
