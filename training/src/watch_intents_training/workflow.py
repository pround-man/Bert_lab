"""The bounded five-run experiment, with selection finished before test access."""

from dataclasses import replace

import numpy as np

from .engine import run_path, train
from .evaluation import calibrate_run, evaluate_run, freeze_runs
from .io import read_json, write_json

LEARNING_RATES = (2e-5, 1e-5, 3e-5)


def select_learning_rate(scores):
    return max(LEARNING_RATES, key=lambda lr: (
        scores[lr]["exact_set_accuracy"],
        scores[lr]["macro_f1"] if scores[lr]["macro_f1"] is not None else -1,
        -LEARNING_RATES.index(lr)))


def experiment(config, prefix, *, resume=False):
    run_path(config, prefix)  # Validate name before deriving file paths.
    summary_path = config.output_dir / f"{prefix}-summary.json"
    state_path = config.output_dir / f"{prefix}-experiment.json"
    if state_path.exists() and not resume:
        raise ValueError("Experiment already exists; use --resume or a different prefix")
    requested = config.payload()
    if state_path.exists() and read_json(state_path)["config"] != requested:
        raise ValueError("Experiment configuration differs from previous invocation")
    write_json(state_path, {"config": requested, "status": "running"})
    try:
        scores, search_runs = {}, {}
        def run_one(lr, seed):
            c = replace(config, learning_rate=lr, seed=seed)
            name = f"{prefix}-lr{round(lr * 1e6)}-s{seed}"
            path = run_path(c, name)
            if path.exists() and not (path / "latest.pt").exists():
                raise ValueError(f"{name} has no full epoch; inspect it and use a new prefix")
            train(c, name, resume=resume and path.exists())
            calibrated = calibrate_run(c, name)
            return name, calibrated["metrics"]
        for lr in LEARNING_RATES:
            name, score = run_one(lr, 42)
            scores[lr], search_runs[lr] = score, name
        best_lr = select_learning_rate(scores)
        candidates = [search_runs[best_lr]]
        validation_scores = {candidates[0]: scores[best_lr]}
        for seed in (43, 44):
            name, score = run_one(best_lr, seed)
            candidates.append(name)
            validation_scores[name] = score
        selected_run = max(candidates, key=lambda name: (
            validation_scores[name]["exact_set_accuracy"],
            validation_scores[name]["macro_f1"] if validation_scores[name]["macro_f1"] is not None else -1,
            -candidates.index(name)))
        freeze = freeze_runs(config, candidates)
        # Persist selection before evaluating any test examples.
        write_json(state_path, {"config": requested, "status": "frozen", "freeze": str(freeze),
                                "selected_run": selected_run})
        tests = {name: evaluate_run(config, name, "test", freeze=freeze)["metrics"] for name in candidates}
        stability = {}
        for scope in ("overall", "en", "zh-Hans", "th"):
            ms = [tests[n]["overall"] if scope == "overall" else tests[n]["locale"][scope] for n in candidates]
            stability[scope] = {}
            for metric in ("exact_set_accuracy", "micro_f1", "macro_f1", "simple_error_rate", "simple_coverage"):
                values = [m[metric]["value"] if isinstance(m[metric], dict) else m[metric] for m in ms]
                valid = [v for v in values if v is not None]
                stability[scope][metric] = {"values": values, "defined_seeds": len(valid),
                    "mean": float(np.mean(valid)) if valid else None,
                    "sample_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None}
        result = {"status": "complete", "search_runs": {str(k): v for k, v in search_runs.items()},
                  "best_learning_rate": best_lr, "candidates": candidates, "selected_run": selected_run,
                  "selection_split": "validation", "freeze": str(freeze), "stability": stability,
                  "scope": "synthetic_only_not_production_acceptance"}
        write_json(summary_path, result)
        write_json(state_path, {"config": requested, **result})
        lines = ["# XLM-R 三语言首轮实验", "", f"验证集选择的学习率：{best_lr:g}",
                 f"验证集选择的模型包：`{selected_run}/model`", "",
                 "完成三组学习率比较；最佳学习率使用 42/43/44 三个种子。测试集未参与选择。", "",
                 "|范围|Exact-set 均值|样本标准差|", "|---|---:|---:|"]
        for scope, values in stability.items():
            item = values["exact_set_accuracy"]
            lines.append(f"|{scope}|{item['mean']:.4f}|{item['sample_std']:.4f}|")
        lines += ["", "完整各指标稳定性见 summary.json；分语言、逐标签、路由风险、场景误例见每个 run 的报告。",
                  "当前数据为合成文本，结果不代表真实线上覆盖率或生产验收。", ""]
        (config.output_dir / f"{prefix}-report.md").write_text("\n".join(lines), encoding="utf-8")
        return result
    except BaseException as error:
        state = read_json(state_path)
        state.update(status="failed", error_type=type(error).__name__)
        write_json(state_path, state)
        raise
