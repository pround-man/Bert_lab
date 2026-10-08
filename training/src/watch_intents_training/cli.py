"""All commands are offline; training runs only when explicitly invoked."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def parser():
    result = argparse.ArgumentParser(description="XLM-R multilingual multilabel baseline")
    result.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/baseline.toml")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Validate frozen exports, mappings, lengths and environment")
    smoke = commands.add_parser("smoke", help="Real model load/update/save/reload check")
    smoke.add_argument("--forward-only", action="store_true", help="Low-memory local real-weight check")
    fit = commands.add_parser("overfit", help="Train-only subset diagnostic; not a formal candidate")
    fit.add_argument("--steps", type=int, default=200)
    fit.add_argument("--samples", type=int, default=96)
    bench = commands.add_parser("benchmark", help="Measured hardware performance")
    bench.add_argument("--steps", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--bundle", type=Path, help="Measure inference instead of backward/optimizer")
    train = commands.add_parser("train")
    train.add_argument("--run", required=True)
    train.add_argument("--resume", action="store_true")
    calibration = commands.add_parser("calibrate", help="Validation-only global threshold sweep")
    calibration.add_argument("--run", required=True)
    freeze = commands.add_parser("freeze", help="Freeze calibrated candidates before test access")
    freeze.add_argument("--runs", nargs="+", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--freeze", type=Path)
    experiment = commands.add_parser("experiment", help="Three learning rates + two extra seeds; then test")
    experiment.add_argument("--name", default="baseline")
    experiment.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--text", required=True)
    pack = commands.add_parser("pack", help="Allowlisted source/model/data transfer archive")
    pack.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    pack.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-transfer")
    verify.add_argument("--root", type=Path, required=True)
    return result


def execute(args):
    if args.command in ("pack", "verify-transfer"):
        from .transfer import pack, verify_transfer
        return ({"archive": str(pack(args.root, args.output))} if args.command == "pack"
                else verify_transfer(args.root))
    config = load_config(args.config)
    if args.command == "check":
        from .diagnostics import check
        return check(config)
    if args.command == "smoke":
        from .diagnostics import smoke
        return smoke(config, forward_only=args.forward_only)
    if args.command == "overfit":
        from .diagnostics import overfit
        return overfit(config, max_steps=args.steps, count=args.samples)
    if args.command == "benchmark":
        from .diagnostics import benchmark
        return benchmark(config, steps=args.steps, warmup=args.warmup, bundle=args.bundle)
    if args.command == "train":
        from .engine import train
        result = train(config, args.run, resume=args.resume)
        return {key: result[key] for key in ("run_id", "status", "completed_epochs", "best_dir")}
    if args.command in ("calibrate", "freeze", "evaluate"):
        from .evaluation import calibrate_run, evaluate_run, freeze_runs
        if args.command == "calibrate":
            result = calibrate_run(config, args.run)
            return {"threshold": result["threshold"], "validation": result["metrics"]}
        if args.command == "freeze":
            return {"freeze": str(freeze_runs(config, args.runs))}
        result = evaluate_run(config, args.run, args.split, freeze=args.freeze)
        return {"report": str(config.output_dir / args.run / f"{args.split}-report.md"),
                "exact_set_accuracy": result["metrics"]["overall"]["exact_set_accuracy"]}
    if args.command == "experiment":
        from .workflow import experiment
        return experiment(config, args.name, resume=args.resume)
    if args.command == "predict":
        import torch

        from .data import encode_texts
        from .engine import autocast, runtime
        from .metrics import predict_labels
        from .model import load_bundle
        device, precision = runtime(config)
        model, tokenizer, meta = load_bundle(args.bundle, device)
        encoded = encode_texts(tokenizer, [args.text], meta["max_length"])
        inputs = tokenizer.pad(encoded, return_tensors="pt")
        with torch.inference_mode(), autocast(device, precision):
            logits = model(**{k: v.to(device) for k, v in inputs.items()})
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        selected = predict_labels(probs, meta["thresholds"])[0]
        return [label for label, on in zip(meta["labels"], selected, strict=True) if on]
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        value = execute(args)
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if isinstance(value, dict) and value.get("ok") is False else 0
    except KeyboardInterrupt:
        print("Interrupted; resume from the latest complete epoch with --resume.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
