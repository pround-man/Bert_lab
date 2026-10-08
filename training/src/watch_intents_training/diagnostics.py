"""Explicit preflight, train-subset overfit, and measured hardware checks."""

import gc
import itertools
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import FrozenData, encode_texts, ensure_disjoint
from .engine import autocast, environment, infer, make_loader, optimize_group, runtime, seed_all
from .io import write_json
from .metrics import compute_metrics, predict_labels
from .model import IntentModel, load_bundle, save_bundle


def check(config):
    data = FrozenData(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
    rows_by_split, counts, lengths = [], {}, {}
    for split in ("train", "validation", "test"):
        rows, _ = data.read_split(split)
        rows_by_split.append(rows)
        counts[split] = len(rows)
        for row in rows:
            encoded = encode_texts(tokenizer, [row["text"]], config.max_length)
            lengths.setdefault(row["locale"], []).append(len(encoded[0]["input_ids"]))
    ensure_disjoint(*rows_by_split)
    device, precision = runtime(config)
    result = {"ok": True, "dataset": data.identity, "counts": counts, "labels": len(data.labels),
              "device": device, "precision": precision, "environment": environment(),
              "lengths": {locale: {"max": max(values), "p50": float(np.percentile(values, 50)),
                                    "p95": float(np.percentile(values, 95)),
                                    "p99": float(np.percentile(values, 99))}
                          for locale, values in lengths.items()}}
    write_json(config.output_dir / "preflight.json", result)
    return result


def smoke(config, *, forward_only=False):
    data = FrozenData(config)
    rows, y = data.read_split("train")
    device, precision = runtime(config)
    seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
    # Keep local real-weight inference within a few GB of RAM.
    c = replace(config, eval_batch=2, micro_batch=2)
    loader = make_loader(rows[:2], y[:2], tokenizer, c)
    model = IntentModel.from_base(config.model_dir, len(data.labels)).to(device)
    probs = infer(model, loader, device, precision)
    if probs.shape != (2, 25) or not np.isfinite(probs).all():
        raise ValueError("Invalid real-model forward output")
    if not forward_only:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate)
        scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
        before = model.classifier.weight.detach().cpu().clone()
        optimize_group(model, list(loader), optimizer, scaler, device, precision, c.max_grad_norm)
        if torch.equal(before, model.classifier.weight.detach().cpu()):
            raise ValueError("Backward/update did not change the classifier")
        expected = infer(model, loader, device, "fp32")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="smoke-", dir=config.output_dir) as temp:
            bundle = Path(temp) / "model"
            save_bundle(bundle, model, tokenizer, {"labels": data.labels,
                        "thresholds": [.5] * 25, "max_length": config.max_length})
            del optimizer, model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            restored, _, _ = load_bundle(bundle, device)
            actual = infer(restored, loader, device, "fp32")
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
                raise ValueError("Saved/reloaded model predictions differ")
    result = {"ok": True, "forward_only": forward_only, "shape": list(probs.shape),
              "device": device, "precision": precision, "environment": environment()}
    write_json(config.output_dir / ("forward-smoke.json" if forward_only else "training-smoke.json"), result)
    return result


def overfit(config, *, max_steps=200, count=96):
    if max_steps < 1 or count < 1:
        raise ValueError("Overfit steps and sample count must be positive")
    data = FrozenData(config)
    rows, y = data.read_split("train")
    buckets = [[i for i, r in enumerate(rows) if r["locale"] == loc and r["kind"] == kind]
               for loc in ("en", "zh-Hans", "th") for kind in ("zero", "single", "multi")]
    indices = [i for group in itertools.zip_longest(*buckets) for i in group if i is not None][:count]
    selected, targets = [rows[i] for i in indices], y[indices]
    device, precision = runtime(config)
    seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
    model = IntentModel.from_base(config.model_dir, 25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    loader = make_loader(selected, targets, tokenizer, config, train=True)
    batches = itertools.cycle(loader)
    evaluation = make_loader(selected, targets, tokenizer, config)
    score, step = 0.0, 0
    for step in range(1, max_steps + 1):
        model.train()
        optimize_group(model, [next(batches) for _ in range(config.gradient_accumulation)],
                       optimizer, scaler, device, precision, config.max_grad_norm)
        if step % 10 == 0 or step == max_steps:
            probs = infer(model, evaluation, device, precision)
            score = compute_metrics(targets, predict_labels(probs, .5), data.labels)["exact_set_accuracy"]["value"]
            print(f"overfit step={step} exact={score:.4f}", flush=True)
            if score >= .98:
                break
    result = {"ok": score >= .98, "exact_set_accuracy": score, "steps": step,
              "samples": len(selected), "sample_ids": [r["sample_id"] for r in selected],
              "scope": "train_subset_diagnostic_not_a_candidate", "learning_rate": 1e-4}
    write_json(config.output_dir / "overfit.json", result)
    return result


def benchmark(config, *, steps=100, warmup=20, bundle=None):
    if steps < 1 or warmup < 0:
        raise ValueError("Invalid benchmark iteration count")
    data = FrozenData(config)
    rows, y = data.read_split("validation" if bundle else "train")
    device, precision = runtime(config)
    seed_all(config.seed)
    if bundle:
        model, tokenizer, meta = load_bundle(bundle, device)
        if meta["max_length"] != config.max_length:
            raise ValueError("Benchmark length differs from bundle")
        # End-to-end local call: tokenization + transfer + forward + labels + serialization.
        import json
        def iteration(i):
            text = rows[i % len(rows)]["text"]
            encoded = encode_texts(tokenizer, [text], config.max_length)
            inputs = tokenizer.pad(encoded, return_tensors="pt")
            with torch.inference_mode(), autocast(device, precision):
                p = torch.sigmoid(model(**{k: v.to(device) for k, v in inputs.items()}).float())
            pred = predict_labels(p.cpu().numpy(), meta["thresholds"])[0]
            json.dumps([n for n, on in zip(meta["labels"], pred, strict=True) if on])
            return 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
        model = IntentModel.from_base(config.model_dir, 25).to(device).train()
        batches = itertools.cycle(make_loader(rows, y, tokenizer, config, train=True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
        def iteration(i):
            group = [next(batches) for _ in range(config.gradient_accumulation)]
            optimize_group(model, group, optimizer, scaler, device, precision, config.max_grad_norm)
            return sum(len(target) for _, target in group)
    for i in range(warmup):
        iteration(i)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations, samples = [], 0
    for i in range(steps):
        start = time.perf_counter()
        samples += iteration(i + warmup)
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    result = {"mode": "inference_batch1_local_call" if bundle else "training_optimizer_steps",
              "steps": steps, "warmup": warmup, "samples": samples,
              "samples_per_second": samples / sum(durations),
              "p50_ms": float(np.percentile(durations, 50) * 1000),
              "p95_ms": float(np.percentile(durations, 95) * 1000),
              "p99_ms": float(np.percentile(durations, 99) * 1000),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "device": device, "precision": precision, "config": config.payload(),
              "environment": environment(), "production_service_latency": False}
    write_json(config.output_dir / ("benchmark-inference.json" if bundle else "benchmark-training.json"), result)
    return result
