"""Single-GPU/CPU training with exact, complete-epoch continuation."""

import contextlib
import importlib.metadata
import math
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import FrozenData, encode_texts, ensure_disjoint
from .io import digest, directory_hashes, file_hash, write_json
from .metrics import compute_metrics, predict_labels
from .model import IntentModel, save_bundle


def run_path(config, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must be a simple name, not a path")
    path = config.output_dir / run_id
    if path.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    return path


def runtime(config):
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    precision = config.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU verification uses fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 not supported on this GPU")
    return device, precision


def autocast(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        inputs, targets = zip(*rows, strict=True)
        return self.tokenizer.pad(list(inputs), padding=True, return_tensors="pt"), torch.stack(targets)


def make_loader(rows, targets, tokenizer, config, *, train=False, epoch=0):
    encoded = encode_texts(tokenizer, [r["text"] for r in rows], config.max_length)
    dataset = list(zip(encoded, torch.as_tensor(targets, dtype=torch.float32), strict=True))
    return DataLoader(dataset, batch_size=config.micro_batch if train else config.eval_batch,
                      shuffle=train, generator=torch.Generator().manual_seed(config.seed + epoch),
                      collate_fn=Collator(tokenizer), num_workers=config.num_workers)


def optimize_group(model, batches, optimizer, scaler, device, precision, max_grad_norm):
    """Weight by examples, including the final short micro-batch and accumulation group."""
    optimizer.zero_grad(set_to_none=True)
    total = sum(len(y) for _, y in batches)
    accumulated = 0.0
    for inputs, targets in batches:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)
        with autocast(device, precision):
            logits = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        weighted = loss * (len(targets) / total)
        if scaler is not None:
            scaler.scale(weighted).backward()
        else:
            weighted.backward()
        accumulated += loss.detach().item() * len(targets)
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                   error_if_nonfinite=scaler is None)
    stepped = True
    if scaler is not None:
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= previous_scale
    else:
        optimizer.step()
    return accumulated, stepped


def infer(model, loader, device, precision):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for inputs, _ in loader:
            with autocast(device, precision):
                logits = model(**{k: v.to(device) for k, v in inputs.items()})
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probabilities)


def environment():
    return {"versions": {name: importlib.metadata.version(name)
                          for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors")},
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def provenance(config, data, device, precision):
    code_dir = Path(__file__).parent
    code_hashes = {p.name: file_hash(p) for p in sorted(code_dir.glob("*.py"))}
    lock = code_dir.parents[1] / "uv.lock"
    return {"config": config.training_contract(), "dataset": data.identity,
            "base_assets": directory_hashes(config.model_dir), "training_code": code_hashes,
            "dependency_lock": file_hash(lock) if lock.exists() else None,
            "environment": environment(), "device": device, "precision": precision}


def _save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, run_id, *, resume=False, stop_after_epoch=None):
    directory = run_path(config, run_id)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(config.output_dir / f".{run_id}.lock"), timeout=0):
        return _train(config, directory, resume, stop_after_epoch)


def _train(config, directory, resume, stop_after_epoch):
    device, precision = runtime(config)
    data = FrozenData(config)
    rows, y = data.read_split("train")
    val_rows, val_y = data.read_split("validation")
    ensure_disjoint(rows, val_rows)
    identity = provenance(config, data, device, precision)
    contract = digest(identity)
    if directory.exists() and not resume:
        raise ValueError("Run already exists; use --resume or a new run ID")
    if resume and not (directory / "latest.pt").exists():
        raise ValueError("No complete-epoch checkpoint to resume")
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
    val_loader = make_loader(val_rows, val_y, tokenizer, config)
    model = IntentModel.from_base(config.model_dir, len(data.labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(math.ceil(len(rows) / config.micro_batch) / config.gradient_accumulation)
    total_steps = config.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * config.warmup_ratio),
                                               total_steps)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    start, best, stale, history, best_dir = 0, -1.0, 0, [], None
    if resume:
        state = torch.load(directory / "latest.pt", map_location=device, weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Resume contract differs: data/model/config/software/device changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler:
            scaler.load_state_dict(state["scaler"])
        start, best, stale, history, best_dir = (state[k] for k in
                                               ("epoch", "best", "stale", "history", "best_dir"))
        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        del state
    result = {"run_id": directory.name, "identity": identity, "contract": contract,
              "status": "running", "history": history, "best_dir": best_dir,
              "completed_epochs": start}
    write_json(directory / "run.json", result)
    try:
        for epoch in range(start, config.epochs):
            if stale >= config.patience:
                break
            loader = make_loader(rows, y, tokenizer, config, train=True, epoch=epoch)
            model.train()
            start_time, loss_sum = time.perf_counter(), 0.0
            group = []
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for index, batch in enumerate(loader):
                group.append(batch)
                if len(group) == config.gradient_accumulation or index == len(loader) - 1:
                    group_loss, stepped = optimize_group(model, group, optimizer, scaler, device,
                                                         precision, config.max_grad_norm)
                    loss_sum += group_loss
                    if stepped:
                        scheduler.step()
                    group = []
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            probs = infer(model, val_loader, device, precision)
            metrics = compute_metrics(val_y, predict_labels(probs, .5), data.labels)
            score = metrics["exact_set_accuracy"]["value"]
            history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(rows),
                            "validation_exact_set": score, "validation_micro_f1": metrics["micro_f1"]})
            if score > best:
                best, stale = score, 0
                best_dir = f"best-epoch-{epoch + 1:03d}"
                target = directory / best_dir
                # An interrupted save before the epoch checkpoint can leave an orphan bundle.
                if target.exists():
                    shutil.rmtree(target)
                save_bundle(target, model, tokenizer,
                            {"labels": data.labels, "label_version": data.catalog["version"],
                             "max_length": config.max_length, "thresholds": [.5] * len(data.labels),
                             "identity": identity, "epoch": epoch + 1, "calibrated": False})
            else:
                stale += 1
            numpy_rng = np.random.get_state()
            _save_checkpoint(directory / "latest.pt", {
                "contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch + 1, "best": best, "stale": stale, "history": history, "best_dir": best_dir,
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            })
            result.update(history=history, best_dir=best_dir, completed_epochs=epoch + 1)
            write_json(directory / "run.json", result)
            write_json(directory / f"performance-epoch-{epoch + 1:03d}.json", {
                "train_seconds": elapsed, "train_samples_per_second": len(rows) / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None})
            for old in directory.glob("best-epoch-*"):
                if old.is_dir() and old.name != best_dir:
                    shutil.rmtree(old)
            print(f"epoch={epoch + 1} loss={loss_sum / len(rows):.6f} validation_exact={score:.4f}", flush=True)
            if stop_after_epoch and epoch + 1 >= stop_after_epoch:
                result["status"] = "interrupted"
                write_json(directory / "run.json", result)
                return result
        result["status"] = "complete"
        write_json(directory / "run.json", result)
        return result
    except BaseException:
        result["status"] = "interrupted"
        write_json(directory / "run.json", result)
        raise
