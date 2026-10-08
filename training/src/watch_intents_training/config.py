"""Validated, portable experiment configuration."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    model_dir: Path
    output_dir: Path
    max_length: int = 128
    micro_batch: int = 16
    gradient_accumulation: int = 2
    eval_batch: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 5
    patience: int = 2
    warmup_ratio: float = 0.1
    seed: int = 42
    precision: str = "auto"
    device: str = "auto"
    num_workers: int = 0

    def payload(self):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}

    def training_contract(self):
        return {key: value for key, value in self.payload().items()
                if key not in ("data_dir", "model_dir", "output_dir")}


def load_config(path):
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = raw.keys() - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for name in ("data_dir", "model_dir", "output_dir"):
        if not isinstance(raw.get(name), str) or not raw[name].strip():
            raise ValueError(f"Required path: {name}")
        raw[name] = (path.parent / raw[name]).resolve()
    c = Config(**raw)
    for name in ("max_length", "micro_batch", "gradient_accumulation", "eval_batch",
                 "epochs", "patience", "seed", "num_workers"):
        value = getattr(c, name)
        minimum = 0 if name in ("seed", "num_workers") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}")
    if c.max_length > 512:
        raise ValueError("max_length must be <=512")
    for name in ("learning_rate", "weight_decay", "max_grad_norm", "warmup_ratio"):
        value = getattr(c, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name}")
    if c.learning_rate == 0 or c.max_grad_norm == 0 or c.warmup_ratio >= 1:
        raise ValueError("Invalid optimizer or warmup parameters")
    if c.device not in ("auto", "cpu", "cuda") or c.precision not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError("Unsupported device or precision")
    if c.output_dir == c.data_dir or c.output_dir.is_relative_to(c.data_dir):
        raise ValueError("Outputs must not be inside frozen data")
    if c.output_dir == c.model_dir or c.output_dir.is_relative_to(c.model_dir):
        raise ValueError("Outputs must not be inside base model")
    return c
