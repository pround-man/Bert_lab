"""Versioned, non-secret dataset configuration and label contract."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

LABEL_NAMES = (
    "OPEN_SPORT", "OPEN_SETTINGS", "OPEN_HEART_RATE", "OPEN_SLEEP",
    "OPEN_ACTIVITY", "OPEN_WORKOUT_HISTORY", "OPEN_WEATHER", "OPEN_COMPASS",
    "OPEN_NAVIGATION", "OPEN_TIMER", "OPEN_STOPWATCH", "OPEN_ALARM",
    "OPEN_MUSIC", "OPEN_BLUETOOTH", "CLOSE_BLUETOOTH", "OPEN_FLASHLIGHT",
    "CLOSE_FLASHLIGHT", "ENABLE_DO_NOT_DISTURB", "DISABLE_DO_NOT_DISTURB",
    "ENABLE_BATTERY_SAVER", "DISABLE_BATTERY_SAVER", "ENABLE_ALWAYS_ON_DISPLAY",
    "DISABLE_ALWAYS_ON_DISPLAY", "LOCK_SCREEN", "FIND_PHONE",
)
LOCALES = ("en", "zh-Hans", "th")
ZERO_WEIGHTS = {
    "chitchat": 50, "negation": 10, "withdrawal": 10, "quoted": 8,
    "past": 6, "howto": 6, "unclear": 5, "unsupported_parameter": 5,
}


@dataclass(frozen=True)
class Settings:
    root: Path
    output_dir: Path = Path("data/generated")
    model_dir: Path = Path("models/xlm-roberta-base")
    version: str = "0.1"
    seed: int = 42
    locales: tuple[str, ...] = LOCALES
    split_sizes: dict[str, int] = field(
        default_factory=lambda: {"train": 3000, "validation": 500, "test": 1000}
    )
    pilot_per_locale: int = 100
    batch_size: int = 10
    max_tokens: int = 512
    semantic_retries: int = 2
    near_duplicate_threshold: float = 0.90
    zero_scenario_weights: dict[str, int] = field(default_factory=lambda: dict(ZERO_WEIGHTS))

    def fingerprint_payload(self) -> dict:
        """Portable behavior contract; secrets and machine paths never enter it."""
        return {
            "version": self.version, "seed": self.seed, "locales": list(self.locales),
            "split_sizes": dict(self.split_sizes), "pilot_per_locale": self.pilot_per_locale,
            "batch_size": self.batch_size, "max_tokens": self.max_tokens,
            "semantic_retries": self.semantic_retries,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "zero_scenario_weights": dict(self.zero_scenario_weights),
            "ratios": {"single": 60, "multi": 20, "zero": 20},
        }


def _keys(value: dict, allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {where} configuration keys: {sorted(unknown)}")


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def load_settings(root: Path, config_path: Path | None = None) -> Settings:
    root = root.resolve()
    path = config_path or root / "configs/dataset.toml"
    if not path.is_absolute():
        path = root / path
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    _keys(raw, {
        "version", "seed", "locales", "pilot_per_locale", "batch_size", "max_tokens",
        "semantic_retries", "near_duplicate_threshold", "paths", "split_sizes",
        "zero_scenario_weights",
    }, "top-level")
    paths = raw.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("paths must be a table")
    _keys(paths, {"output_dir", "model_dir"}, "paths")
    version = raw.get("version", "0.1")
    if version != "0.1":
        raise ValueError("Only dataset/label version 0.1 is supported")
    locales_raw = raw.get("locales", list(LOCALES))
    if not isinstance(locales_raw, list) or tuple(locales_raw) != LOCALES:
        raise ValueError("locales must be exactly ['en', 'zh-Hans', 'th'] in that order")
    sizes = raw.get("split_sizes", {"train": 3000, "validation": 500, "test": 1000})
    if not isinstance(sizes, dict) or set(sizes) != {"train", "validation", "test"}:
        raise ValueError("split_sizes must contain train, validation, and test")
    for split, size in sizes.items():
        _positive_int(size, f"split_sizes.{split}")
        if size % 50:
            raise ValueError(f"split_sizes.{split} must be a multiple of 50")
    pilot = _positive_int(raw.get("pilot_per_locale", 100), "pilot_per_locale")
    if pilot % 10 or pilot > sizes["train"]:
        raise ValueError("pilot_per_locale must be a multiple of 10 and no larger than train")
    if pilot * 6 // 10 < 25:
        raise ValueError("pilot_per_locale must cover all 25 single labels")
    batch = _positive_int(raw.get("batch_size", 10), "batch_size")
    max_tokens = _positive_int(raw.get("max_tokens", 512), "max_tokens")
    if max_tokens > 512:
        raise ValueError("max_tokens cannot exceed the agreed 512-token limit")
    retries = _positive_int(raw.get("semantic_retries", 2), "semantic_retries", allow_zero=True)
    if retries > 2:
        raise ValueError("semantic_retries cannot exceed two replacement attempts")
    seed = raw.get("seed", 42)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    threshold = raw.get("near_duplicate_threshold", 0.90)
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 < threshold <= 1):
        raise ValueError("near_duplicate_threshold must be finite and in (0, 1]")
    weights = raw.get("zero_scenario_weights", dict(ZERO_WEIGHTS))
    if not isinstance(weights, dict) or set(weights) != set(ZERO_WEIGHTS):
        raise ValueError(f"zero_scenario_weights must contain {sorted(ZERO_WEIGHTS)}")
    for scenario, weight in weights.items():
        _positive_int(weight, f"zero_scenario_weights.{scenario}", allow_zero=True)
    if sum(weights.values()) != 100:
        raise ValueError("zero_scenario_weights must sum to 100")

    def resolve_path(key: str, fallback: str) -> Path:
        value = paths.get(key, fallback)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a nonempty path string")
        parsed = Path(value)
        return (root / parsed).resolve() if not parsed.is_absolute() else parsed.resolve()

    return Settings(
        root=root, output_dir=resolve_path("output_dir", "data/generated"),
        model_dir=resolve_path("model_dir", "models/xlm-roberta-base"), version=version,
        seed=seed, locales=LOCALES, split_sizes=dict(sizes), pilot_per_locale=pilot,
        batch_size=batch, max_tokens=max_tokens, semantic_retries=retries,
        near_duplicate_threshold=float(threshold), zero_scenario_weights=dict(weights),
    )


def load_catalog(root: Path) -> dict:
    with (root / "configs/labels.v0.1.json").open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict) or catalog.get("version") != "0.1":
        raise ValueError("Label catalog must have version 0.1")
    rules = catalog.get("rules")
    if not isinstance(rules, list) or not rules or not all(isinstance(x, str) and x for x in rules):
        raise ValueError("Catalog rules must be nonempty strings")
    labels = catalog.get("labels")
    if not isinstance(labels, list) or len(labels) != 25:
        raise ValueError("Catalog must contain exactly 25 labels")
    for index, (entry, name) in enumerate(zip(labels, LABEL_NAMES, strict=True)):
        if (not isinstance(entry, dict) or type(entry.get("id")) is not int
                or entry["id"] != index or entry.get("name") != name):
            raise ValueError(f"Label {index} must be {name}; fixed order cannot change")
        if not isinstance(entry.get("description"), str) or not entry["description"].strip():
            raise ValueError(f"Missing description for {name}")
        for field_name in ("include", "exclude"):
            items = entry.get(field_name)
            if not isinstance(items, list) or not items or not all(isinstance(x, str) and x for x in items):
                raise ValueError(f"{name}.{field_name} must contain nonempty strings")
        examples = entry.get("examples")
        if not isinstance(examples, dict) or set(examples) != set(LOCALES):
            raise ValueError(f"{name} must have examples in all three locales")
        for locale, pair in examples.items():
            if (not isinstance(pair, dict) or set(pair) != {"positive", "negative"}
                    or not all(isinstance(x, str) and x.strip() for x in pair.values())):
                raise ValueError(f"Invalid {name} example pair for {locale}")
    return catalog
