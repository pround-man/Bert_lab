"""Local structural checks and conservative text deduplication, without inference."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from rapidfuzz import fuzz, process
from tokenizers import Tokenizer

from .models import Candidate, DatasetError, Judgement, ReasonCode

if TYPE_CHECKING:
    from .planning import Slot


class TextInspector:
    """Inspect the local model and count real XLM-R tokens, including <s> and </s>."""

    def __init__(self, model_dir: Path, max_tokens: int = 512):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("max_tokens must be an integer between 1 and 512")
        self.max_tokens = max_tokens
        model_dir = Path(model_dir)
        try:
            config_bytes = (model_dir / "config.json").read_bytes()
            config = json.loads(config_bytes)
            if (config.get("model_type") != "xlm-roberta"
                    or config.get("bos_token_id") != 0
                    or config.get("pad_token_id") != 1
                    or config.get("eos_token_id") != 2
                    or config.get("max_position_embeddings", 0) < max_tokens + 2):
                raise ValueError("Incompatible XLM-R configuration")

            weights = model_dir / "model.safetensors"
            size = weights.stat().st_size
            with weights.open("rb") as handle:
                header_length = struct.unpack("<Q", handle.read(8))[0]
                if not 2 <= header_length <= min(size - 8, 4 * 1024 * 1024):
                    raise ValueError("Invalid safetensors header; download the actual weights")
                header = json.loads(handle.read(header_length))
            if header.get("__metadata__", {}).get("format") != "pt":
                raise ValueError("Expected PyTorch safetensors weights")
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            if not tensors or "roberta.embeddings.word_embeddings.weight" not in header:
                raise ValueError("Missing XLM-R weights")
            ends = []
            for tensor in tensors:
                offsets = tensor["data_offsets"]
                if (not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(value) is not int for value in offsets)
                        or not 0 <= offsets[0] <= offsets[1]):
                    raise ValueError("Invalid safetensors offsets")
                ends.append(offsets[1])
            if max(ends) + header_length + 8 != size:
                raise ValueError("Incomplete safetensors weights")

            tokenizer_bytes = (model_dir / "tokenizer.json").read_bytes()
            self.tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            for token, token_id in (("<s>", 0), ("<pad>", 1), ("</s>", 2)):
                if self.tokenizer.token_to_id(token) != token_id:
                    raise ValueError("Incompatible XLM-R special tokens")
            empty_ids = self.tokenizer.encode("", add_special_tokens=True).ids
            if empty_ids != [0, 2]:
                raise ValueError("Tokenizer must add the XLM-R boundary tokens")
            if self.tokenizer.get_vocab_size() != config.get("vocab_size"):
                raise ValueError("Tokenizer and model vocabulary sizes do not match")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error) as exc:
            raise DatasetError("Local XLM-R model or tokenizer is missing or incompatible") from exc

        self._metadata = {
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "weights_size_bytes": size,
            "weights_format": "safetensors",
            "framework": "pt",
            "max_tokens": max_tokens,
        }

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)

    def metadata(self) -> dict:
        return dict(self._metadata)


_PLACEHOLDER = re.compile(
    r"\{\{[^{}\r\n]+\}\}|\$\{[^{}\r\n]+\}"
    r"|<(?:text|command|utterance|label|intent|placeholder)>"
    r"|\[(?:text|command|utterance|label|intent|placeholder|文本|指令|标签)\]",
    re.IGNORECASE,
)
_SENTINELS = {"todo", "tbd", "placeholder", "...", "…", "待填写", "占位文本"}


def _script(char: str) -> str:
    name = unicodedata.name(char, "")
    if "LATIN" in name:
        return "Latin"
    if "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name:
        return "Han"
    if "THAI" in name:
        return "Thai"
    return "other"


def _language_plausible(text: str, locale: str) -> bool:
    """A script guard only; fluency and language identity belong to the judge."""
    required = {"en": "Latin", "zh-Hans": "Han", "th": "Thai"}.get(locale)
    scripts = {_script(char) for char in text if char.isalpha()}
    return required is not None and required in scripts and scripts <= {required, "Latin"}


def validate_candidate(
    candidate: Candidate, slot: Slot, label_names: list[str], inspector: TextInspector,
) -> list[str]:
    errors = []
    if candidate.sample_id != slot.slot_id:
        errors.append("sample_id_mismatch")
    if candidate.locale != slot.locale:
        errors.append("locale_mismatch")
    if set(candidate.labels) - set(label_names):
        errors.append("unknown_labels")
    if len(candidate.labels) != len(set(candidate.labels)):
        errors.append("duplicate_labels")
    if set(candidate.labels) != set(slot.labels):
        errors.append("planned_label_mismatch")
    if not candidate.text.strip():
        errors.append("blank_text")
    if (_PLACEHOLDER.search(candidate.text)
            or candidate.text.strip().casefold() in _SENTINELS):
        errors.append("placeholder_text")
    if inspector.count(candidate.text) > inspector.max_tokens:
        errors.append("too_many_tokens")
    if not _language_plausible(candidate.text, slot.locale):
        errors.append("language_mismatch")
    return errors


def validate_judgement(
    judgement: Judgement, candidate: Candidate, label_names: list[str],
) -> list[str]:
    errors = []
    if judgement.sample_id != candidate.sample_id:
        errors.append("sample_id_mismatch")
    if set(judgement.labels) - set(label_names):
        errors.append("unknown_labels")
    if len(judgement.labels) != len(set(judgement.labels)):
        errors.append("duplicate_labels")
    if set(judgement.labels) != set(candidate.labels):
        errors.append("label_mismatch")
    if not judgement.language_ok:
        errors.append("language_mismatch")
    if not judgement.naturalness_ok:
        errors.append("unnatural")
    if judgement.ambiguous:
        errors.append("ambiguous")
    safe_codes = set(get_args(ReasonCode))
    for code in judgement.reason_codes:
        errors.append(code if code in safe_codes else "invalid_reason_code")
    return list(dict.fromkeys(errors))


def _comparison_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


class DuplicateIndex:
    """Compare wording, retaining original text and cross-label negation contrasts.

    Exact matches are global. Fuzzy matches require the same locale and final label
    set and at least 20 normalized characters on both sides. We neither case-fold
    nor remove punctuation/negation. This intentionally favors fewer false positives
    for short commands; it is not a semantic similarity or translation detector.
    """

    def __init__(self, threshold: float = 0.90):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be finite and in (0, 1]")
        self.threshold = threshold
        self._exact: dict[str, tuple[str, ...]] = {}
        self._groups: dict[str, str] = {}
        self._near: dict[tuple[str, tuple[str, ...]], dict[str, list[str]]] = {}

    def check(
        self, text: str, locale: str, labels: list[str], group_id: str, split: str,
    ) -> str | None:
        normalized = _comparison_text(text)
        label_key = tuple(sorted(set(labels)))
        prior_labels = self._exact.get(normalized)
        if prior_labels is not None and prior_labels != label_key:
            return "conflicting_labels"
        if group_id in self._groups and self._groups[group_id] != split:
            return "group_leakage"
        if prior_labels is not None:
            return "exact_duplicate"
        if len(normalized) < 20:
            return None
        buckets = self._near.get((locale, label_key), {})
        # Give an inter-split match priority even if an intra-split match is closer.
        for prior_split in sorted(buckets, key=lambda value: value == split):
            match = process.extractOne(
                normalized, buckets[prior_split], scorer=fuzz.ratio,
                score_cutoff=self.threshold * 100,
            )
            if match is not None:
                return "near_duplicate" if prior_split == split else "cross_split_near_duplicate"
        return None

    def add(self, record: dict) -> None:
        reason = self.check(
            record["text"], record["locale"], record["labels"], record["group_id"], record["split"],
        )
        if reason is not None:
            raise ValueError(f"Cannot index rejected record: {reason}")
        normalized = _comparison_text(record["text"])
        label_key = tuple(sorted(set(record["labels"])))
        self._exact[normalized] = label_key
        self._groups[record["group_id"]] = record["split"]
        if len(normalized) >= 20:
            bucket = self._near.setdefault((record["locale"], label_key), {})
            bucket.setdefault(record["split"], []).append(normalized)
