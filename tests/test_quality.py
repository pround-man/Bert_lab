"""Checks protect final labels, multilingual text, and held-out evaluation integrity."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from watch_intents.models import Candidate, DatasetError, Judgement
from watch_intents.quality import (
    DuplicateIndex,
    TextInspector,
    validate_candidate,
    validate_judgement,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/xlm-roberta-base"
LABELS = ["OPEN_BLUETOOTH", "CLOSE_BLUETOOTH", "OPEN_SETTINGS"]


@pytest.fixture(scope="module")
def inspector():
    return TextInspector(MODEL_DIR)


def candidate(text="Please open Bluetooth.", labels=None, locale="en"):
    return Candidate(
        sample_id="sample-1", text=text,
        labels=["OPEN_BLUETOOTH"] if labels is None else labels, locale=locale,
    )


def slot(locale="en", labels=None):
    return SimpleNamespace(
        slot_id="sample-1", locale=locale,
        labels=("OPEN_BLUETOOTH",) if labels is None else tuple(labels),
    )


def judgement(**overrides):
    fields = {
        "sample_id": "sample-1", "labels": ["OPEN_BLUETOOTH"],
        "language_ok": True, "naturalness_ok": True, "ambiguous": False, "reason_codes": [],
    }
    fields.update(overrides)
    return Judgement(**fields)


@pytest.mark.parametrize("locale,text", [
    ("en", "Please open Bluetooth."),
    ("zh-Hans", "请打开 Bluetooth"),
    ("th", "ช่วยเปิด Bluetooth ให้หน่อย"),
])
def test_real_tokenizer_and_each_language_accept_natural_commands(inspector, locale, text):
    value = candidate(text=text, locale=locale)
    assert validate_candidate(value, slot(locale), LABELS, inspector) == []
    encoded = inspector.tokenizer.encode(text, add_special_tokens=False)
    assert inspector.count(text) == len(encoded.ids) + 2


def test_real_tokenizer_counts_past_512_without_truncation_or_padding(inspector):
    assert inspector.count("") == 2
    long_text = "Turn on Bluetooth please. " * 200
    assert inspector.count(long_text) > 512
    assert "too_many_tokens" in validate_candidate(candidate(long_text), slot(), LABELS, inspector)
    metadata = inspector.metadata()
    assert metadata["weights_size_bytes"] > 1_000_000_000
    assert len(metadata["tokenizer_sha256"]) == len(metadata["model_config_sha256"]) == 64
    assert metadata["framework"] == "pt"
    metadata["max_tokens"] = 1
    assert inspector.metadata()["max_tokens"] == 512


@pytest.mark.parametrize("locale,text", [
    ("en", "เปิด Bluetooth ให้หน่อย"),
    ("en", "请打开 Bluetooth"),
    ("en", "открой Bluetooth"),
    ("zh-Hans", "Open Bluetooth please"),
    ("th", "Bluetooth 123!?"),
    ("th", "123๙๐!"),
])
def test_wrong_script_is_rejected_even_when_latin_product_name_appears(inspector, locale, text):
    assert "language_mismatch" in validate_candidate(
        candidate(text, locale=locale), slot(locale), LABELS, inspector,
    )


def test_final_labels_are_sets_but_duplicates_and_unplanned_labels_fail(inspector):
    value = candidate(labels=["OPEN_SETTINGS", "OPEN_BLUETOOTH"])
    assert validate_candidate(value, slot(labels=reversed(value.labels)), LABELS, inspector) == []
    value = candidate(labels=["OPEN_BLUETOOTH", "OPEN_BLUETOOTH", "INVENTED"])
    errors = validate_candidate(value, slot(), LABELS, inspector)
    assert {"duplicate_labels", "unknown_labels", "planned_label_mismatch"} <= set(errors)


def test_identity_locale_empty_text_and_placeholders_fail(inspector):
    value = candidate(" \n ")
    value.sample_id = "wrong-id"
    value.locale = "th"
    assert {"sample_id_mismatch", "locale_mismatch", "blank_text"} <= set(
        validate_candidate(value, slot(), LABELS, inspector)
    )
    assert "placeholder_text" in validate_candidate(
        candidate("Please {{command}}."), slot(), LABELS, inspector,
    )


def test_zero_labels_and_negation_are_for_independent_judge_not_keywords(inspector):
    value = candidate("Don't turn on Bluetooth.", labels=[])
    assert validate_candidate(value, slot(labels=[]), LABELS, inspector) == []
    assert validate_judgement(judgement(labels=[]), value, LABELS) == []


def test_judge_must_agree_on_all_labels_and_all_quality_dimensions():
    assert validate_judgement(judgement(), candidate(), LABELS) == []
    errors = validate_judgement(judgement(
        sample_id="other", labels=["INVENTED", "INVENTED"], language_ok=False,
        naturalness_ok=False, ambiguous=True, reason_codes=["unsupported_parameter"],
    ), candidate(), LABELS)
    assert set(errors) == {
        "sample_id_mismatch", "unknown_labels", "duplicate_labels", "label_mismatch",
        "language_mismatch", "unnatural", "ambiguous", "unsupported_parameter",
    }


def test_judge_nonempty_reason_codes_reject_even_with_positive_booleans():
    assert validate_judgement(
        judgement(reason_codes=["extra_intent"]), candidate(), LABELS,
    ) == ["extra_intent"]
    unsafe = judgement().model_copy(update={"reason_codes": ["arbitrary secret text"]})
    assert validate_judgement(unsafe, candidate(), LABELS) == ["invalid_reason_code"]


def record(text, labels=None, group_id="g1", split="train", locale="en"):
    return {
        "text": text, "locale": locale, "labels": ["OPEN_BLUETOOTH"] if labels is None else labels,
        "group_id": group_id, "split": split,
    }


def check(index, value):
    return index.check(**value)


def test_exact_nfc_whitespace_comparison_preserves_original_and_flags_conflicts():
    index = DuplicateIndex()
    original = record("Open  Bluetooth near Café\nplease.")
    index.add(original)
    assert original["text"] == "Open  Bluetooth near Café\nplease."
    text = "Open Bluetooth near Cafe\u0301 please."
    assert check(index, record(text, group_id="g2")) == "exact_duplicate"
    assert check(index, record(text, labels=[], group_id="g3")) == "conflicting_labels"


def test_group_leakage_detected_even_for_different_translation_and_label_set():
    index = DuplicateIndex()
    index.add(record("Open Bluetooth please."))
    assert check(index, record(
        "请打开蓝牙", locale="zh-Hans", labels=[], split="test",
    )) == "group_leakage"


def test_near_duplicate_checks_locale_label_set_and_split():
    index = DuplicateIndex()
    index.add(record("Please turn on Bluetooth on my watch."))
    text = "Please turn on Bluetooth on the watch."
    assert check(index, record(text, group_id="g2")) == "near_duplicate"
    assert check(index, record(text, group_id="g2", split="test")) == "cross_split_near_duplicate"
    assert check(index, record(text, group_id="g2", locale="th")) is None
    assert check(index, record(text, group_id="g2", labels=[])) is None


def test_short_commands_and_cross_label_negation_minimal_pairs_are_preserved():
    index = DuplicateIndex()
    index.add(record("Open Bluetooth."))
    assert check(index, record("Open Bluetooth!", group_id="g2")) is None
    index.add(record("Please turn Bluetooth on for me now.", group_id="g3"))
    assert check(index, record(
        "Please don't turn Bluetooth on for me now.", labels=[], group_id="g4",
    )) is None


def test_add_cannot_silently_corrupt_index_with_rejected_record():
    index = DuplicateIndex()
    index.add(record("Open Bluetooth."))
    with pytest.raises(ValueError, match="conflicting_labels"):
        index.add(record("Open Bluetooth.", labels=[]))
    with pytest.raises(ValueError, match="group_leakage"):
        index.add(record("Open settings.", split="test"))


def test_lfs_pointer_is_rejected_without_loading_model(tmp_path):
    shutil.copyfile(MODEL_DIR / "config.json", tmp_path / "config.json")
    (tmp_path / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\nsize 1115567652\n"
    )
    with pytest.raises(DatasetError, match="Local XLM-R"):
        TextInspector(tmp_path)


def test_wrong_model_configuration_is_rejected(tmp_path):
    config = json.loads((MODEL_DIR / "config.json").read_text())
    config["model_type"] = "bert"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(DatasetError, match="Local XLM-R"):
        TextInspector(tmp_path)
