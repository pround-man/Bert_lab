import json

import pytest

from watch_intents.models import DatasetError
from watch_intents.storage import Store, atomic_json, digest


def test_resume_preserves_completed_samples_and_rejects_different_config(tmp_path):
    with Store(tmp_path, {"version": "one"}, resume=True) as store:
        store.accept({"sample_id": "s1", "text": "accepted", "labels": ["A"]})
        store.reject("s2", ["ambiguous"], {"locale": "th"})
    with Store(tmp_path, {"version": "one"}, resume=True) as store:
        assert store.samples()[0]["text"] == "accepted"
        assert store.rejection_count("s2") == 1
    with pytest.raises(DatasetError, match="fingerprint"):
        Store(tmp_path, {"version": "two"}, resume=True)
    with pytest.raises(DatasetError, match="resume"):
        Store(tmp_path, {"version": "one"}, resume=False)


def test_rejection_records_metadata_only_and_calls_mark_uncertain_usage(tmp_path):
    with Store(tmp_path, {"version": "one"}, resume=True) as store:
        store.reject("s1", ["label_mismatch"], {"locale": "en", "kind": "single"})
        call_id = store.begin_call("generator", ["s1"])
        assert store.calls()[0]["status"] == "started"
        store.finish_call(call_id, {"total_tokens": 123}, 1)
        assert store.calls()[0]["usage"] == {"total_tokens": 123}
        assert store.events()[0]["reason_codes"] == ["label_mismatch"]
        assert store.samples() == []


def test_atomic_json_is_valid_utf8_and_digest_is_order_independent(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"text": "打开蓝牙"})
    assert json.loads(path.read_text("utf-8"))["text"] == "打开蓝牙"
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
