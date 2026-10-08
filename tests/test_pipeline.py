import json
import sqlite3
from dataclasses import replace

import pytest

from watch_intents import pipeline
from watch_intents.config import Settings, load_catalog
from watch_intents.models import (
    BatchResult,
    Candidate,
    DatasetError,
    Judgement,
    TransientProviderError,
)
from watch_intents.planning import Slot
from watch_intents.storage import Store

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT)
TEXTS = {"a": "Enable Bluetooth.", "b": "Open settings and compass.", "c": "I like this park.",
         "d": "Show weather.", "e": "My shoes fit."}
LABELS = {"a": ("OPEN_BLUETOOTH",), "b": ("OPEN_SETTINGS", "OPEN_COMPASS"), "c": (),
          "d": ("OPEN_WEATHER",), "e": ()}


class Inspector:
    max_tokens = 512

    def count(self, text):
        return len(text.split()) + 2

    def metadata(self):
        return {"tokenizer_sha256": "test-fixture"}


class Provider:
    def __init__(self, bad_ids=(), fail_after=None):
        self.bad_ids = set(bad_ids)
        self.fail_after = fail_after
        self.generated_ids = []
        self.judged_ids = []

    def metadata(self):
        return {"generator_model": "glm-5.2", "judge_model": "qwen3.8-max", "prompt_version": "test"}

    def generate(self, slots):
        if self.fail_after is not None and len(self.generated_ids) >= self.fail_after:
            raise TransientProviderError("mock outage")
        self.generated_ids.extend(s.slot_id for s in slots)
        return BatchResult([Candidate(sample_id=s.slot_id, locale=s.locale,
                                      text=("PRIVATE_REJECTED_SENTINEL do this" if s.slot_id in self.bad_ids else TEXTS[s.slot_id]),
                                      labels=list(s.labels)) for s in slots], usage={"total_tokens": 20})

    def judge(self, candidates):
        self.judged_ids.extend(c.sample_id for c in candidates)
        return BatchResult([Judgement(sample_id=c.sample_id, labels=[] if c.sample_id in self.bad_ids else c.labels,
                                      language_ok=True, naturalness_ok=True, ambiguous=False, reason_codes=[])
                            for c in candidates], usage={"total_tokens": 10})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    slots = []
    for sample_id, split in (("a", "train"), ("b", "train"), ("c", "train"), ("d", "validation"), ("e", "test")):
        labels = LABELS[sample_id]
        kind = "single" if len(labels) == 1 else "multi" if labels else "zero"
        slots.append(Slot(sample_id, "en", split, kind, labels, "direct", "group-" + sample_id,
                          split + "-family", "independent seed"))

    def plan(settings, catalog, stage):
        if stage == "pilot":
            return slots[:3]
        if stage == "smoke":
            return [replace(s, split="smoke", family_id="smoke-family") for s in slots[:3]]
        return slots

    monkeypatch.setattr(pipeline, "plan_slots", plan)
    settings = Settings(root=tmp_path, output_dir=tmp_path / "data", locales=("en",), batch_size=1)
    return settings, slots


def build(settings, provider, stage="pilot", resume=True):
    return pipeline.run_build(settings, CATALOG, Inspector(), provider, stage,
                              resume=resume, progress=lambda *a, **kw: None)


def test_resume_reuses_accepted_records_after_outage_and_exports_partial(setup):
    settings, _ = setup
    first = Provider(fail_after=1)
    with pytest.raises(TransientProviderError):
        build(settings, first)
    directory = pipeline.state_directory(settings, "pilot")
    manifest = json.loads((directory / "exports/pilot/manifest.json").read_text())
    assert manifest["accepted"] == 1 and not manifest["complete"]
    second = Provider()
    assert build(settings, second)["complete"]
    assert second.generated_ids == ["b", "c"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]


def test_rejected_text_never_persists_and_replacement_is_bounded(setup):
    settings, _ = setup
    provider = Provider(bad_ids={"a"})
    report = build(settings, provider)
    assert not report["complete"] and report["missing"] == 1
    assert provider.generated_ids.count("a") == 3
    assert report["rejection_reasons"] == {"label_mismatch": 3}
    for file in settings.output_dir.rglob("*"):
        if file.is_file():
            assert b"PRIVATE_REJECTED_SENTINEL" not in file.read_bytes()
    resumed = Provider()
    assert build(settings, resumed)["missing"] == 1
    assert resumed.generated_ids == []  # --resume cannot reset the cost bound.


def test_full_requires_explicit_pilot_verification_then_reuses_pilot(setup):
    settings, _ = setup
    provider = Provider()
    with pytest.raises(DatasetError, match="pilot"):
        build(settings, provider, "full")
    assert provider.generated_ids == []
    build(settings, provider)
    with pytest.raises(DatasetError, match="verify"):
        build(settings, Provider(), "full")
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]
    remaining = Provider()
    assert build(settings, remaining, "full")["complete"]
    assert remaining.generated_ids == ["d", "e"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "full")["ok"]


def test_verify_detects_export_tampering_and_revokes_pilot_gate(setup):
    settings, _ = setup
    build(settings, Provider())
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]
    path = pipeline.state_directory(settings, "pilot") / "exports/pilot/train.jsonl"
    path.write_text("", encoding="utf-8")
    result = pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")
    assert not result["ok"]
    assert any("export_mismatch" in e["reason_codes"] for e in result["errors"])
    with pytest.raises(DatasetError, match="pilot"):
        build(settings, Provider(), "full")


def test_build_refuses_changed_contract_without_model_calls(setup):
    settings, _ = setup
    build(settings, Provider())
    provider = Provider()
    with pytest.raises(DatasetError, match="fingerprint"):
        build(replace(settings, seed=43), provider)
    assert provider.generated_ids == []


def test_smoke_does_not_satisfy_pilot_and_does_not_replace_rejections(setup):
    settings, _ = setup
    provider = Provider(bad_ids={"a"})
    report = build(settings, provider, "smoke", resume=False)
    assert report["missing"] == 1 and provider.generated_ids.count("a") == 1
    assert not (pipeline.state_directory(settings, "pilot") / "state.sqlite3").exists()
    with pytest.raises(DatasetError, match="resume"):
        build(settings, Provider(), "smoke", resume=False)


def test_frozen_dataset_cannot_regenerate_a_deleted_test_row(setup):
    settings, _ = setup
    build(settings, Provider())
    pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")
    build(settings, Provider(), "full")
    pipeline.verify_dataset(settings, CATALOG, Inspector(), "full")
    db = pipeline.state_directory(settings, "full") / "state.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM samples WHERE sample_id='e'")
    provider = Provider()
    with pytest.raises(DatasetError, match="frozen"):
        build(settings, provider, "full")
    assert provider.generated_ids == []


def test_accepted_metadata_and_label_mapping_are_portable(setup):
    settings, _ = setup
    build(settings, Provider())
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        rows = store.samples()
        assert rows[0]["review_status"] == "machine_accepted"
        assert rows[1]["labels"] == ["OPEN_SETTINGS", "OPEN_COMPASS"]
        assert store.calls()[0]["usage"]["total_tokens"] == 20
    out = pipeline.state_directory(settings, "pilot") / "exports/pilot"
    assert json.loads((out / "label2id.json").read_text())["FIND_PHONE"] == 24
    assert len((out / "en/train.jsonl").read_text().splitlines()) == 3


def test_judge_outages_cannot_reset_generation_budget_across_resume(setup):
    settings, _ = setup
    class BrokenJudge(Provider):
        def judge(self, candidates):
            raise TransientProviderError("mock judge outage")
    generated = []
    for _ in range(4):
        provider = BrokenJudge()
        with pytest.raises(TransientProviderError):
            build(settings, provider)
        generated.extend(provider.generated_ids)
    # On the fourth command slot a is exhausted; generation moves to the next slot.
    assert generated.count("a") == 3
    assert generated == ["a", "a", "a", "b"]



def test_continuous_build_retries_beyond_old_limit_and_resume_keeps_counts(setup, monkeypatch):
    settings, _ = setup
    build(settings, Provider(bad_ids={"a"}))
    provider = Provider()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot",
                                resume=True, until_complete=True, progress=lambda *a, **kw: None)
    assert report["complete"]
    assert provider.generated_ids == ["a"]
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        assert store.generation_count("a") == 4
        assert len(store.samples()) == 3


def test_continuous_build_waits_through_generator_and_judge_outages(setup, monkeypatch):
    settings, _ = setup
    sleeps = []
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    class FlakyProvider(Provider):
        generator_failures = 2
        judge_failures = 2
        def generate(self, slots):
            if self.generator_failures:
                self.generator_failures -= 1
                raise TransientProviderError("mock outage")
            return super().generate(slots)
        def judge(self, candidates):
            if self.judge_failures:
                self.judge_failures -= 1
                raise TransientProviderError("mock outage")
            return super().judge(candidates)
    provider = FlakyProvider()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot",
                                resume=True, until_complete=True, progress=lambda *a, **kw: None)
    assert report["complete"]
    assert len(sleeps) == 4
    assert provider.generated_ids == ["a", "b", "c"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]


def test_continuous_does_not_swallow_authentication_or_interrupt(setup, monkeypatch):
    from watch_intents.models import FatalProviderError
    settings, _ = setup
    for error in (FatalProviderError("HTTP 401"), KeyboardInterrupt()):
        class BrokenProvider(Provider):
            def generate(self, slots, error=error):
                raise error
        with pytest.raises(type(error)):
            pipeline.run_build(settings, CATALOG, Inspector(), BrokenProvider(), "pilot",
                               resume=True, until_complete=True, progress=lambda *a, **kw: None)



def legacy_checkpoint(settings):
    from copy import deepcopy

    from watch_intents.migrations import LEGACY_CODE_SHA256
    from watch_intents.reports import export_dataset
    directory = pipeline.state_directory(settings, "pilot")
    with Store(directory, None, resume=True) as store:
        identity = deepcopy(store.get_meta("identity"))
        identity["code_sha256"] = LEGACY_CODE_SHA256
        identity["provider"]["prompt_version"] = "v0.1"
        store.set_meta("identity", identity)
        export_dataset(store, pipeline.plan_slots(settings, CATALOG, "pilot"), "pilot", CATALOG)


def test_upgrade_preserves_samples_counts_audit_and_creates_backup(setup):
    from watch_intents.migrations import upgrade_dataset
    settings, _ = setup
    build(settings, Provider(bad_ids={"a"}))
    legacy_checkpoint(settings)
    directory = pipeline.state_directory(settings, "pilot")
    with Store(directory, None, resume=True) as store:
        original = store.samples(), store.events(), store.calls(), store.generation_count("a")
    metadata = Provider().metadata() | {"prompt_version": "v0.2"}
    result = upgrade_dataset(settings, CATALOG, Inspector(), metadata)
    assert result["accepted"] == 2 and result["api_calls"] == 0
    with sqlite3.connect(result["backup"]) as backup:
        identity = json.loads(backup.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        assert identity["provider"]["prompt_version"] == "v0.1"
    with Store(directory, None, resume=True) as store:
        assert original == (store.samples(), store.events(), store.calls(), store.generation_count("a"))
        assert store.get_meta("identity_history")[0]["accepted"] == 2
    assert upgrade_dataset(settings, CATALOG, Inspector(), metadata)["status"] == "already_current"
    class UpdatedProvider(Provider):
        def metadata(self):
            return metadata
    provider = UpdatedProvider()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot",
                                resume=True, until_complete=True, progress=lambda *a, **kw: None)
    assert report["complete"] and provider.generated_ids == ["a"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]
    build(settings, UpdatedProvider(), "full")
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "full")["ok"]


@pytest.mark.parametrize("change", ["prompt", "seed", "corrupt_export", "unknown_code", "frozen"])
def test_upgrade_rejects_incompatible_or_corrupt_state_without_mutation(setup, change):
    from watch_intents.migrations import upgrade_dataset
    settings, _ = setup
    build(settings, Provider())
    legacy_checkpoint(settings)
    directory = pipeline.state_directory(settings, "pilot")
    metadata = Provider().metadata() | {"prompt_version": "v0.2"}
    if change == "prompt":
        metadata["prompt_version"] = "different"
    if change == "seed":
        settings = replace(settings, seed=99)
    if change == "corrupt_export":
        (directory / "exports/pilot/train.jsonl").write_text("")
    with Store(directory, None, resume=True) as store:
        if change == "unknown_code":
            identity = store.get_meta("identity")
            identity["code_sha256"] = "unrecognized"
            store.set_meta("identity", identity)
        if change == "frozen":
            store.set_meta("full_frozen_signature", "locked")
        identity_before = store.get_meta("identity")
        records_before = store.samples()
    with pytest.raises(DatasetError):
        upgrade_dataset(settings, CATALOG, Inspector(), metadata)
    with Store(directory, None, resume=True) as store:
        assert store.get_meta("identity") == identity_before
        assert store.samples() == records_before
    assert not (directory / "backups").exists()


def test_continuous_stalls_back_off_without_relaxing_judgement(setup, monkeypatch):
    settings, _ = setup
    delays = []
    monkeypatch.setattr(pipeline.time, "sleep", delays.append)
    class EventuallyGood(Provider):
        reviews = 0
        def judge(self, candidates):
            self.reviews += 1
            self.bad_ids = {c.sample_id for c in candidates} if self.reviews <= 12 else set()
            return super().judge(candidates)
    provider = EventuallyGood()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot",
                                resume=True, until_complete=True, progress=lambda *a, **kw: None)
    assert report["complete"]
    assert report["rejection_reasons"]["label_mismatch"] >= 6
    assert delays and all(delay == 60 for delay in delays)



def test_parallel_stages_overlap_and_database_writes_stay_on_coordinator(setup, monkeypatch):
    import threading

    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    owner = threading.get_ident()
    for name in ("begin_call", "finish_call", "reject", "accept"):
        original = getattr(Store, name)
        def checked(self, *args, _original=original, **kwargs):
            assert threading.get_ident() == owner
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(Store, name, checked)
    barrier = threading.Barrier(2)
    judging = threading.Event()
    judge_barrier = threading.Barrier(2)
    next_generation = threading.Event()
    class ConcurrentProvider(Provider):
        def generate(self, slots):
            if slots[0].slot_id in {"a", "b"}:
                barrier.wait(timeout=3)
            else:
                assert judging.wait(3)
                next_generation.set()
            return super().generate(slots)
        def judge(self, candidates):
            judging.set()
            if candidates[0].sample_id in {"a", "b"}:
                judge_barrier.wait(timeout=3)
                assert next_generation.wait(3)
            return super().judge(candidates)
    provider = ConcurrentProvider()
    result = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    assert result["complete"]
    assert sorted(provider.generated_ids) == ["a", "b", "c"]
    assert sorted(provider.judged_ids) == ["a", "b", "c"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]



def test_parallel_final_admission_catches_duplicates_between_reviewed_batches(setup, monkeypatch):
    import threading

    from watch_intents.runtime import RuntimeSettings
    settings, slots = setup
    slots = [slots[0], replace(slots[1], labels=slots[0].labels, kind="single")]
    monkeypatch.setattr(pipeline, "plan_slots", lambda *args: slots)
    gen_barrier, judge_barrier = threading.Barrier(2), threading.Barrier(2)
    shared = "Please enable Bluetooth so the watch can connect to my nearby phone."
    unique = "I need the Bluetooth connection enabled on this watch now, thank you."
    class RacingProvider(Provider):
        def __init__(self):
            super().__init__()
            self.attempts = {}
            self.lock = threading.Lock()
        def generate(self, batch):
            slot = batch[0]
            with self.lock:
                n = self.attempts[slot.slot_id] = self.attempts.get(slot.slot_id, 0) + 1
            if n == 1:
                gen_barrier.wait(timeout=3)
            return BatchResult([Candidate(sample_id=slot.slot_id, locale=slot.locale,
                                          labels=list(slot.labels), text=shared if n == 1 else unique)])
        def judge(self, candidates):
            if candidates[0].text == shared:
                judge_barrier.wait(timeout=3)
            return super().judge(candidates)
    provider = RacingProvider()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    assert report["complete"]
    assert report["rejection_reasons"] == {"exact_duplicate": 1}
    assert sorted(provider.attempts.values()) == [1, 2]
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        assert len({row["text"] for row in store.samples()}) == 2
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]


def test_parallel_transient_judge_retry_keeps_candidates_and_reservations(setup, monkeypatch):
    from watch_intents import concurrent_build
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    monkeypatch.setattr(concurrent_build, "retry_delay", lambda n: 0)
    class FlakyJudge(Provider):
        tries = 0
        def judge(self, candidates):
            if candidates[0].sample_id == "a" and self.tries < 2:
                self.tries += 1
                raise TransientProviderError("private error text")
            return super().judge(candidates)
    provider = FlakyJudge()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    assert report["complete"] and sorted(provider.generated_ids) == ["a", "b", "c"]
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        assert store.generation_count("a") == 1
        assert sum(call["status"] == "failed" for call in store.calls()) == 2
        assert all(call["status"] != "started" for call in store.calls())
    for path in settings.output_dir.rglob("*"):
        if path.is_file():
            assert b"private error text" not in path.read_bytes()


@pytest.mark.parametrize("reason", ["too_long", "ambiguous", "label_mismatch", "unnatural", "wrong_locale"])
def test_parallel_replacements_keep_all_quality_gates(setup, reason):
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    class BadThenGood(Provider):
        attempts = 0
        def generate(self, slots):
            result = super().generate(slots)
            if slots[0].slot_id == "a":
                self.attempts += 1
                if self.attempts == 1:
                    result.items[0].text = "PRIVATE_REJECTED_SENTINEL " + ("word " * 600 if reason == "too_long" else "enable Bluetooth")
                    if reason == "wrong_locale":
                        result.items[0].locale = "th"
            return result
        def judge(self, candidates):
            result = super().judge(candidates)
            if candidates[0].text.startswith("PRIVATE_REJECTED_SENTINEL"):
                verdict = result.items[0]
                if reason == "ambiguous":
                    verdict.ambiguous = True
                elif reason == "label_mismatch":
                    verdict.labels = []
                elif reason == "unnatural":
                    verdict.naturalness_ok = False
            return result
    provider = BadThenGood()
    report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    assert report["complete"] and provider.attempts == 2
    for path in settings.output_dir.rglob("*"):
        if path.is_file():
            assert b"PRIVATE_REJECTED_SENTINEL" not in path.read_bytes()


def test_parallel_interrupt_drains_calls_and_resume_reuses_saved_samples(setup):
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    interrupted = False
    def progress(message, **kwargs):
        nonlocal interrupted
        if "batch accepted=" in message and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_build(settings, CATALOG, Inspector(), Provider(), "pilot", resume=True,
                           until_complete=True, runtime=RuntimeSettings(), progress=progress)
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        saved = {row["sample_id"] for row in store.samples()}
        assert saved
        assert all(call["status"] != "started" for call in store.calls())
    provider = Provider()
    result = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    assert result["complete"]
    assert not set(provider.generated_ids) & saved



@pytest.mark.parametrize("source_code", ["serial", "parallel", "runtime_file"])
def test_serial_v02_partial_full_checkpoint_migrates_without_prompt_or_data_changes(setup, source_code):
    from watch_intents.migrations import (
        PARALLEL_CODE_SHA256,
        RUNTIME_FILE_CODE_SHA256,
        SERIAL_CODE_SHA256,
        upgrade_dataset,
    )
    from watch_intents.reports import export_dataset
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    class VersionedProvider(Provider):
        def metadata(self):
            return super().metadata() | {"prompt_version": "v0.2"}
    build(settings, VersionedProvider())
    pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")
    with pytest.raises(TransientProviderError):
        build(settings, VersionedProvider(fail_after=1), "full")
    directory = pipeline.state_directory(settings, "full")
    with Store(directory, None, resume=True) as store:
        previous = store.get_meta("identity")
        previous["code_sha256"] = {"serial": SERIAL_CODE_SHA256, "parallel": PARALLEL_CODE_SHA256,
                                   "runtime_file": RUNTIME_FILE_CODE_SHA256}[source_code]
        store.set_meta("identity", previous)
        records, events, calls = store.samples(), store.events(), store.calls()
        for stage in ("pilot", "full"):
            export_dataset(store, pipeline.plan_slots(settings, CATALOG, stage), stage, CATALOG)
    metadata = VersionedProvider().metadata() | {"generator_model": "changed-model",
                                               "generator_base_url": "https://new.example/v1"}
    migrated = upgrade_dataset(settings, CATALOG, Inspector(), metadata)
    assert migrated["api_calls"] == 0 and migrated["accepted"] == 4
    with Store(directory, None, resume=True) as store:
        assert store.samples() == records and store.events() == events and store.calls() == calls
        assert store.get_meta("identity")["provider"] == previous["provider"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]
    provider = VersionedProvider()
    assert pipeline.run_build(settings, CATALOG, Inspector(), provider, "full", resume=True,
                              until_complete=True, runtime=RuntimeSettings(generator_workers=3, judge_workers=3),
                              progress=lambda *a, **kw: None)["complete"]
    assert provider.generated_ids == ["e"]
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "full")["ok"]


def test_worker_limit_changes_are_operational_and_preserve_checkpoint(setup):
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    for workers in (1, 3):
        provider = Provider()
        assert pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                  until_complete=True,
                                  runtime=RuntimeSettings(generator_workers=workers, judge_workers=workers),
                                  progress=lambda *a, **kw: None)["complete"]
        if workers == 3:
            assert provider.generated_ids == []
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        assert [entry["runtime"]["generator_workers"] for entry in store.get_meta("execution_runs")] == [1, 3]


def test_parallel_permanent_error_stops_dispatch_and_finalizes_audit(setup):
    from watch_intents.models import FatalProviderError
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    class Unauthorized(Provider):
        def generate(self, slots):
            self.generated_ids.extend(slot.slot_id for slot in slots)
            raise FatalProviderError("HTTP 401")
    provider = Unauthorized()
    with pytest.raises(FatalProviderError, match="401"):
        pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                           until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    assert len(provider.generated_ids) <= 2
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        assert store.samples() == []
        assert all(call["status"] != "started" for call in store.calls())



def test_parallel_real_provider_mock_http_end_to_end(setup):
    import threading
    import time
    from collections import Counter

    import httpx

    from watch_intents.provider import LLMProvider
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    (settings.root / "dashscope.env").write_text(
        "DASHSCOPE_API_KEY=fake-generator-key\nSUUNTO_EVAL_JUDGE_API_KEY=fake-review-key\n"
        "SUUNTO_CHAT_MODEL_CN_NAME=mock-generator\nSUUNTO_EVAL_JUDGE_NAME=mock-judge\n"
    )
    lock = threading.Lock()
    active, peaks = Counter(), Counter()
    def handler(request):
        body = json.loads(request.content)
        role = "generator" if body["model"] == "mock-generator" else "judge"
        payload = json.loads(body["messages"][1]["content"])
        assert body["response_format"] == {"type": "json_object"}
        with lock:
            active[role] += 1
            peaks[role] = max(peaks[role], active[role])
        time.sleep(0.03)
        if role == "generator":
            rows = [{"sample_id": row["sample_id"], "text": TEXTS[row["sample_id"]],
                     "labels": row["target_labels"], "locale": row["locale"]}
                    for row in payload["slots"]]
            result = {"samples": rows}
        else:
            assert all(set(row) == {"sample_id", "text", "locale"} for row in payload["samples"])
            labels_by_text = {TEXTS[key]: value for key, value in LABELS.items()}
            result = {"judgements": [
                {"sample_id": row["sample_id"], "labels": list(labels_by_text[row["text"]]),
                 "language_ok": True, "naturalness_ok": True, "ambiguous": False, "reason_codes": []}
                for row in payload["samples"]]}
        with lock:
            active[role] -= 1
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
                                       "usage": {"total_tokens": 100}})
    provider = LLMProvider(settings.root, CATALOG, transport=httpx.MockTransport(handler))
    try:
        report = pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                                    until_complete=True, runtime=RuntimeSettings(), progress=lambda *a, **kw: None)
    finally:
        provider.close()
    assert report["complete"] and peaks == {"generator": 2, "judge": 2}
    assert report["api_usage"]["reported_tokens"]["total_tokens"] == 600
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]



@pytest.mark.parametrize("parallel", [False, True])
def test_resume_after_model_and_endpoint_change_preserves_old_samples_and_records_sources(setup, parallel):
    from watch_intents.runtime import RuntimeSettings
    settings, _ = setup
    with pytest.raises(TransientProviderError):
        build(settings, Provider(fail_after=1))
    class NewConnection(Provider):
        def metadata(self):
            return super().metadata() | {"generator_model": "new-generator", "judge_model": "new-judge",
                                         "generator_base_url": "https://generator.example/v1",
                                         "judge_base_url": "https://judge.example/v1"}
    provider = NewConnection()
    assert pipeline.run_build(settings, CATALOG, Inspector(), provider, "pilot", resume=True,
                              runtime=RuntimeSettings() if parallel else None,
                              progress=lambda *a, **kw: None)["complete"]
    assert sorted(provider.generated_ids) == ["b", "c"]
    with Store(pipeline.state_directory(settings, "pilot"), None, resume=True) as store:
        rows = {row["sample_id"]: row for row in store.samples()}
        assert rows["a"]["generator_model"] == "glm-5.2"
        assert rows["b"]["generator_model"] == "new-generator"
        profiles = {run["profile_id"]: run["provider"] for run in store.get_meta("provider_runs")}
        assert profiles[rows["b"]["provider_profile_id"]]["judge_base_url"] == "https://judge.example/v1"
    assert pipeline.verify_dataset(settings, CATALOG, Inspector(), "pilot")["ok"]


def test_prompt_change_still_blocks_resume(setup):
    settings, _ = setup
    build(settings, Provider())
    class ChangedPrompt(Provider):
        def metadata(self):
            return super().metadata() | {"prompt_version": "changed"}
    with pytest.raises(DatasetError, match="fingerprint"):
        build(settings, ChangedPrompt())
