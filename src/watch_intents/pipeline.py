"""Generate, blind-review, filter, checkpoint, and verify a fixed quota plan."""

import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from filelock import FileLock, Timeout

from . import __version__
from .concurrent_build import Coordinator
from .models import Candidate, DatasetError, FatalProviderError, Judgement, TransientProviderError
from .planning import plan_slots
from .quality import DuplicateIndex, validate_candidate, validate_judgement
from .reports import export_dataset, sample_signature, selected_samples
from .storage import Store, atomic_json, digest, file_digest, now


def state_directory(settings, stage: str) -> Path:
    # Smoke cannot contaminate or satisfy the pilot/full build gate.
    return settings.output_dir / ("smoke" if stage == "smoke" else "dataset")


def identity_for(settings, catalog: dict, inspector, provider_metadata: dict) -> dict:
    source_dir = Path(__file__).parent
    return {
        "pipeline_version": __version__, "settings": settings.fingerprint_payload(),
        "catalog_sha256": digest(catalog), "assets": inspector.metadata(),
        "provider": provider_metadata,
        "dependency_lock_sha256": file_digest(settings.root / "uv.lock") if (settings.root / "uv.lock").exists() else None,
        "code_sha256": digest({path.name: file_digest(path) for path in sorted(source_dir.glob("*.py"))}),
    }


def _validate_records(records: list[dict], slots: list, catalog: dict, inspector,
                      threshold: float) -> list[dict]:
    expected = {slot.slot_id: slot for slot in slots}
    labels = [label["name"] for label in catalog["labels"]]
    index = DuplicateIndex(threshold)
    errors = []
    seen = set()
    families = {}
    for record in records:
        sample_id = record.get("sample_id")
        slot = expected.get(sample_id)
        reasons = []
        if slot is None or sample_id in seen:
            errors.append({"sample_id": sample_id, "reason_codes": ["unknown_or_duplicate_id"]})
            continue
        seen.add(sample_id)
        try:
            candidate = Candidate.model_validate({key: record[key] for key in ("sample_id", "text", "labels", "locale")})
            reasons.extend(validate_candidate(candidate, slot, labels, inspector))
            for key in ("group_id", "family_id", "split", "kind", "scenario"):
                if record.get(key) != getattr(slot, key):
                    reasons.append("metadata_mismatch")
            if record.get("label_version") != catalog["version"]:
                reasons.append("label_version_mismatch")
            if record.get("source") != "llm_synthetic" or record.get("review_status") != "machine_accepted":
                reasons.append("missing_review")
            if record["labels"] != sorted(record["labels"], key=labels.index):
                reasons.append("label_order")
            if record.get("n_tokens") != inspector.count(record["text"]):
                reasons.append("token_count_mismatch")
            family = record["family_id"]
            if family in families and families[family] != record["split"]:
                reasons.append("family_leakage")
            families[family] = record["split"]
            duplicate = index.check(record["text"], record["locale"], record["labels"],
                                    record["group_id"], record["split"])
            if duplicate:
                reasons.append(duplicate)
            if not reasons:
                index.add(record)
        except (ValueError, TypeError, KeyError):
            # Exceptions can include input text; only persist a bounded code.
            reasons.append("invalid_record")
        if reasons:
            errors.append({"sample_id": sample_id, "reason_codes": sorted(set(reasons))})
    return errors


def _call(store: Store, provider, role: str, items):
    ids = [item.slot_id if role == "generator" else item.sample_id for item in items]
    call_id = store.begin_call(role, ids)
    try:
        result = provider.generate(items) if role == "generator" else provider.judge(items)
    except (FatalProviderError, TransientProviderError) as error:
        store.finish_call(call_id, getattr(error, "usage", {}), getattr(error, "request_count", 1),
                          status="failed", error_type=type(error).__name__)
        raise
    store.finish_call(call_id, result.usage, result.request_count)
    return result


def run_build(settings, catalog: dict, inspector, provider, stage: str, *, resume: bool = False,
              progress=print, until_complete: bool = False, runtime=None) -> dict:
    directory = state_directory(settings, stage)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(directory / ".run.lock"), timeout=0):
            return _build_locked(settings, catalog, inspector, provider, stage, resume, progress,
                                 until_complete and stage != "smoke", runtime if stage != "smoke" else None)
    except Timeout:
        raise DatasetError("Another command is using this dataset; wait for it to finish.") from None


def _build_locked(settings, catalog, inspector, provider, stage, resume, progress, until_complete=False, runtime=None):
    directory = state_directory(settings, stage)
    slots = plan_slots(settings, catalog, stage)
    all_slots = slots if stage == "smoke" else plan_slots(settings, catalog, "full")
    identity = identity_for(settings, catalog, inspector, provider.metadata())
    label_names = [label["name"] for label in catalog["labels"]]
    limit = 1 if stage == "smoke" else settings.semantic_retries + 1
    with Store(directory, identity, resume=resume) as store:
        records = store.samples()
        frozen = store.get_meta("full_frozen_signature")
        if frozen is not None and frozen != sample_signature(records):
            raise DatasetError("The frozen dataset changed; restore its checkpoint or start a new version. "
                               "Frozen test samples cannot be regenerated by --resume.")
        corrupt = _validate_records(records, all_slots, catalog, inspector, settings.near_duplicate_threshold)
        if corrupt:
            raise DatasetError("Existing accepted data failed validation; build stopped without model calls.")
        if stage == "full":
            pilot_slots = plan_slots(settings, catalog, "pilot")
            pilot_rows = selected_samples(store, pilot_slots)
            if (len(pilot_rows) != len(pilot_slots)
                    or store.get_meta("pilot_verified_signature") != sample_signature(pilot_rows)):
                raise DatasetError("Full build requires a complete pilot and successful verify --stage pilot.")
        store.record_provider(provider.metadata(), stage)
        done = {record["sample_id"] for record in records}
        index = DuplicateIndex(settings.near_duplicate_threshold)
        for record in records:
            index.add(record)
        # Save only non-text seed metadata before the first paid call.
        atomic_json(directory / f"{stage}-slots.json", [asdict(slot) for slot in slots])
        lookup = {slot.slot_id: slot for slot in slots}
        generation_counts = {slot.slot_id: store.generation_count(slot.slot_id) for slot in slots}

        if runtime is not None:
            try:
                Coordinator(store, settings, catalog, inspector, provider, slots, records,
                            runtime, until_complete, progress, stage).run()
            finally:
                report = export_dataset(store, slots, stage, catalog)
            return report

        def reject(slot, reasons):
            store.reject(slot.slot_id, sorted(set(reasons)),
                         {key: getattr(slot, key) for key in ("locale", "split", "kind", "scenario")})

        def call_with_recovery(role, items):
            failures = 0
            while True:
                try:
                    result = _call(store, provider, role, items)
                    if result.invalid_ids:
                        counts = Counter(code for codes in result.invalid_reasons.values() for code in codes)
                        progress(f"{stage}: {role} invalid={len(result.invalid_ids)}; "
                                 f"reasons={dict(counts)}", flush=True)
                    return result
                except TransientProviderError:
                    if not until_complete:
                        raise
                    failures += 1
                    seconds = min(15 * 2 ** min(failures - 1, 2), 60)
                    progress(f"{stage}: {role} temporarily unavailable; retry in {seconds}s "
                             f"(outage round {failures})", flush=True)
                    time.sleep(seconds)

        progress(f"{stage}: mode={'until_complete' if until_complete else 'bounded'}; "
                 "accepted checkpoints are reused; quality checks unchanged", flush=True)
        stagnant_batches = 0
        previous_done = len(done)
        try:
            while True:
                if len(done) == previous_done:
                    stagnant_batches += 1
                else:
                    stagnant_batches = 0
                previous_done = len(done)
                pending = [slot for slot in slots if slot.slot_id not in done
                           and (until_complete or generation_counts[slot.slot_id] < limit)]
                if not pending:
                    break
                # Give all missing slots a turn before retrying a difficult slot again.
                if until_complete:
                    pending.sort(key=lambda slot: generation_counts[slot.slot_id])
                if until_complete and stagnant_batches > 5:
                    progress(f"{stage}: no new accepted samples in recent batches; "
                             "wait 60s, then continue with unchanged quality checks", flush=True)
                    time.sleep(60)
                    stagnant_batches = 0
                batch = pending[:settings.batch_size]
                progress(f"{stage}: accepted {sum(s.slot_id in done for s in slots)}/{len(slots)}; "
                         f"generate {len(batch)} candidates", flush=True)
                generation_batch = [replace(slot, seed_brief=slot.seed_brief + (
                    f" Fresh candidate attempt {generation_counts[slot.slot_id] + 1}; "
                    "use a new natural phrasing within this family and scenario, without changing the target set."
                )) for slot in batch]
                generated = call_with_recovery("generator", generation_batch)
                for slot in batch:
                    generation_counts[slot.slot_id] = store.generation_count(slot.slot_id)
                by_id = {item.sample_id: item for item in generated.items if isinstance(item, Candidate)}
                candidates = []
                for slot in batch:
                    candidate = by_id.get(slot.slot_id)
                    if candidate is None or slot.slot_id in generated.invalid_ids:
                        reject(slot, ["invalid_generation_response",
                                      *generated.invalid_reasons.get(slot.slot_id, [])])
                        continue
                    reasons = validate_candidate(candidate, slot, label_names, inspector)
                    if not reasons:
                        duplicate = index.check(candidate.text, candidate.locale, candidate.labels,
                                                slot.group_id, slot.split)
                        if duplicate:
                            reasons.append(duplicate)
                    if reasons:
                        reject(slot, reasons)
                        continue
                    candidates.append(candidate)
                if not candidates:
                    continue
                reviewed = call_with_recovery("judge", candidates)
                verdicts = {item.sample_id: item for item in reviewed.items if isinstance(item, Judgement)}
                for candidate in candidates:
                    slot = lookup[candidate.sample_id]
                    verdict = verdicts.get(candidate.sample_id)
                    if verdict is None or candidate.sample_id in reviewed.invalid_ids:
                        reject(slot, ["invalid_judge_response",
                                      *reviewed.invalid_reasons.get(slot.slot_id, [])])
                        continue
                    reasons = validate_judgement(verdict, candidate, label_names)
                    canonical_labels = sorted(candidate.labels, key=label_names.index)
                    if not reasons:
                        duplicate = index.check(candidate.text, candidate.locale, canonical_labels,
                                                slot.group_id, slot.split)
                        if duplicate:
                            reasons.append(duplicate)
                    if reasons:
                        reject(slot, reasons)
                        continue
                    record = {
                        "sample_id": slot.slot_id, "text": candidate.text, "labels": canonical_labels,
                        "locale": slot.locale, "split": slot.split, "group_id": slot.group_id,
                        "family_id": slot.family_id, "kind": slot.kind, "scenario": slot.scenario,
                        "source": "llm_synthetic", "label_version": catalog["version"],
                        "review_status": "machine_accepted", "n_tokens": inspector.count(candidate.text),
                        "generator_model": provider.metadata()["generator_model"],
                        "judge_model": provider.metadata()["judge_model"],
                        "generation_attempt": generation_counts[slot.slot_id],
                        "provider_profile_id": store.provider_profile_id,
                    }
                    store.accept(record)
                    index.add(record)
                    done.add(slot.slot_id)
        finally:
            # A keyboard interrupt or provider outage still exports all completed accepted data.
            report = export_dataset(store, slots, stage, catalog)
        return report


def verify_dataset(settings, catalog, inspector, stage: str) -> dict:
    directory = state_directory(settings, stage)
    if not (directory / "state.sqlite3").exists():
        raise DatasetError("No dataset found; run build first.")
    try:
        with FileLock(str(directory / ".run.lock"), timeout=0):
            return _verify_locked(settings, catalog, inspector, stage, directory)
    except Timeout:
        raise DatasetError("Build is running; verify after it finishes.") from None


def _verify_locked(settings, catalog, inspector, stage, directory):
    with Store(directory, None, resume=True) as store:
        identity = store.get_meta("identity")
        current = identity_for(settings, catalog, inspector, identity["provider"])
        if digest(current) != digest(identity):
            raise DatasetError("Dataset fingerprint changed; use the original config/code/assets to verify.")
        slots = plan_slots(settings, catalog, stage)
        records = selected_samples(store, slots)
        all_slots = slots if stage == "smoke" else plan_slots(settings, catalog, "full")
        errors = _validate_records(store.samples(), all_slots, catalog, inspector,
                                   settings.near_duplicate_threshold)
        frozen = store.get_meta("full_frozen_signature")
        if frozen is not None and frozen != sample_signature(store.samples()):
            errors.append({"reason_codes": ["frozen_dataset_changed"]})
        ids = {record["sample_id"] for record in records}
        for slot in slots:
            if slot.slot_id not in ids:
                errors.append({"sample_id": slot.slot_id, "reason_codes": ["missing_slot"]})
        export_dir = directory / "exports" / stage
        try:
            manifest = json.loads((export_dir / "manifest.json").read_text("utf-8"))
            if manifest["fingerprint"] != digest(identity) or manifest["sample_signature"] != sample_signature(records):
                errors.append({"reason_codes": ["manifest_mismatch"]})
            # Verify both stored hashes and actual equality to the accepted checkpoint.
            for split in sorted({slot.split for slot in slots}):
                for locale in (None, *settings.locales):
                    name = f"{locale}/{split}.jsonl" if locale else f"{split}.jsonl"
                    path = export_dir / name
                    expected = [record for record in records if record["split"] == split
                                and (locale is None or record["locale"] == locale)]
                    actual = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
                    if (actual != expected or file_digest(path) != manifest["files"][name]["sha256"]
                            or manifest["files"][name]["rows"] != len(expected)):
                        errors.append({"file": name, "reason_codes": ["export_mismatch"]})
            for filename, expected in (
                ("labels.json", catalog),
                ("label2id.json", {label["name"]: label["id"] for label in catalog["labels"]}),
                ("id2label.json", {str(label["id"]): label["name"] for label in catalog["labels"]}),
            ):
                if json.loads((export_dir / filename).read_text("utf-8")) != expected:
                    errors.append({"file": filename, "reason_codes": ["label_mapping_mismatch"]})
        except (OSError, ValueError, KeyError, TypeError):
            errors.append({"reason_codes": ["missing_or_invalid_export"]})
        result = {"stage": stage, "ok": not errors, "accepted": len(records), "expected": len(slots),
                  "errors": errors, "verified_at": now(), "sample_signature": sample_signature(records),
                  "checks": ["schema", "quota", "label_mapping", "token_lengths", "seed_family_isolation",
                             "exact_and_lexical_near_duplicates", "export_hashes", "checkpoint_parity"]}
        atomic_json(export_dir / "verification.json", result)
        if stage == "pilot":
            store.set_meta("pilot_verified_signature", sample_signature(records) if not errors else None)
        if stage == "full" and not errors:
            # No further generation is needed for a verified complete test set.
            store.set_meta("full_frozen_signature", sample_signature(records))
        return result
