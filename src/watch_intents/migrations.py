"""Offline, backed-up migrations for explicitly supported predecessor versions."""

import json
import sqlite3
import time
from copy import deepcopy

from filelock import FileLock, Timeout

from . import pipeline
from .models import DatasetError
from .reports import export_dataset, sample_signature, selected_samples
from .storage import Store, canonical, digest, file_digest, now, same_dataset_contract

LEGACY_CODE_SHA256 = "28c10f38a657d4902f92268d2eea242f6d890ba1dde8261bc5371c0621b7097c"
SERIAL_CODE_SHA256 = "31bc2c4f7e03c20bb796faaf1848e74e30993557e656c7d4a371ffaef9b2ac7a"
PARALLEL_CODE_SHA256 = "e67c8b0ff1267847dfea6abd37db503d3a90d49d22072dda637960a8e683fb3d"
RUNTIME_FILE_CODE_SHA256 = "5b5b39eb6dc37d345806ba147ae5402b576add68d151c0ff8e081378606db79c"


def upgrade_dataset(settings, catalog, inspector, provider_metadata):
    directory = pipeline.state_directory(settings, "pilot")
    if not (directory / "state.sqlite3").exists():
        raise DatasetError("No checkpoint to upgrade.")
    current = pipeline.identity_for(settings, catalog, inspector, provider_metadata)
    try:
        with FileLock(str(directory / ".run.lock"), timeout=0), Store(directory, None, resume=True) as store:
            previous = store.get_meta("identity")
            if same_dataset_contract(previous, current):
                return {"status": "already_current", "accepted": len(store.samples())}
            expected = deepcopy(previous)
            known = {LEGACY_CODE_SHA256: "v0.1", SERIAL_CODE_SHA256: "v0.2",
                     PARALLEL_CODE_SHA256: "v0.2", RUNTIME_FILE_CODE_SHA256: "v0.2"}
            if (expected.get("code_sha256") not in known
                    or expected.get("provider", {}).get("prompt_version") != known[expected["code_sha256"]]):
                raise DatasetError("Unsupported checkpoint version; automatic migration refused.")
            expected["code_sha256"] = current["code_sha256"]
            expected["provider"]["prompt_version"] = "v0.2"
            if not same_dataset_contract(expected, current):
                raise DatasetError("Migration requires unchanged prompts/config/catalog/assets/dependencies.")
            if store.get_meta("full_frozen_signature") is not None:
                raise DatasetError("A frozen full dataset cannot be migrated in place.")
            records = store.samples()
            all_slots = pipeline.plan_slots(settings, catalog, "full")
            errors = pipeline._validate_records(records, all_slots, catalog, inspector,
                                                settings.near_duplicate_threshold)
            if errors or any(
                record.get(f"{role}_model") != previous["provider"][f"{role}_model"]
                for record in records for role in ("generator", "judge")
            ):
                raise DatasetError("Accepted checkpoint failed offline validation; migration refused.")
            stages = ["pilot"]
            if (directory / "exports/full/manifest.json").exists():
                stages.append("full")
            for stage in stages:
                output = directory / "exports" / stage
                try:
                    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
                    rows = selected_samples(store, pipeline.plan_slots(settings, catalog, stage))
                    if (manifest["fingerprint"] != digest(previous)
                            or manifest["sample_signature"] != sample_signature(rows)):
                        raise ValueError("manifest mismatch")
                    for name, metadata in manifest["files"].items():
                        # Only exporter-owned relative JSONL names are allowed.
                        path = (output / name).resolve()
                        if not path.is_relative_to(output.resolve()) or path.suffix != ".jsonl":
                            raise ValueError("invalid export path")
                        if file_digest(path) != metadata["sha256"]:
                            raise ValueError("export changed")
                except (OSError, ValueError, KeyError, TypeError):
                    raise DatasetError("Existing exports failed integrity checks; migration refused.") from None
            backup = directory / "backups" / f"before-concurrency-{time.time_ns()}.sqlite3"
            backup.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(backup) as target:
                store.db.backup(target)
            history = store.get_meta("identity_history") or []
            reason = ("runtime_configuration_file" if previous["code_sha256"] == PARALLEL_CODE_SHA256
                      else "parallel_generation_and_review")
            if previous["code_sha256"] == RUNTIME_FILE_CODE_SHA256:
                reason = "provider_configuration_as_provenance"
            history.append({"at": now(), "reason": reason,
                            "previous_identity": previous, "accepted": len(records),
                            "sample_signature": sample_signature(records), "backup": str(backup)})
            with store.db:
                for key, value in (("identity", expected), ("identity_history", history),
                                   ("pilot_verified_signature", None)):
                    store.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value)))
            for stage in stages:
                export_dataset(store, pipeline.plan_slots(settings, catalog, stage), stage, catalog)
            return {"status": "upgraded", "accepted": len(records), "backup": str(backup),
                    "sample_signature": sample_signature(records), "api_calls": 0}
    except Timeout:
        raise DatasetError("Build is running; stop it before upgrading the checkpoint.") from None
