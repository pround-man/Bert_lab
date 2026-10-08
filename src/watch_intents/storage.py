"""Checkpoint only accepted text; rejected text never enters a file or database."""

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from .models import DatasetError


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


CONNECTION_FIELDS = ("generator_model", "judge_model", "generator_base_url", "judge_base_url")


def same_dataset_contract(left: dict, right: dict) -> bool:
    """Connection choices are provenance; prompts and all quality rules remain locked."""
    def contract(identity):
        result = deepcopy(identity)
        for key in CONNECTION_FIELDS:
            result.get("provider", {}).pop(key, None)
        return result
    return digest(contract(left)) == digest(contract(right))


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class Store:
    def __init__(self, directory: Path, identity: dict | None, *, resume: bool = False):
        path = directory / "state.sqlite3"
        existed = path.exists()
        if existed and not resume:
            raise DatasetError("Dataset already exists; use --resume. Smoke cannot be rerun here.")
        if identity is None and not existed:
            raise DatasetError("No dataset state found; run build first.")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.provider_profile_id = None
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (sample_id TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (sample_id TEXT PRIMARY KEY, rejections INTEGER NOT NULL,
                                                generations INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, record TEXT NOT NULL);
        """)
        stored = self.get_meta("identity")
        if stored is None and identity is not None:
            self.set_meta("identity", identity)
            self.set_meta("created_at", now())
        elif identity is not None and not same_dataset_contract(stored, identity):
            self.db.close()
            raise DatasetError("Dataset fingerprint changed (config/catalog/prompts/assets/code/dependencies). "
                               "Use a new output directory for a new version; old data was not changed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value: object) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value)))

    def samples(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT record FROM samples ORDER BY sample_id")]

    def record_provider(self, metadata: dict, stage: str) -> None:
        safe = {key: metadata[key] for key in (*CONNECTION_FIELDS, "prompt_version", "max_response_tokens")
                if key in metadata}
        self.provider_profile_id = digest(safe)
        runs = self.get_meta("provider_runs") or []
        runs.append({"at": now(), "stage": stage, "profile_id": self.provider_profile_id, "provider": safe})
        self.set_meta("provider_runs", runs)

    def accept(self, record: dict) -> None:
        with self.db:
            # No overwrite: a resumed run must never change an accepted label or text.
            self.db.execute("INSERT INTO samples VALUES (?,?)",
                            (record["sample_id"], canonical(record)))

    def rejection_count(self, sample_id: str) -> int:
        row = self.db.execute("SELECT rejections FROM attempts WHERE sample_id=?", (sample_id,)).fetchone()
        return row[0] if row else 0

    def generation_count(self, sample_id: str) -> int:
        row = self.db.execute("SELECT generations FROM attempts WHERE sample_id=?", (sample_id,)).fetchone()
        return row[0] if row else 0

    def reject(self, sample_id: str, reason_codes: list[str], metadata: dict) -> None:
        # Whitelist metadata: do not allow model-supplied text/reasoning into rejection records.
        safe = {key: metadata[key] for key in ("locale", "split", "kind", "scenario") if key in metadata}
        event = {"sample_id": sample_id, "reason_codes": reason_codes, "at": now(), **safe}
        with self.db:
            self.db.execute("INSERT INTO attempts VALUES (?,1,0) ON CONFLICT(sample_id) "
                            "DO UPDATE SET rejections=rejections+1", (sample_id,))
            self.db.execute("INSERT INTO events(record) VALUES (?)", (canonical(event),))

    def events(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT record FROM events ORDER BY id")]

    def begin_call(self, role: str, sample_ids: list[str]) -> int:
        record = {"role": role, "sample_ids": sample_ids, "status": "started", "started_at": now(),
                  "usage": {}, "request_count": None, "provider_profile_id": self.provider_profile_id}
        with self.db:
            if role == "generator":
                # Reserve before network I/O: crashes and judge outages cannot reset the paid budget.
                self.db.executemany("INSERT INTO attempts VALUES (?,0,1) ON CONFLICT(sample_id) "
                                    "DO UPDATE SET generations=generations+1", [(item,) for item in sample_ids])
            cursor = self.db.execute("INSERT INTO calls(record) VALUES (?)", (canonical(record),))
        return cursor.lastrowid

    def finish_call(self, call_id: int, usage: dict, request_count: int,
                    *, status: str = "completed", error_type: str | None = None) -> None:
        record = json.loads(self.db.execute("SELECT record FROM calls WHERE id=?", (call_id,)).fetchone()[0])
        record.update(status=status, usage=usage, request_count=request_count, finished_at=now())
        if error_type:
            record["error_type"] = error_type
        with self.db:
            self.db.execute("UPDATE calls SET record=? WHERE id=?", (canonical(record), call_id))

    def calls(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT record FROM calls ORDER BY id")]
