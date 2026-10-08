"""Small file primitives shared by the offline training commands."""

import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def directory_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def verify_files(directory, hashes):
    root = Path(directory).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing, changed or unsafe asset: {name}")
