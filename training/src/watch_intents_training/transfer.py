"""Allowlisted transfer archives; credentials and build state never enter them."""

import io
import json
import tarfile
from pathlib import Path

from .io import file_hash, read_json, verify_files


def pack(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Archive already exists; choose a new output name")
    folders = (root / "training", root / "models/xlm-roberta-base",
               root / "data/generated/dataset/exports/full")
    excluded = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".cache", "dist", "build"}
    files = []
    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Missing transfer input: {folder}")
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(root)
            if excluded.intersection(relative.parts) or path.name.endswith((".pyc", ".env")):
                continue
            if path.is_symlink():
                raise ValueError(f"Symlinks are not allowed in transfer inputs: {relative}")
            if path.is_file():
                files.append(path)
    for name in ("docs/training-implementation.md", "README.md", ".python-version"):
        if (root / name).is_file():
            files.append(root / name)
    if any(output == p or output.is_relative_to(folder) for p in files for folder in folders):
        raise ValueError("Write transfer archives outside their input directories")
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in files}
    manifest = json.dumps({"format_version": 1, "files": hashes}, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname="Bert_lab/" + path.relative_to(root).as_posix(), recursive=False)
        info = tarfile.TarInfo("Bert_lab/TRANSFER.json")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    temporary.replace(output)
    return output


def verify_transfer(root):
    root = Path(root)
    manifest = read_json(root / "TRANSFER.json")
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported transfer manifest")
    verify_files(root, manifest["files"])
    return {"ok": True, "files": len(manifest["files"])}
