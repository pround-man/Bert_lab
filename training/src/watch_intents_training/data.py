"""Read frozen exports without importing or changing the generation pipeline."""

import json

import numpy as np

from .io import digest, file_hash, read_json, verify_files


class FrozenData:
    def __init__(self, config):
        self.root = config.data_dir
        self.manifest = read_json(self.root / "manifest.json")
        verification = read_json(self.root / "verification.json")
        m = self.manifest
        if (m.get("stage") != "full" or m.get("complete") is not True
                or m.get("accepted") != m.get("expected")
                or verification.get("ok") is not True or verification.get("errors") != []
                or verification.get("sample_signature") != m.get("sample_signature")
                or verification.get("accepted") != m["accepted"]
                or digest(m["identity"]) != m["fingerprint"]):
            raise ValueError("Dataset must have a consistent frozen full manifest and verification")
        self.catalog = read_json(self.root / "labels.json")
        entries = self.catalog["labels"]
        if len(entries) != 25 or [x["id"] for x in entries] != list(range(25)):
            raise ValueError("Expected 25 consecutive fixed label IDs")
        self.labels = [x["name"] for x in entries]
        self.label2id = {name: i for i, name in enumerate(self.labels)}
        if (len(self.label2id) != 25
                or read_json(self.root / "label2id.json") != self.label2id
                or read_json(self.root / "id2label.json") != {str(i): n for i, n in enumerate(self.labels)}
                or digest(self.catalog) != m["identity"]["catalog_sha256"]):
            raise ValueError("Label mapping or catalog mismatch")
        for split in ("train", "validation", "test"):
            if f"{split}.jsonl" not in m["files"]:
                raise ValueError(f"Missing split: {split}")
        verify_files(self.root, {name: info["sha256"] for name, info in m["files"].items()})
        if sum(m["files"][f"{s}.jsonl"]["rows"] for s in ("train", "validation", "test")) != m["accepted"]:
            raise ValueError("Manifest split counts do not sum to accepted count")
        assets = m["identity"]["assets"]
        for name, key in (("tokenizer.json", "tokenizer_sha256"), ("config.json", "model_config_sha256")):
            if file_hash(config.model_dir / name) != assets[key]:
                raise ValueError(f"Base model/tokenizer mismatch: {name}")
        if (config.model_dir / "model.safetensors").stat().st_size != assets["weights_size_bytes"]:
            raise ValueError("Base weights size mismatch")
        self.identity = {"manifest_sha256": file_hash(self.root / "manifest.json"),
                         "sample_signature": m["sample_signature"],
                         "catalog_sha256": digest(self.catalog)}

    def read_split(self, split):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        path = self.root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows or len(rows) != self.manifest["files"][path.name]["rows"]:
            raise ValueError(f"Invalid row count in {split}")
        seen = set()
        targets = np.zeros((len(rows), 25), dtype=np.float32)
        for i, row in enumerate(rows):
            labels = row.get("labels")
            if (not isinstance(row.get("text"), str) or not row["text"].strip()
                    or row.get("split") != split or row.get("locale") not in ("en", "zh-Hans", "th")
                    or row.get("review_status") != "machine_accepted"
                    or row.get("label_version") != self.catalog["version"]
                    or not isinstance(labels, list) or any(x not in self.label2id for x in labels)
                    or len(labels) != len(set(labels))
                    or labels != sorted(labels, key=self.label2id.get)
                    or not isinstance(row.get("sample_id"), str) or not row["sample_id"]
                    or row["sample_id"] in seen
                    or not isinstance(row.get("group_id"), str) or not row["group_id"]
                    or not isinstance(row.get("scenario"), str) or not row["scenario"]):
                raise ValueError(f"Invalid sample at {split} row {i + 1}")
            expected_kind = "zero" if not labels else "single" if len(labels) == 1 else "multi"
            if row.get("kind") != expected_kind:
                raise ValueError("Sample kind does not match labels")
            targets[i, [self.label2id[x] for x in labels]] = 1
            seen.add(row["sample_id"])
        return rows, targets


def ensure_disjoint(*splits):
    seen_ids, seen_groups = set(), set()
    for rows in splits:
        ids = {r["sample_id"] for r in rows}
        groups = {r["group_id"] for r in rows}
        if seen_ids & ids or seen_groups & groups:
            raise ValueError("Cross-split sample/group leakage")
        seen_ids |= ids
        seen_groups |= groups


def encode_texts(tokenizer, texts, max_length):
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Texts must be nonempty strings")
    encoded = tokenizer(texts, truncation=False, padding=False, return_token_type_ids=False)
    if any(len(ids) > max_length for ids in encoded["input_ids"]):
        raise ValueError(f"Input exceeds validated length {max_length}; truncation is disabled")
    return [{key: value[i] for key, value in encoded.items()} for i in range(len(texts))]
