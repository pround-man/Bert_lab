"""Deterministic exports and descriptive reports, never claims of production accuracy."""

from collections import Counter, defaultdict

from .storage import Store, atomic_json, atomic_text, canonical, digest, file_digest, now


def selected_samples(store: Store, slots: list) -> list[dict]:
    ids = {slot.slot_id for slot in slots}
    return [record for record in store.samples() if record["sample_id"] in ids]


def sample_signature(records: list[dict]) -> str:
    return digest(sorted(records, key=lambda record: record["sample_id"]))


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return round(values[left] + (values[right] - values[left]) * (index - left), 2)


def build_report(store: Store, slots: list, stage: str, label_names: list[str]) -> dict:
    records = selected_samples(store, slots)
    ids = {slot.slot_id for slot in slots}
    by_id = {record["sample_id"]: record for record in records}
    missing = [slot for slot in slots if slot.slot_id not in by_id]
    events = [event for event in store.events() if event["sample_id"] in ids]
    counts = Counter((r["locale"], r["split"], r["kind"]) for r in records)
    planned = Counter((s.locale, s.split, s.kind) for s in slots)
    coverage = []
    for (locale, split, kind), expected in sorted(planned.items()):
        accepted = counts[locale, split, kind]
        coverage.append({"locale": locale, "split": split, "kind": kind, "expected": expected,
                         "accepted": accepted, "missing": expected - accepted})
    languages = {}
    for locale in sorted({s.locale for s in slots}):
        rows = [r for r in records if r["locale"] == locale]
        lengths = [r["n_tokens"] for r in rows]
        supports = Counter(label for r in rows for label in r["labels"])
        locale_rejects = [e for e in events if e["locale"] == locale]
        denominator = len(rows) + len(locale_rejects)
        languages[locale] = {
            "accepted": len(rows), "rejected_attempts": len(locale_rejects),
            "accepted_fraction_of_decided_candidates": len(rows) / denominator if denominator else None,
            "label_support": {label: supports[label] for label in label_names},
            "single_label_support_by_split": {
                split: dict(Counter(r["labels"][0] for r in rows if r["split"] == split and r["kind"] == "single"))
                for split in sorted({s.split for s in slots})},
            "token_lengths": {"p50": percentile(lengths, .50), "p95": percentile(lengths, .95),
                              "p99": percentile(lengths, .99), "max": max(lengths, default=None),
                              "over_128": sum(n > 128 for n in lengths),
                              "over_256": sum(n > 256 for n in lengths)},
            "scenarios": dict(Counter(r["scenario"] for r in rows)),
            "rejected_scenarios": dict(Counter(e["scenario"] for e in locale_rejects)),
            "multilabel_combinations": dict(Counter("|".join(r["labels"]) for r in rows if r["kind"] == "multi")),
        }
    usage = Counter()
    calls = [call for call in store.calls() if set(call["sample_ids"]) & ids]
    for call in calls:
        usage.update(call["usage"])
    return {
        "stage": stage, "evaluation_scope": "synthetic_only_not_production_acceptance",
        "expected": len(slots), "accepted": len(records), "missing": len(missing),
        "complete": not missing, "coverage": coverage, "languages": languages,
        "execution_runs": store.get_meta("execution_runs") or [],
        "provider_runs": store.get_meta("provider_runs") or [],
        "rejection_reasons": dict(Counter(code for e in events for code in e["reason_codes"])),
        "missing_slots": [{"sample_id": s.slot_id, "locale": s.locale, "split": s.split,
                           "kind": s.kind, "labels": list(s.labels), "scenario": s.scenario,
                           "rejections": store.rejection_count(s.slot_id),
                           "generation_attempts": store.generation_count(s.slot_id)} for s in missing],
        "api_usage": {"reported_tokens": dict(usage),
                      "reported_request_count": sum(call["request_count"] or 0 for call in calls),
                      "unknown_usage_calls": sum("total_tokens" not in call["usage"] for call in calls),
                      "unfinished_or_failed_calls": sum(call["status"] != "completed" for call in calls),
                      "note": "Provider-reported usage only; timed-out/retried requests may incur additional cost."},
        "near_duplicate_policy": "NFC/whitespace exact matching; fuzzy wording check within locale and identical label sets; short expressions and semantic equivalence require separate review.",
    }


def export_dataset(store: Store, slots: list, stage: str, catalog: dict) -> dict:
    directory = store.directory / "exports" / stage
    rows = selected_samples(store, slots)
    groups = defaultdict(list)
    # Include empty expected files so absence cannot be mistaken for success.
    for slot in slots:
        groups[f"{slot.split}.jsonl"]
        groups[f"{slot.locale}/{slot.split}.jsonl"]
    for row in rows:
        groups[f"{row['split']}.jsonl"].append(row)
        groups[f"{row['locale']}/{row['split']}.jsonl"].append(row)
    files = {}
    for name, records in sorted(groups.items()):
        path = directory / name
        atomic_text(path, "".join(canonical(record) + "\n" for record in records))
        files[name] = {"rows": len(records), "sha256": file_digest(path)}
    atomic_json(directory / "labels.json", catalog)
    atomic_json(directory / "label2id.json", {label["name"]: label["id"] for label in catalog["labels"]})
    atomic_json(directory / "id2label.json", {str(label["id"]): label["name"] for label in catalog["labels"]})
    slot_ids = {slot.slot_id for slot in slots}
    atomic_text(directory / "rejections.jsonl", "".join(canonical(event) + "\n" for event in store.events()
                if event["sample_id"] in slot_ids))
    report = build_report(store, slots, stage, [label["name"] for label in catalog["labels"]])
    atomic_json(directory / "report.json", report)
    manifest = {"created_at": now(), "dataset_created_at": store.get_meta("created_at"),
                "identity": store.get_meta("identity"), "fingerprint": digest(store.get_meta("identity")),
                "stage": stage, "source": "llm_synthetic", "review": "machine_only",
                "accepted": len(rows), "expected": len(slots), "complete": report["complete"],
                "sample_signature": sample_signature(rows), "files": files,
                "identity_history": store.get_meta("identity_history") or [],
                "provider_runs": store.get_meta("provider_runs") or [],
                "test_policy": "Freeze after data verification; never tune against synthetic test results."}
    atomic_json(directory / "manifest.json", manifest)
    return report
