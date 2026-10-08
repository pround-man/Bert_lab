from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from watch_intents.config import LABEL_NAMES, load_catalog, load_settings
from watch_intents.planning import plan_slots

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(ROOT)


@pytest.fixture(scope="module")
def settings():
    return load_settings(ROOT)


@pytest.fixture(scope="module")
def full(settings, catalog):
    return plan_slots(settings, catalog, "full")


def test_final_budget_and_label_balance(full):
    assert len(full) == 13500
    assert len({slot.slot_id for slot in full}) == 13500
    for locale in ("en", "zh-Hans", "th"):
        for split, total, per_label in (("train", 3000, 72), ("validation", 500, 12), ("test", 1000, 24)):
            subset = [slot for slot in full if slot.locale == locale and slot.split == split]
            assert len(subset) == total
            assert Counter(slot.kind for slot in subset) == {
                "single": total * 6 // 10, "multi": total * 2 // 10, "zero": total * 2 // 10,
            }
            assert Counter(slot.labels[0] for slot in subset if slot.kind == "single") == {
                name: per_label for name in LABEL_NAMES
            }
            assert all(2 <= len(slot.labels) <= 3 for slot in subset if slot.kind == "multi")
            assert all(not slot.labels for slot in subset if slot.kind == "zero")
            assert sum(slot.scenario == "chitchat" for slot in subset) == total // 10


def test_pilot_is_reusable_training_prefix(settings, catalog, full):
    pilot = plan_slots(settings, catalog, "pilot")
    assert len(pilot) == 300
    full_by_id = {slot.slot_id: slot for slot in full}
    for locale in settings.locales:
        subset = [slot for slot in pilot if slot.locale == locale]
        assert len(subset) == 100
        assert Counter(slot.kind for slot in subset) == {"single": 60, "multi": 20, "zero": 20}
        assert {slot.labels[0] for slot in subset if slot.kind == "single"} == set(LABEL_NAMES)
    assert all(slot.split == "train" and full_by_id[slot.slot_id] == slot for slot in pilot)


def test_smoke_has_nine_distinct_non_dataset_roots(settings, catalog, full):
    smoke = plan_slots(settings, catalog, "smoke")
    assert len(smoke) == 9
    assert {slot.slot_id for slot in smoke}.isdisjoint(slot.slot_id for slot in full)
    for locale in settings.locales:
        assert Counter(slot.kind for slot in smoke if slot.locale == locale) == {
            "single": 1, "multi": 1, "zero": 1,
        }


def test_reproducibility_and_seed_change(settings, catalog):
    first = plan_slots(settings, catalog, "pilot")
    assert first == plan_slots(settings, catalog, "pilot")
    changed = plan_slots(replace(settings, seed=settings.seed + 1), catalog, "pilot")
    assert first != changed
    assert {slot.slot_id for slot in first}.isdisjoint(slot.slot_id for slot in changed)


def test_split_origins_and_held_out_combinations_do_not_overlap(full):
    groups = defaultdict(set)
    families = defaultdict(set)
    combinations = defaultdict(set)
    challenge = set()
    for slot in full:
        groups[slot.group_id].add(slot.split)
        families[slot.family_id].add(slot.split)
        if slot.kind == "multi":
            combinations[slot.split].add(frozenset(slot.labels))
            if slot.scenario == "unseen_combination":
                challenge.add(frozenset(slot.labels))
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(len(splits) == 1 for splits in families.values())
    assert challenge
    assert challenge.isdisjoint(combinations["train"])
    assert challenge.isdisjoint(combinations["validation"])
    assert combinations["validation"] <= combinations["train"]
    assert (combinations["test"] - challenge) <= combinations["train"]
    assert sum(slot.scenario == "unseen_combination" for slot in full) == 120
    # Separation is also carried in generation instructions, not just UUIDs.
    for split in ("train", "validation", "test"):
        assert all(slot.seed_brief and "Compose natively" in slot.seed_brief for slot in full if slot.split == split)


@pytest.mark.parametrize("config", [
    'mystery = true\n',
    'version = "0.2"\n',
    'locales = ["en", "th"]\n',
    'seed = true\n',
    'semantic_retries = 3\n',
    'max_tokens = 513\n',
    'near_duplicate_threshold = nan\n',
    'near_duplicate_threshold = true\n',
    '[paths]\nsecret_key = "should-not-be-here"\n',
    '[split_sizes]\ntrain = 101\nvalidation = 50\ntest = 50\n',
    '[split_sizes]\ntrain = 100\nvalidation = 0\ntest = 50\n',
    '[zero_scenario_weights]\nchitchat = 100\n',
])
def test_invalid_config_fails_closed(tmp_path, config):
    path = tmp_path / "bad.toml"
    path.write_text(config, encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(tmp_path, path)


def test_config_fingerprint_is_portable_and_sensitive_to_behavior(settings):
    relocated = replace(settings, root=Path("/different"), output_dir=Path("/elsewhere"), model_dir=Path("/model"))
    assert settings.fingerprint_payload() == relocated.fingerprint_payload()
    assert settings.fingerprint_payload() != replace(settings, max_tokens=256).fingerprint_payload()
    assert json.loads(json.dumps(settings.fingerprint_payload())) == settings.fingerprint_payload()


@pytest.mark.parametrize("mutation", ["order", "id", "version", "locale", "example"])
def test_invalid_catalog_rejected(tmp_path, catalog, mutation):
    changed = json.loads(json.dumps(catalog))
    if mutation == "order":
        changed["labels"][0], changed["labels"][1] = changed["labels"][1], changed["labels"][0]
    elif mutation == "id":
        changed["labels"][0]["id"] = False
    elif mutation == "version":
        changed["version"] = "2"
    elif mutation == "locale":
        del changed["labels"][0]["examples"]["th"]
    else:
        changed["labels"][0]["examples"]["en"]["positive"] = ""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/labels.v0.1.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalog(tmp_path)


def test_zero_weights_change_scenarios_without_changing_ratios(settings, catalog):
    weights = {key: 0 for key in settings.zero_scenario_weights}
    weights["howto"] = 100
    slots = plan_slots(replace(settings, zero_scenario_weights=weights), catalog, "pilot")
    assert {slot.scenario for slot in slots if slot.kind == "zero"} == {"howto"}
    assert Counter(slot.kind for slot in slots) == {"single": 180, "multi": 60, "zero": 60}


def test_unknown_stage_is_an_error(settings, catalog):
    with pytest.raises(ValueError):
        plan_slots(settings, catalog, "all")
