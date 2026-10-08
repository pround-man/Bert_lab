"""Allocate immutable semantic roots before any model call is made."""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class Slot:
    slot_id: str
    locale: str
    split: str
    kind: str
    labels: tuple[str, ...]
    scenario: str
    group_id: str
    family_id: str
    seed_brief: str


_STYLES = {
    "train": (
        ("direct", "Use a direct everyday imperative with an explicit action verb."),
        ("polite", "Use a short polite request addressed to the watch."),
        ("casual", "Use casual spoken phrasing with the object stated clearly."),
        ("urgent", "Use a brief urgent request; urgency is not an additional action."),
    ),
    "validation": (
        ("purpose_first", "Lead with a practical purpose, then state the requested action."),
        ("observation", "Begin with a short observation, then explicitly request the action."),
        ("hesitation", "Use a natural hesitation followed by an unambiguous complete request."),
        ("courtesy_tail", "State the action first and put a courtesy phrase at the end."),
    ),
    "test": (
        ("background_sandwich", "Place the operative request between two irrelevant background remarks."),
        ("digression", "Use a brief digression, return to the watch, and make the request explicit."),
        ("contrast", "Contrast irrelevant circumstances before stating the desired watch action."),
        ("reflective", "Begin with a personal reflection and end with the clear watch request."),
    ),
    "smoke": (("smoke", "Use a short, natural and unambiguous spoken utterance."),),
}

_CONTEXTS = (
    "taking a quiet break at home", "getting ready to leave home", "resting after a walk",
    "sitting on a park bench", "waiting near the office entrance", "finishing household chores",
    "putting on a jacket", "taking a break from reading", "returning from the grocery store",
    "standing in the hallway", "packing a light day bag", "resting in the living room",
    "getting settled at a desk", "waiting for a friend", "taking a pause in a conversation",
    "coming back indoors", "preparing for an ordinary day", "sitting down after tidying up",
    "having a quiet moment outside", "checking the watch before heading out",
)

_SINGLE_SCENARIOS = (
    "direct", "polite", "background", "repeat_same", "correction", "negated_distractor",
)
_MULTI_SCENARIOS = (
    "direct_multiple", "second_in_background", "partial_withdrawal", "repeat_one", "reordered",
)
_SCENARIO_GUIDANCE = {
    "direct": "State exactly the target request.",
    "polite": "Make exactly the target request with natural courtesy.",
    "background": "Include irrelevant background without adding any other supported request.",
    "repeat_same": "Repeat the same target request naturally; repetitions produce one label.",
    "correction": "Mention a different action, explicitly cancel it, then request only the target action.",
    "negated_distractor": "Explicitly prohibit a different action, then request the target action; prohibitions are not inverse commands.",
    "direct_multiple": "Explicitly request every target action; do not add other actions.",
    "second_in_background": "Put one target request early and another after irrelevant background; every target remains active.",
    "partial_withdrawal": "Also mention one action outside the target set and explicitly retract only that action; all target actions remain active.",
    "repeat_one": "Request all target actions and repeat one of them; output remains a set.",
    "reordered": "Request target actions in a different order from the provided label order; do not cancel any of them.",
    "unseen_combination": "This is a held-out label combination challenge. Explicitly request all target actions naturally; every target remains active. Do not mention that this is a test.",
    "chitchat": "Write ordinary small talk or a personal remark, with no request to operate the watch.",
    "negation": "Only prohibit one or more watch actions. Do not request their inverse or any other action.",
    "withdrawal": "Start a watch request, then clearly retract the entire request and make no replacement request.",
    "quoted": "Quote or report a watch command as a discussion topic; explicitly avoid instructing the watch to execute it now.",
    "past": "Describe an action that already happened in the past, with no present request.",
    "howto": "Ask for instructions, an explanation, or an on/off status; do not ask the watch to perform an operation.",
    "unclear": "Refer to an unspecified object or feature without prior context. The absence of a recoverable supported intent must be certain: this is a deliberate out-of-scope reference, not competing plausible supported labels.",
    "unsupported_parameter": "Only request an unsupported execution or parameterized operation, such as setting a timed alarm, starting a timed countdown, recording a run, selecting a route, or playing a specific song. Do not separately request a supported page.",
}


def _rng(seed: int, *parts: object) -> random.Random:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _combination_pools(names: tuple[str, ...], seed: int) -> dict:
    """Reserve label SETS for a separate test challenge slice."""
    pools = {split: {} for split in ("eligible", "heldout")}
    for size in (2, 3):
        combinations = list(itertools.combinations(names, size))
        _rng(seed, "combination-holdout", size).shuffle(combinations)
        eligible_end = len(combinations) * 85 // 100
        pools["eligible"][size] = combinations[:eligible_end]
        pools["heldout"][size] = combinations[eligible_end:]
    return pools


def _weighted_sequence(weights: dict[str, int], length: int) -> list[str]:
    """Largest current deficit keeps prefixes representative and resumes stable."""
    counts: Counter = Counter()
    choices = sorted(key for key, weight in weights.items() if weight > 0)
    sequence = []
    for index in range(length):
        choice = max(choices, key=lambda key: (weights[key] * (index + 1) - 100 * counts[key], key))
        sequence.append(choice)
        counts[choice] += 1
    return sequence


def _multi_sequence(pools: dict[int, list[tuple[str, ...]]], length: int, rng: random.Random) -> list[tuple[str, ...]]:
    pools = {size: list(items) for size, items in pools.items()}
    for items in pools.values():
        rng.shuffle(items)
    counts: Counter = Counter()
    used: Counter = Counter()
    result = []
    for index in range(length):
        size = 3 if index % 5 == 4 else 2
        # Prefer diverse combinations, then balance marginal label coverage.
        combo = min(pools[size], key=lambda item: (used[item], sum(counts[name] for name in item)))
        used[combo] += 1
        counts.update(combo)
        result.append(combo)
    return result


def _make_slot(settings: Settings, locale: str, split: str, index: int, kind: str,
               labels: tuple[str, ...], scenario: str, rng: random.Random) -> Slot:
    fingerprint = hashlib.sha256(str(settings.seed).encode()).hexdigest()[:8]
    slot_id = f"{split}-{locale}-{fingerprint}-{index:06d}"
    style_name, style_instruction = _STYLES[split][rng.randrange(len(_STYLES[split]))]
    context = _CONTEXTS[rng.randrange(len(_CONTEXTS))]
    root_variation = rng.randrange(10**9)
    brief = (
        f"Native locale: {locale}. Independent semantic root {slot_id}; variation key {root_variation}. "
        f"Expression family: {style_instruction} Context, only if helpful: {context}. "
        f"Scenario: {_SCENARIO_GUIDANCE[scenario]} "
        "Compose natively rather than translating a stored example. Context is background, never a time/location/route parameter. "
        "Do not copy catalog examples, identifiers, variation keys, or this instruction into the utterance. "
        "Use natural vocabulary and vary syntax. Preserve the full final request set."
    )
    return Slot(
        slot_id=slot_id, locale=locale, split=split, kind=kind, labels=labels,
        scenario=scenario, group_id=f"group-{slot_id}",
        family_id=f"{split}-{style_name}", seed_brief=brief,
    )


def plan_slots(settings: Settings, catalog: dict, stage: str) -> list[Slot]:
    """Pilot slots are a stable training prefix; smoke never shares dataset IDs."""
    if stage not in {"pilot", "full", "smoke"}:
        raise ValueError("stage must be pilot, full, or smoke")
    names = tuple(entry["name"] for entry in catalog["labels"])
    if stage == "smoke":
        smoke = []
        for locale in settings.locales:
            rng = _rng(settings.seed, "smoke", locale)
            for index, (kind, labels, scenario) in enumerate((
                ("single", (names[13],), "direct"),
                ("multi", (names[1], names[7]), "direct_multiple"),
                ("zero", (), "chitchat"),
            )):
                smoke.append(_make_slot(settings, locale, "smoke", index, kind, labels, scenario, rng))
        return smoke
    pools = _combination_pools(names, settings.seed)
    result = []
    splits = ("train",) if stage == "pilot" else ("train", "validation", "test")
    for locale in settings.locales:
        training_multis = _multi_sequence(
            pools["eligible"], settings.split_sizes["train"] * 2 // 10,
            _rng(settings.seed, locale, "train", "multi"),
        )
        observed_training_pools = {
            size: sorted({item for item in training_multis if len(item) == size})
            for size in (2, 3)
        }
        for split in splits:
            total = settings.pilot_per_locale if stage == "pilot" else settings.split_sizes[split]
            rng = _rng(settings.seed, locale, split)
            singles = list(names)
            rng.shuffle(singles)
            # Always allocate using full size so pilot and full prefixes are identical.
            full_size = settings.split_sizes[split]
            multi_count = full_size * 2 // 10
            if split == "train":
                multis = training_multis
            elif split == "validation":
                multis = _multi_sequence(observed_training_pools, multi_count, _rng(settings.seed, locale, split, "multi"))
            else:
                heldout_count = multi_count // 5
                familiar = iter(_multi_sequence(observed_training_pools, multi_count - heldout_count, _rng(settings.seed, locale, split, "familiar")))
                heldout = iter(_multi_sequence(pools["heldout"], heldout_count, _rng(settings.seed, locale, split, "heldout")))
                multis = [next(heldout) if index % 5 == 4 else next(familiar) for index in range(multi_count)]
            zeros = _weighted_sequence(settings.zero_scenario_weights, full_size * 2 // 10)
            seen: Counter = Counter()
            for index in range(total):
                position = index % 10
                kind = "single" if position < 6 else "multi" if position < 8 else "zero"
                sequence_index = seen[kind]
                if kind == "single":
                    labels = (singles[sequence_index % len(singles)],)
                    scenario = _SINGLE_SCENARIOS[sequence_index % len(_SINGLE_SCENARIOS)]
                elif kind == "multi":
                    labels = multis[sequence_index]
                    scenario = (
                        "unseen_combination" if split == "test" and sequence_index % 5 == 4
                        else _MULTI_SCENARIOS[sequence_index % len(_MULTI_SCENARIOS)]
                    )
                else:
                    labels = ()
                    scenario = zeros[sequence_index]
                result.append(_make_slot(settings, locale, split, index, kind, labels, scenario, rng))
                seen[kind] += 1
    return result
