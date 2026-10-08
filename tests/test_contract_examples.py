"""Regression contracts with fixed reviewer verdicts, not a learned semantic test."""

from types import SimpleNamespace

import pytest

from watch_intents.config import LABEL_NAMES
from watch_intents.models import Candidate, Judgement
from watch_intents.quality import validate_candidate, validate_judgement


@pytest.mark.parametrize("locale,text,expected", [
    ("zh-Hans", "不要打开蓝牙", []),
    ("zh-Hans", "打开蓝牙，算了别打开了", []),
    ("zh-Hans", "打开蓝牙和手电筒，手电筒还是别开了", ["OPEN_BLUETOOTH"]),
    ("zh-Hans", "打开蓝牙，不，改成关闭蓝牙", ["CLOSE_BLUETOOTH"]),
    ("zh-Hans", "帮我设一个七点的闹钟", []),
    ("en", "Do not turn on Bluetooth.", []),
    ("en", "Open settings. Actually, cancel that.", []),
    ("en", "Turn on Bluetooth and the flashlight.", ["OPEN_BLUETOOTH", "OPEN_FLASHLIGHT"]),
    ("en", "Start a five-minute countdown.", []),
    ("th", "อย่าเปิดบลูทูธ", []),
    ("th", "เปิดบลูทูธให้หน่อย อ้อ ไม่ต้องเปิดแล้ว", []),
    ("th", "เปิดบลูทูธและไฟฉาย", ["OPEN_BLUETOOTH", "OPEN_FLASHLIGHT"]),
])
def test_final_effective_set_survives_validation_without_relabeling(locale, text, expected):
    candidate = Candidate(sample_id="case", text=text, locale=locale, labels=expected)
    slot = SimpleNamespace(slot_id="case", locale=locale, labels=tuple(expected))
    inspector = SimpleNamespace(max_tokens=512, count=lambda _: 20)
    verdict = Judgement(sample_id="case", labels=expected, language_ok=True,
                        naturalness_ok=True, ambiguous=False, reason_codes=[])
    assert validate_candidate(candidate, slot, list(LABEL_NAMES), inspector) == []
    assert validate_judgement(verdict, candidate, list(LABEL_NAMES)) == []
    assert candidate.text == text
    assert candidate.labels == expected


def test_reviewer_disagreement_on_negation_is_rejected_not_relabelled():
    candidate = Candidate(sample_id="case", text="Don't open Bluetooth.",
                          locale="en", labels=["OPEN_BLUETOOTH"])
    verdict = Judgement(sample_id="case", labels=[], language_ok=True,
                        naturalness_ok=True, ambiguous=False, reason_codes=[])
    assert validate_judgement(verdict, candidate, list(LABEL_NAMES)) == ["label_mismatch"]
    assert candidate.labels == ["OPEN_BLUETOOTH"]
