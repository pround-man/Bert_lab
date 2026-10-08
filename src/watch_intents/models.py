"""Small, strict interfaces shared by the API and dataset pipeline."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sample_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    labels: list[str]
    locale: str


ReasonCode = Literal[
    "language_mismatch", "unnatural", "ambiguous", "unsupported_parameter",
    "label_mismatch", "missing_intent", "extra_intent", "other_quality",
]


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sample_id: str = Field(min_length=1)
    labels: list[str]
    language_ok: bool
    naturalness_ok: bool
    ambiguous: bool
    reason_codes: list[ReasonCode]


@dataclass
class BatchResult:
    items: list[Candidate | Judgement]
    invalid_ids: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    request_count: int = 1
    invalid_reasons: dict[str, list[str]] = field(default_factory=dict)


class DatasetError(RuntimeError):
    """User-actionable failure with an intentionally sanitized message."""


class FatalProviderError(DatasetError):
    pass


class TransientProviderError(DatasetError):
    pass


class CancelledProviderError(DatasetError):
    pass
