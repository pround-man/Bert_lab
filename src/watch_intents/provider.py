"""Synchronous, bounded API calls with no raw-response logging or persistence."""

import json
import os
import random
import time
from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from pydantic import ValidationError

from .models import (
    BatchResult,
    CancelledProviderError,
    Candidate,
    FatalProviderError,
    Judgement,
    TransientProviderError,
)
from .prompts import PROMPT_VERSION, generation_messages, judgement_messages
from .runtime import ModelGate

if TYPE_CHECKING:
    from .planning import Slot

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 3


def _base_url(value: str) -> str:
    """Prevent credentials in URLs from entering sanitized metadata or errors."""
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"https", "http"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise FatalProviderError("API base URL must be HTTP(S), without credentials or query")
    return value.rstrip("/")


def _usage(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return {}
    return {
        key: value
        for key, value in payload["usage"].items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and type(value) is int
        and value >= 0
    }


def _delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                seconds = (parsed - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = -1
        if 0 <= seconds <= float("inf"):
            return min(seconds, 60.0)
    return min(2**attempt + random.uniform(0, 0.5), 60.0)


def _error(error_type, message: str, request_count: int, usage: dict[str, int]):
    error = error_type(message)
    error.request_count = request_count
    error.usage = usage
    return error


class LLMProvider:
    def __init__(
        self,
        root: Path,
        catalog: dict,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120,
        max_response_tokens: int = 8192,
        max_retries: int = MAX_RETRIES,
    ):
        # No mutation of os.environ and no dotenv interpolation of other secrets.
        values = dotenv_values(root / "dashscope.env", interpolate=False)

        def setting(name: str, default: str = "") -> str:
            # Explicit file values, including blanks, override stale shell exports.
            value = values[name] if name in values else os.environ.get(name, default)
            return (value or "").strip()

        self.generator_key = setting("DASHSCOPE_API_KEY")
        self.judge_key = setting("SUUNTO_EVAL_JUDGE_API_KEY")
        for name, value in (
            ("DASHSCOPE_API_KEY", self.generator_key),
            ("SUUNTO_EVAL_JUDGE_API_KEY", self.judge_key),
        ):
            if not value:
                raise FatalProviderError(f"Missing credential: {name}")
        self.generator_model = setting("SUUNTO_CHAT_MODEL_CN_NAME")
        self.judge_model = setting("SUUNTO_EVAL_JUDGE_NAME")
        for name, value in (
            ("SUUNTO_CHAT_MODEL_CN_NAME", self.generator_model),
            ("SUUNTO_EVAL_JUDGE_NAME", self.judge_model),
        ):
            if not value:
                raise FatalProviderError(f"Missing model: {name}")
        if setting("SUUNTO_EVAL_JUDGE_PROVIDER", "dashscope") != "dashscope":
            raise FatalProviderError("SUUNTO_EVAL_JUDGE_PROVIDER must be dashscope")
        self.generator_base_url = _base_url(setting("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
        self.judge_base_url = _base_url(
            setting("SUUNTO_EVAL_JUDGE_BASE_URL", DEFAULT_BASE_URL)
        )
        if timeout <= 0 or max_response_tokens <= 0:
            raise FatalProviderError("API timeout and max_response_tokens must be positive")
        if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
            raise FatalProviderError("API max_retries must be an integer from 0 to 3")
        self._gates = {}
        self._stop = None
        self.catalog = catalog
        self.max_response_tokens = max_response_tokens
        self.max_retries = max_retries
        self.client = httpx.Client(
            transport=transport, timeout=timeout, follow_redirects=False
        )

    def configure_parallel(self, runtime, stop, notify):
        self._stop = stop
        groups = {}
        for role in ("generator", "judge"):
            key = (getattr(self, role + "_base_url"), getattr(self, role + "_model"))
            groups.setdefault(key, []).append(role)
        self._gates = {}
        for key, roles in groups.items():
            self._gates[key] = ModelGate(
                workers=max(getattr(runtime, role + "_workers") for role in roles),
                rpm=min(getattr(runtime, role + "_rpm") for role in roles),
                tpm=min(getattr(runtime, role + "_tpm") for role in roles),
                notify=lambda event, roles=roles: notify({"roles": "+".join(roles), **event}),
            )

    def _pause(self, seconds, request_count, usage):
        if self._stop is None:
            time.sleep(seconds)
        elif self._stop.wait(seconds):
            raise _error(CancelledProviderError, "API retry cancelled", request_count, usage)

    def metadata(self) -> dict:
        return {
            "generator_model": self.generator_model,
            "generator_base_url": self.generator_base_url,
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "prompt_version": PROMPT_VERSION,
            "max_response_tokens": self.max_response_tokens,
        }

    def close(self) -> None:
        self.client.close()

    def generate(self, slots: list["Slot"]) -> BatchResult:
        return self._request(
            messages=generation_messages(slots, self.catalog),
            expected_ids=[slot.slot_id for slot in slots],
            model=self.generator_model,
            key=self.generator_key,
            base_url=self.generator_base_url,
            envelope="samples",
            item_type=Candidate,
        )

    def judge(self, candidates: list[Candidate]) -> BatchResult:
        return self._request(
            messages=judgement_messages(candidates, self.catalog),
            expected_ids=[item.sample_id for item in candidates],
            model=self.judge_model,
            key=self.judge_key,
            base_url=self.judge_base_url,
            envelope="judgements",
            item_type=Judgement,
        )

    def _request(
        self,
        *,
        messages: list[dict[str, str]],
        expected_ids: list[str],
        model: str,
        key: str,
        base_url: str,
        envelope: str,
        item_type: type[Candidate] | type[Judgement],
    ) -> BatchResult:
        if not expected_ids:
            return BatchResult(items=[], request_count=0)
        if len(set(expected_ids)) != len(expected_ids):
            raise FatalProviderError("Request contains duplicate sample identifiers")
        usage: dict[str, int] = {}
        gate = self._gates.get((base_url, model))
        reservation = self.max_response_tokens + 512 + sum(
            len(message["content"].encode("utf-8")) for message in messages
        )
        for attempt in range(self.max_retries + 1):
            ticket = None
            if gate is not None:
                try:
                    ticket = gate.acquire(reservation, self._stop)
                except (CancelledProviderError, FatalProviderError) as error:
                    raise _error(type(error), str(error), attempt, usage) from None
            response = None
            try:
                response = self.client.post(
                    base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + key},
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": self.max_response_tokens,
                        "response_format": {"type": "json_object"},
                    },
                )
            except httpx.TransportError:
                if gate is not None:
                    gate.finish(ticket, success=False)
                # Never expose httpx exception details, URLs, headers or text.
                if attempt == self.max_retries:
                    raise _error(
                        TransientProviderError,
                        f"API transport failed after {attempt + 1} attempts",
                        attempt + 1, usage,
                    ) from None
                self._pause(_delay(attempt, None), attempt + 1, usage)
                continue
            except BaseException:
                if gate is not None:
                    gate.finish(ticket, success=False)
                raise

            try:
                payload = response.json()
            except (ValueError, UnicodeError):
                payload = None
            for name, value in _usage(payload).items():
                usage[name] = usage.get(name, 0) + value
            status = response.status_code
            if gate is not None:
                gate.finish(ticket, _usage(payload).get("total_tokens"), throttled=status == 429,
                            retry_after=_delay(attempt, response.headers.get("Retry-After"))
                            if status == 429 else 0, success=200 <= status < 300)
            if status == 429 or 500 <= status < 600:
                if attempt == self.max_retries:
                    raise _error(
                        TransientProviderError,
                        f"API temporarily unavailable (HTTP {status}) after {attempt + 1} attempts",
                        attempt + 1, usage,
                    ) from None
                self._pause(_delay(attempt, response.headers.get("Retry-After")), attempt + 1, usage)
                continue
            if not 200 <= status < 300:
                raise _error(
                    FatalProviderError, f"API request rejected (HTTP {status}); check configuration",
                    attempt + 1, usage,
                ) from None
            return self._parse(payload, expected_ids, envelope, item_type, usage, attempt + 1)
        raise AssertionError("Unreachable retry state")

    @staticmethod
    def _parse(payload, expected_ids, envelope, item_type, usage, request_count) -> BatchResult:
        def invalid_all(code):
            return BatchResult([], list(expected_ids), usage, request_count,
                               {sample_id: [code] for sample_id in expected_ids})

        try:
            choice = payload["choices"][0]
            finish = choice.get("finish_reason")
            if finish == "length":
                return invalid_all("output_truncated")
            if finish not in (None, "stop"):
                return invalid_all("output_not_complete")
            content = choice["message"]["content"]
        except (TypeError, KeyError, IndexError, AttributeError):
            return invalid_all("invalid_api_envelope")
        if not isinstance(content, str) or not content.strip():
            return invalid_all("empty_or_nonstring_content")
        try:
            parsed = json.loads(content)
        except ValueError:
            return invalid_all("invalid_json")
        if not isinstance(parsed, dict) or set(parsed) != {envelope}:
            return invalid_all("invalid_response_envelope")
        rows = parsed[envelope]
        if not isinstance(rows, list):
            return invalid_all("rows_not_array")
        expected = set(expected_ids)
        counts = Counter(
            row["sample_id"]
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("sample_id"), str)
        )
        if set(counts) - expected:
            return invalid_all("unknown_sample_id")
        valid = {}
        reasons = {}
        for sample_id in expected_ids:
            if counts[sample_id] != 1:
                reasons[sample_id] = ["duplicate_sample_id" if counts[sample_id] else "missing_sample"]
        for row in rows:
            if not isinstance(row, dict):
                continue
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or counts[sample_id] != 1:
                continue
            try:
                valid[sample_id] = item_type.model_validate(row, strict=True)
            except ValidationError as error:
                # Persist only locally bounded codes, never values, messages, or unknown field names.
                codes = set()
                for detail in error.errors(include_input=False, include_context=False, include_url=False):
                    field = detail["loc"][0] if detail["loc"] else None
                    if field not in item_type.model_fields:
                        field = "unknown_field"
                    kind = detail["type"]
                    if kind not in {"missing", "extra_forbidden", "string_type", "list_type",
                                    "bool_type", "literal_error", "string_too_short"}:
                        kind = "invalid"
                    codes.add(f"schema_{field}_{kind}")
                reasons[sample_id] = sorted(codes)
        return BatchResult(
            items=[valid[sample_id] for sample_id in expected_ids if sample_id in valid],
            invalid_ids=[sample_id for sample_id in expected_ids if sample_id not in valid],
            usage=usage,
            request_count=request_count,
            invalid_reasons=reasons,
        )
