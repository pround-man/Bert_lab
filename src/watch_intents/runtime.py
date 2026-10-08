"""Operational concurrency and model-scoped sliding-window admission control."""

import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import dotenv_values

from .models import CancelledProviderError, FatalProviderError


@dataclass(frozen=True)
class RuntimeSettings:
    generator_workers: int = 2
    judge_workers: int = 2
    generator_rpm: int = 6
    judge_rpm: int = 6
    generator_tpm: int = 120000
    judge_tpm: int = 120000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1 or (name.endswith("workers") and value > 16):
                raise FatalProviderError(f"Invalid runtime setting: {name}")


def load_runtime(root: Path, **overrides) -> RuntimeSettings:
    values = dotenv_values(root / "concurrent.env", interpolate=False)
    defaults = asdict(RuntimeSettings())
    for name, default in defaults.items():
        key = "WATCH_INTENTS_" + name.upper()
        value = overrides.get(name)
        if value is None:
            value = values[key] if key in values else os.environ.get(key, default)
        try:
            defaults[name] = int(value)
        except (ValueError, TypeError):
            raise FatalProviderError(f"Invalid runtime setting: {key}") from None
    return RuntimeSettings(**defaults)


class ModelGate:
    """Reserve request/token budget before I/O; refund only known token usage.

    A shared gate is keyed by endpoint/model, not API key. At most the configured
    number of requests run concurrently. 429 lowers this cap; sustained success
    recovers it gradually, never beyond the configured maximum.
    """

    def __init__(self, *, workers, rpm, tpm, window=60.0, recovery_successes=20, notify=None):
        self.maximum = workers
        self.limit = workers
        self.rpm, self.tpm, self.window = rpm, tpm, window
        self.recovery_successes = recovery_successes
        self.notify = notify or (lambda _: None)
        self.condition = threading.Condition()
        self.entries = deque()
        self.active = 0
        self.cooldown = 0.0
        self.last_throttle = float("-inf")
        self.successes = 0

    def acquire(self, tokens, stop):
        if tokens > self.tpm:
            raise FatalProviderError("Per-request token reservation exceeds configured TPM; increase the runtime TPM ceiling")
        with self.condition:
            while True:
                if stop.is_set():
                    raise CancelledProviderError("API request cancelled before dispatch")
                now = time.monotonic()
                while self.entries and now - self.entries[0][0] >= self.window:
                    self.entries.popleft()
                if (self.active < self.limit and now >= self.cooldown
                        and len(self.entries) < self.rpm
                        and sum(entry[1] for entry in self.entries) + tokens <= self.tpm):
                    ticket = [now, tokens]
                    self.entries.append(ticket)
                    self.active += 1
                    return ticket
                self.condition.wait(timeout=0.1)

    def finish(self, ticket, actual_tokens=None, *, throttled=False, retry_after=0.0, success=True):
        with self.condition:
            self.active -= 1
            if type(actual_tokens) is int and actual_tokens >= 0:
                ticket[1] = actual_tokens
            now = time.monotonic()
            if throttled:
                self.limit = max(1, self.limit // 2)
                self.last_throttle = now
                self.cooldown = max(self.cooldown, now + retry_after)
                self.successes = 0
                self.notify({"event": "rate_limited", "active_limit": self.limit,
                             "retry_seconds": retry_after})
            elif success:
                self.successes += 1
                if (self.successes >= self.recovery_successes and self.limit < self.maximum
                        and now - self.last_throttle >= self.window):
                    self.limit += 1
                    self.successes = 0
                    self.notify({"event": "capacity_recovered", "active_limit": self.limit})
            else:
                self.successes = 0
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "active_limit": self.limit,
                    "configured_workers": self.maximum, "rpm": self.rpm, "tpm": self.tpm}
