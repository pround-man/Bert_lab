import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from watch_intents.models import CancelledProviderError, FatalProviderError
from watch_intents.runtime import ModelGate, load_runtime


def test_runtime_file_values_win_and_invalid_blanks_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_INTENTS_GENERATOR_WORKERS", "8")
    (tmp_path / "concurrent.env").write_text("WATCH_INTENTS_GENERATOR_WORKERS=3\n")
    assert load_runtime(tmp_path).generator_workers == 3
    assert load_runtime(tmp_path).judge_workers == 2
    (tmp_path / "concurrent.env").write_text("WATCH_INTENTS_GENERATOR_WORKERS=\n")
    with pytest.raises(FatalProviderError):
        load_runtime(tmp_path)


def test_gate_rolling_requests_tokens_and_cancellation():
    gate = ModelGate(workers=2, rpm=1, tpm=100, window=0.12)
    stop = threading.Event()
    first = gate.acquire(60, stop)
    gate.finish(first, 20)
    started = time.monotonic()
    second = gate.acquire(60, stop)
    assert time.monotonic() - started >= 0.10
    gate.finish(second, 20)
    stop.set()
    with pytest.raises(CancelledProviderError):
        gate.acquire(10, stop)
    with pytest.raises(FatalProviderError):
        gate.acquire(101, threading.Event())


def test_gate_429_reduces_capacity_and_waits():
    gate = ModelGate(workers=3, rpm=100, tpm=10000, window=0.02, recovery_successes=2)
    stop = threading.Event()
    first, second = gate.acquire(10, stop), gate.acquire(10, stop)
    gate.finish(first, 10, throttled=True, retry_after=0.05)
    assert gate.snapshot()["active_limit"] == 1
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(gate.acquire, 10, stop)
        time.sleep(0.025)
        assert not future.done()
        gate.finish(second, 10)
        ticket = future.result(timeout=1)
        gate.finish(ticket, 10)
    assert gate.snapshot()["active_limit"] == 2



def test_tpm_budget_blocks_even_when_rpm_and_workers_allow_more():
    gate = ModelGate(workers=3, rpm=100, tpm=100, window=0.15)
    stop = threading.Event()
    first = gate.acquire(60, stop)
    gate.finish(first, 60)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(gate.acquire, 60, stop)
        time.sleep(0.03)
        assert not future.done()
        ticket = future.result(timeout=1)
        gate.finish(ticket, 60)



def test_cancellation_wakes_waiter_without_starting_request():
    gate = ModelGate(workers=1, rpm=1, tpm=100)
    stop = threading.Event()
    ticket = gate.acquire(80, stop)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(gate.acquire, 10, stop)
        time.sleep(0.02)
        stop.set()
        with pytest.raises(CancelledProviderError):
            future.result(timeout=1)
    gate.finish(ticket, 80)
    assert gate.snapshot()["active"] == 0



def test_runtime_reads_only_concurrent_file_not_model_credentials_file(tmp_path, monkeypatch):
    for name in ("GENERATOR_WORKERS", "JUDGE_WORKERS", "GENERATOR_RPM", "JUDGE_RPM",
                 "GENERATOR_TPM", "JUDGE_TPM"):
        monkeypatch.delenv("WATCH_INTENTS_" + name, raising=False)
    (tmp_path / "dashscope.env").write_text("WATCH_INTENTS_GENERATOR_WORKERS=8\nWATCH_INTENTS_JUDGE_RPM=99\n")
    assert load_runtime(tmp_path).generator_workers == 2
    (tmp_path / "concurrent.env").write_text("WATCH_INTENTS_GENERATOR_WORKERS=3\nWATCH_INTENTS_JUDGE_RPM=5\n")
    assert load_runtime(tmp_path).generator_workers == 3
    assert load_runtime(tmp_path).judge_rpm == 5
    assert load_runtime(tmp_path, generator_workers=4).generator_workers == 4
