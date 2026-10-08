"""API workers with a single owner for quotas, validation, deduplication and SQLite."""

import queue
import threading
import time
from collections import Counter, deque
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace

from .models import (
    CancelledProviderError,
    Candidate,
    DatasetError,
    FatalProviderError,
    Judgement,
    TransientProviderError,
)
from .quality import DuplicateIndex, validate_candidate, validate_judgement
from .storage import now


def retry_delay(failures):
    return min(15 * 2 ** min(failures - 1, 2), 60)


@dataclass
class Batch:
    sequence: int
    role: str
    items: list
    ready_at: float = 0.0
    failures: int = 0
    call_id: int | None = None

    def ids(self):
        return [item.slot_id if self.role == "generator" else item.sample_id for item in self.items]


class Coordinator:
    def __init__(self, store, settings, catalog, inspector, provider, slots, records,
                 runtime, until_complete, progress, stage):
        self.store, self.settings, self.catalog = store, settings, catalog
        self.inspector, self.provider, self.runtime = inspector, provider, runtime
        self.until_complete, self.progress, self.stage = until_complete, progress, stage
        self.slots = slots
        self.lookup = {slot.slot_id: slot for slot in slots}
        self.done = {row["sample_id"] for row in records}
        self.counts = {slot.slot_id: store.generation_count(slot.slot_id) for slot in slots}
        self.labels = [row["name"] for row in catalog["labels"]]
        self.index = DuplicateIndex(settings.near_duplicate_threshold)
        for record in records:
            self.index.add(record)
        self.reserved = set()
        self.futures = {}
        self.ready = {"generator": deque(), "judge": deque()}
        self.sequence = 0
        self.stop = threading.Event()
        self.events = queue.SimpleQueue()
        self.no_growth = 0
        self.generation_pause_until = 0.0
        self.capacity = runtime.generator_workers + runtime.judge_workers
        self.pools = {}

    def log(self, message):
        self.progress(f"{self.stage}: {message}", flush=True)

    def reject(self, sample_id, reasons):
        slot = self.lookup[sample_id]
        self.store.reject(sample_id, sorted(set(reasons)),
                          {key: getattr(slot, key) for key in ("locale", "split", "kind", "scenario")})
        self.reserved.discard(sample_id)

    def pending(self):
        slots = [slot for slot in self.slots
                 if slot.slot_id not in self.done and slot.slot_id not in self.reserved
                 and (self.until_complete or self.counts[slot.slot_id] < self.settings.semantic_retries + 1)]
        if self.until_complete:
            slots.sort(key=lambda slot: self.counts[slot.slot_id])
        return slots

    def allocate(self, pending):
        slots = pending[:self.settings.batch_size]
        self.reserved.update(slot.slot_id for slot in slots)
        self.sequence += 1
        items = [replace(slot, seed_brief=slot.seed_brief + (
            f" Fresh candidate attempt {self.counts[slot.slot_id] + 1}; "
            "use a new natural phrasing within this family and scenario, without changing the target set."
        )) for slot in slots]
        return Batch(self.sequence, "generator", items)

    def dispatch(self, batch):
        # Every checkpoint operation is performed here on the coordinator thread.
        batch.call_id = self.store.begin_call(batch.role, batch.ids())
        if batch.role == "generator":
            for sample_id in batch.ids():
                self.counts[sample_id] = self.store.generation_count(sample_id)
        method = self.provider.generate if batch.role == "generator" else self.provider.judge
        try:
            future = self.pools[batch.role].submit(method, batch.items)
        except BaseException:
            self.store.finish_call(batch.call_id, {}, 0, status="cancelled", error_type="DispatchCancelled")
            raise
        self.futures[future] = batch

    def generated(self, batch, result):
        by_id = {item.sample_id: item for item in result.items if isinstance(item, Candidate)}
        candidates = []
        for slot in batch.items:
            candidate = by_id.get(slot.slot_id)
            if candidate is None or slot.slot_id in result.invalid_ids:
                self.reject(slot.slot_id, ["invalid_generation_response",
                                          *result.invalid_reasons.get(slot.slot_id, [])])
                continue
            reasons = validate_candidate(candidate, slot, self.labels, self.inspector)
            if not reasons:
                duplicate = self.index.check(candidate.text, candidate.locale, candidate.labels,
                                             slot.group_id, slot.split)
                if duplicate:
                    reasons.append(duplicate)
            if reasons:
                self.reject(slot.slot_id, reasons)
            else:
                candidates.append(candidate)
        if candidates:
            self.ready["judge"].append(Batch(batch.sequence, "judge", candidates))
        else:
            self.note_growth(False)

    def reviewed(self, batch, result):
        verdicts = {item.sample_id: item for item in result.items if isinstance(item, Judgement)}
        added = 0
        for candidate in batch.items:
            slot = self.lookup[candidate.sample_id]
            verdict = verdicts.get(candidate.sample_id)
            if verdict is None or candidate.sample_id in result.invalid_ids:
                self.reject(slot.slot_id, ["invalid_judge_response",
                                          *result.invalid_reasons.get(slot.slot_id, [])])
                continue
            reasons = validate_judgement(verdict, candidate, self.labels)
            canonical_labels = sorted(candidate.labels, key=self.labels.index)
            if not reasons:
                # Mandatory second global check: another batch may have committed
                # overlapping text while this candidate was being independently judged.
                duplicate = self.index.check(candidate.text, candidate.locale, canonical_labels,
                                             slot.group_id, slot.split)
                if duplicate:
                    reasons.append(duplicate)
            if reasons:
                self.reject(slot.slot_id, reasons)
                continue
            record = {
                "sample_id": slot.slot_id, "text": candidate.text, "labels": canonical_labels,
                "locale": slot.locale, "split": slot.split, "group_id": slot.group_id,
                "family_id": slot.family_id, "kind": slot.kind, "scenario": slot.scenario,
                "source": "llm_synthetic", "label_version": self.catalog["version"],
                "review_status": "machine_accepted", "n_tokens": self.inspector.count(candidate.text),
                "generator_model": self.provider.metadata()["generator_model"],
                "judge_model": self.provider.metadata()["judge_model"],
                "generation_attempt": self.counts[slot.slot_id],
                "provider_profile_id": self.store.provider_profile_id,
            }
            self.store.accept(record)
            self.index.add(record)
            self.done.add(slot.slot_id)
            self.reserved.discard(slot.slot_id)
            added += 1
        self.note_growth(added > 0)
        self.log(f"accepted {sum(slot.slot_id in self.done for slot in self.slots)}/{len(self.slots)}; "
                 f"batch accepted={added}/{len(batch.items)}; in_flight_slots={len(self.reserved)}")

    def note_growth(self, added):
        self.no_growth = 0 if added else self.no_growth + 1
        if self.until_complete and self.no_growth >= 6:
            self.generation_pause_until = time.monotonic() + 60
            self.no_growth = 0
            self.log("no recent growth; pause new generation for 60s; quality checks unchanged")

    def finish(self, future, *, draining=False):
        batch = self.futures.pop(future)
        try:
            result = future.result()
        except BaseException as error:
            cancelled = isinstance(error, (CancelledError, CancelledProviderError, KeyboardInterrupt))
            self.store.finish_call(batch.call_id, getattr(error, "usage", {}),
                                   getattr(error, "request_count", 0 if cancelled else 1),
                                   status="cancelled" if cancelled else "failed", error_type=type(error).__name__)
            if draining:
                return
            if isinstance(error, TransientProviderError) and self.until_complete:
                batch.failures += 1
                seconds = retry_delay(batch.failures)
                batch.ready_at = time.monotonic() + seconds
                self.ready[batch.role].append(batch)
                self.log(f"{batch.role} temporarily unavailable; retry in {seconds}s")
                return
            if isinstance(error, (DatasetError, KeyboardInterrupt)):
                raise
            raise FatalProviderError("API worker failed; no unreviewed samples were accepted") from None
        self.store.finish_call(batch.call_id, result.usage, result.request_count)
        if result.invalid_ids:
            reasons = Counter(code for codes in result.invalid_reasons.values() for code in codes)
            self.log(f"{batch.role} invalid={len(result.invalid_ids)}; reasons={dict(reasons)}")
        if batch.role == "judge":
            self.reviewed(batch, result)
        elif not draining:
            self.generated(batch, result)
        # On shutdown completed generation is deliberately not submitted for review.
        # Only reviewed, validated results may be committed while draining workers.

    def schedule(self):
        for role in ("judge", "generator"):
            active = sum(batch.role == role for batch in self.futures.values())
            maximum = getattr(self.runtime, role + "_workers")
            queued = self.ready[role]
            for _ in range(len(queued)):
                batch = queued.popleft()
                if active < maximum and time.monotonic() >= batch.ready_at:
                    self.dispatch(batch)
                    active += 1
                else:
                    queued.append(batch)
            if role == "generator" and time.monotonic() >= self.generation_pause_until:
                while active < maximum:
                    total = len(self.futures) + sum(len(q) for q in self.ready.values())
                    if total >= self.capacity:
                        break
                    pending = self.pending()
                    if not pending:
                        break
                    self.dispatch(self.allocate(pending))
                    active += 1

    def run(self):
        configure = getattr(self.provider, "configure_parallel", None)
        if configure:
            configure(self.runtime, self.stop, self.events.put)
        executions = self.store.get_meta("execution_runs") or []
        executions.append({"at": now(), "stage": self.stage, "until_complete": self.until_complete,
                           "runtime": asdict(self.runtime)})
        self.store.set_meta("execution_runs", executions)
        self.log(f"mode={'until_complete' if self.until_complete else 'bounded'}; "
                 f"parallel generator={self.runtime.generator_workers}, judge={self.runtime.judge_workers}; "
                 f"batch_size={self.settings.batch_size}; max_pending_batches={self.capacity}")
        try:
            self.pools = {role: ThreadPoolExecutor(max_workers=getattr(self.runtime, role + "_workers"),
                                                 thread_name_prefix="watch-" + role)
                          for role in ("generator", "judge")}
            while True:
                for future in sorted([f for f in self.futures if f.done()],
                                     key=lambda f: self.futures[f].sequence):
                    self.finish(future)
                while not self.events.empty():
                    self.log(f"rate_control={self.events.get()}")
                self.schedule()
                if not self.futures and not any(self.ready.values()) and not self.pending():
                    break
                if self.futures:
                    wait(self.futures, timeout=0.1)
                else:
                    time.sleep(0.1)
        finally:
            self.stop.set()
            for future in self.futures:
                future.cancel()
            if self.futures:
                self.log("stopping dispatch; waiting for in-flight calls and saving completed reviews")
            try:
                while self.futures:
                    completed, _ = wait(self.futures, timeout=0.2)
                    for future in sorted(completed, key=lambda f: self.futures[f].sequence):
                        self.finish(future, draining=True)
            finally:
                for pool in self.pools.values():
                    pool.shutdown(wait=True, cancel_futures=True)
                self.ready.clear()
                self.reserved.clear()
