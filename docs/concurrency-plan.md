# Concurrent generation implementation ledger

Scope: preserve model/prompt/batch size, fixed slots, independent judgement and all quality checks; default two generator and two judge workers. Only the coordinator reserves IDs, validates, deduplicates and writes SQLite. API workers never access checkpoints. Bounded queues and model-scoped rate/capacity gates prevent uncontrolled requests. Back up and migrate the known serial v0.2 checkpoint, preserving samples and audit history.

Ruling: work in the current checkout without spawning agents or changing git state, as required by this side conversation; existing implementation and checkpoints are uncommitted. Perform a separate self-review after tests.
Ruling: concurrency and rate settings are operational configuration in dashscope.env (with optional CLI worker overrides), excluded from dataset identity. Models, prompts, quotas, assets and quality rules remain fingerprinted. Runtime settings are recorded separately for audit.

Tasks: (1) runtime/model limiter and HTTP cancellation; (2) bounded concurrent scheduler and central admission; (3) CLI/migration/docs; (4) regression tests, self-review, offline migration verification. No bulk or paid generation will be launched.
Baseline: 510 accepted; code hash 31bc2c4f7e03c20bb796faaf1848e74e30993557e656c7d4a371ffaef9b2ac7a; prompt v0.2; source matches checkpoint, directory lock available.


Progress: runtime limits, bounded coordinator, HTTP cancellation, CLI overrides and known-serial migration implemented. Test-first runtime import/parallel entry failed before implementation. Integration tests exercise real provider parsing through local HTTP mocks with two generation and two judge calls overlapping. Quality tests cover concurrent duplicate races, overlength/wrong-language rejection, label disagreement, ambiguity, unnaturalness, retry reuse, coordinator-only writes and interrupt/resume. Model gate tests cover RPM/TPM, shared model limits, 429 shrink/recovery and cancel while waiting.

Self-review: performed in this side conversation without agents as requested. No models, prompt strings/version, batch size, split planning or quality functions changed. Fixed a discovered accounting edge: a request denied locally by the token budget must count as zero API requests. New regression passes. Shutdown drains only already-reviewed results; candidates awaiting review remain memory-only. Permanent errors end the build; transient failures keep reservations and retry. Runtime defaults are operational ceilings, not claimed provider entitlements.

Completion: 161 offline tests passed; Ruff clean; offline check reports generator_workers=2 and judge_workers=2. Known serial checkpoint migrated with SQLite backup before-concurrency-1790737085921995627.sqlite3. All 510 samples, rejection events, API calls and attempt counts match the backup exactly. Provider metadata is unchanged; current fingerprint matches; all saved samples pass local validation. Pilot verification passed 300/300 with zero errors. No paid requests or bulk generation were launched. User may run the original command chain to resume.

Follow-up: user moved runtime settings to concurrent.env. Runtime loader and CLI help now use concurrent.env exclusively for file-backed operational settings. Existing concurrent.env values are preserved; stale runtime keys are removed from dashscope.env and its example. Added concurrent.env.example, file precedence/no-model-file-fallback tests, and explicit compatibility for the known previous concurrency checkpoint.
