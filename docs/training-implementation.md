# XLM-R training implementation record

Approved scope: develop branch; one shared XLM-R Base encoder for en, zh-Hans,
th; 25-label BCE classification; frozen 9000/1500/3000 splits. No paid training
or generation API calls locally. GPU baseline: RTX 4090 24GB, Python 3.12,
PyTorch 2.5.1/cu124 (user-selected image), Transformers 4.49.0.

## Tasks and completion contracts

1. Independent training environment, validated data/configuration, tested metrics.
2. Encoder/head, training loop, epoch resume, portable model bundle; CPU tests.
3. Calibration, sealed-test evaluation, reports, benchmark and experiment orchestration.
4. Migration archive and Chinese GPU runbook; real-asset local smoke and independent review.

## Fixed decisions

- Preserve root source and uv.lock, which are bound into the frozen data fingerprint.
- Paths in TOML are relative to that TOML. Only text enters the model.
- First-token hidden state -> dropout(0.1) -> linear(25); no pooling/MLM head.
- Length 128, dynamic padding, no truncation; batch 16 x accumulation 2.
- AdamW, learning rates 2e-5/1e-5/3e-5, weight decay .01, clip 1,
  warmup .1, up to 5 epochs, patience 2, BF16 then FP16 on CUDA.
- Select epochs by validation exact-set at .5 (earlier tie), calibrate global
  threshold .10-.90 step .01 (closest to .5 then smaller tie).
- Select LR by calibrated validation exact-set, then Macro F1, then listed LR
  order. Re-run winner at seeds 43/44; include seed 42, freeze all before test.
- Report undefined denominators as null; Wilson intervals describe this synthetic
  dataset only, not independent production-risk evidence.
- Retain best portable model and latest complete-epoch optimizer checkpoint.
- Local 7.5 GiB RAM is suitable for tiny-model training tests and real-model
  inference. Full-encoder optimizer/backward and throughput gate run on rented GPU.

## Progress

- Baseline: develop c1f102e, clean; existing 168 tests and Ruff passed previously.
- Planning completed in conversation; this file records the accepted specification.
- Implemented: independent TOML/uv environment, frozen data validation, XLM-R head,
  weighted gradient accumulation, complete-epoch resume, global threshold sweep,
  explicit freeze gate, five-run experiment, per-language/routing reports,
  hardware diagnostics, offline prediction and allowlisted migration archive.
- Independent read-only review identified FP16 overflow handling, precision parity
  in prediction, and interrupted calibration completion. All three were reproduced
  with failing regression tests, then fixed; the 31-test suite passed.
- Real data preflight passed: 9000/1500/3000 rows, 25 labels, consistent hashes and
  disjoint sample/group IDs. All texts fit 128 tokens (maximum 85).
- Real 1.1GB XLM-R weights loaded locally and produced finite [2,25] outputs.
  Local environment: torch 2.5.1+cpu, transformers 4.49.0, Python 3.12.13.
- Ruling: local full-weight backward/optimizer testing is deferred to GPU because
  WSL has only 7.5 GiB RAM; tiny-model backward, reload and exact resume are tested.
  Cost: actual CUDA memory/throughput and BF16 kernels remain GPU acceptance gates.
- Ruling: PyTorch CPU download required parallel HTTP ranges from the official
  host; the assembled wheel matched the exact SHA256 in training/uv.lock before
  installation. Dependency versions and portable registry lock were unchanged.
- No paid API calls or full real-data training runs were performed locally.
