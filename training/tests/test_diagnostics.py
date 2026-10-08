import pytest


def test_check_real_contract_and_full_smoke(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.diagnostics import check, smoke
    c = load_config(fixture_project)
    checked = check(c)
    assert checked["counts"] == {"train": 9, "validation": 9, "test": 9}
    result = smoke(c)
    assert result["ok"] and not result["forward_only"]


def test_training_and_batch_one_inference_benchmark(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.diagnostics import benchmark
    from watch_intents_training.engine import train
    from watch_intents_training.evaluation import calibrate_run
    c = load_config(fixture_project)
    measured = benchmark(c, steps=2, warmup=1)
    assert measured["mode"] == "training_optimizer_steps"
    assert measured["samples_per_second"] > 0
    train(c, "bench")
    calibrate_run(c, "bench")
    measured = benchmark(c, steps=2, warmup=1, bundle=c.output_dir / "bench/model")
    assert measured["mode"] == "inference_batch1_local_call"
    assert measured["samples"] == 2
    assert measured["p95_ms"] > 0


def test_overfit_is_train_only_and_not_a_candidate(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.diagnostics import overfit
    c = load_config(fixture_project)
    result = overfit(c, max_steps=1, count=9)
    assert all(name.startswith("train-") for name in result["sample_ids"])
    assert result["scope"] == "train_subset_diagnostic_not_a_candidate"
    assert not (c.output_dir / "latest.pt").exists()


def test_cli_reports_invalid_input_as_failure(fixture_project, capsys):
    from watch_intents_training.cli import main
    assert main(["--config", str(fixture_project), "train", "--run", "../escape"]) == 1
    assert "run_id" in capsys.readouterr().err


def test_cross_split_groups_and_long_text_are_rejected():
    from watch_intents_training.data import ensure_disjoint
    with pytest.raises(ValueError, match="leakage"):
        ensure_disjoint([{"sample_id": "train-a", "group_id": "common"}],
                        [{"sample_id": "validation-b", "group_id": "common"}])
