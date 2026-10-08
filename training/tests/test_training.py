from dataclasses import replace

import numpy as np
import pytest


def test_bundle_reload_preserves_predictions(fixture_project, tmp_path):
    import torch
    from transformers import AutoTokenizer

    from watch_intents_training.model import IntentModel, load_bundle, save_bundle
    model = IntentModel.from_base(fixture_project.parent / "base", 25).eval()
    tokenizer = AutoTokenizer.from_pretrained(fixture_project.parent / "base")
    inputs = tokenizer(["open sport"], return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        expected = model(**inputs)
    save_bundle(tmp_path / "bundle", model, tokenizer,
                {"labels": [f"L{i}" for i in range(25)], "max_length": 128,
                 "thresholds": [0.5] * 25})
    loaded, _, metadata = load_bundle(tmp_path / "bundle")
    with torch.no_grad():
        actual = loaded(**inputs)
    assert torch.equal(expected, actual)
    assert len(metadata["thresholds"]) == 25
    from watch_intents_training.io import read_json
    assert read_json(tmp_path / "bundle/label2id.json") == {f"L{i}": i for i in range(25)}
    assert read_json(tmp_path / "bundle/id2label.json") == {str(i): f"L{i}" for i in range(25)}
    assert (tmp_path / "bundle/environment.uv.lock").is_file()


def test_epoch_resume_matches_uninterrupted(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.engine import train
    from watch_intents_training.io import read_json
    c = load_config(fixture_project)
    full = train(c, "full")
    interrupted = train(c, "resumed", stop_after_epoch=1)
    assert interrupted["completed_epochs"] == 1
    resumed = train(c, "resumed", resume=True)
    assert full["history"] == resumed["history"]
    assert read_json(c.output_dir / "full/run.json")["status"] == "complete"
    with pytest.raises(ValueError, match="contract"):
        train(replace(c, learning_rate=1e-4), "resumed", resume=True)


def test_accumulation_weights_partial_groups():
    import torch

    from watch_intents_training.engine import optimize_group
    # Dropout-free model: micro batches of 2 and 1 equal one batch of 3.
    a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    b.load_state_dict(a.state_dict())
    x, y = torch.tensor([[1., 0.], [0., 1.], [1., 1.]]), torch.tensor([[1.], [0.], [1.]])
    oa, ob = torch.optim.SGD(a.parameters(), lr=0.1), torch.optim.SGD(b.parameters(), lr=0.1)
    loss = torch.nn.BCEWithLogitsLoss()
    oa.zero_grad()
    loss(a(x), y).backward()
    oa.step()
    optimize_group(b, [({"input": x[:2]}, y[:2]), ({"input": x[2:]}, y[2:])],
                   ob, None, "cpu", "fp32", 100.)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-7)


def test_model_rejects_changed_bundle(fixture_project, tmp_path):
    from transformers import AutoTokenizer

    from watch_intents_training.io import read_json, write_json
    from watch_intents_training.model import IntentModel, load_bundle, save_bundle
    path = tmp_path / "bundle"
    save_bundle(path, IntentModel.from_base(fixture_project.parent / "base", 25),
                AutoTokenizer.from_pretrained(fixture_project.parent / "base"),
                {"labels": [f"L{i}" for i in range(25)], "max_length": 128,
                 "thresholds": [0.5] * 25})
    meta = read_json(path / "bundle.json")
    meta["labels"][0] = "different"
    write_json(path / "bundle.json", meta)
    with pytest.raises(ValueError):
        load_bundle(path)


def test_validation_calibration_and_test_gate(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.engine import train
    from watch_intents_training.evaluation import calibrate_run, evaluate_run, freeze_runs
    c = load_config(fixture_project)
    train(c, "run")
    result = calibrate_run(c, "run")
    assert 0.1 <= result["threshold"] <= 0.9
    with pytest.raises(ValueError, match="freeze"):
        evaluate_run(c, "run", "test")
    freeze = freeze_runs(c, ["run"])
    report = evaluate_run(c, "run", "test", freeze=freeze)
    assert report["metrics"]["overall"]["n"] == 9
    assert np.isfinite(report["metrics"]["overall"]["exact_set_accuracy"]["value"])


def test_scaler_overflow_skips_step_then_recovers():
    import torch

    from watch_intents_training.engine import optimize_group
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16.)
    batches = [({"input": torch.ones(2, 2)}, torch.ones(2, 1))]
    before = model.weight.detach().clone()
    hook = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    _, stepped = optimize_group(model, batches, optimizer, scaler, "cpu", "fp32", 1.)
    assert stepped is False
    assert scaler.get_scale() == 8.
    assert torch.equal(before, model.weight)
    hook.remove()
    _, stepped = optimize_group(model, batches, optimizer, scaler, "cpu", "fp32", 1.)
    assert stepped is True
    assert not torch.equal(before, model.weight)


def test_calibration_interrupted_report_is_rebuilt(fixture_project, monkeypatch):
    from watch_intents_training import evaluation
    from watch_intents_training.config import load_config
    from watch_intents_training.engine import train
    c = load_config(fixture_project)
    train(c, "interrupt-calibration")
    original = evaluation.write_report
    def interrupted(*args, **kwargs):
        raise OSError("simulated disk interruption")
    monkeypatch.setattr(evaluation, "write_report", interrupted)
    with pytest.raises(OSError, match="simulated"):
        evaluation.calibrate_run(c, "interrupt-calibration")
    monkeypatch.setattr(evaluation, "write_report", original)
    evaluation.calibrate_run(c, "interrupt-calibration")
    assert (c.output_dir / "interrupt-calibration/validation-report.md").exists()


def test_cli_prediction_uses_runtime_precision(fixture_project, monkeypatch):
    import contextlib

    from transformers import AutoTokenizer

    from watch_intents_training import engine
    from watch_intents_training.cli import execute, parser
    from watch_intents_training.model import IntentModel, save_bundle
    path = fixture_project.parent / "bundle"
    save_bundle(path, IntentModel.from_base(fixture_project.parent / "base", 25),
                AutoTokenizer.from_pretrained(fixture_project.parent / "base"),
                {"labels": [f"L{i}" for i in range(25)], "max_length": 128,
                 "thresholds": [0.5] * 25})
    called = []
    def capture_autocast(device, precision):
        called.append((device, precision))
        return contextlib.nullcontext()
    monkeypatch.setattr(engine, "autocast", capture_autocast)
    args = parser().parse_args(["--config", str(fixture_project), "predict",
                                "--bundle", str(path), "--text", "open sport"])
    assert isinstance(execute(args), list)
    assert called == [("cpu", "fp32")]
