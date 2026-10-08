import tarfile

import pytest


def test_choose_learning_rate_and_seed_tie_policy():
    from watch_intents_training.workflow import select_learning_rate
    scores = {2e-5: {"exact_set_accuracy": .8, "macro_f1": .7},
              1e-5: {"exact_set_accuracy": .8, "macro_f1": .7},
              3e-5: {"exact_set_accuracy": .8, "macro_f1": .6}}
    assert select_learning_rate(scores) == 2e-5
    scores[1e-5]["macro_f1"] = .9
    assert select_learning_rate(scores) == 1e-5


def test_migration_archive_is_allowlisted_and_portable(tmp_path):
    from watch_intents_training.transfer import pack, verify_transfer
    root = tmp_path / "repo"
    for name in ("training/src/pkg/core.py", "training/configs/baseline.toml", "training/uv.lock",
                 "models/xlm-roberta-base/model.safetensors", "data/generated/dataset/exports/full/train.jsonl",
                 "training/.venv/secret", "dashscope.env", "build.log.txt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    archive = pack(root, tmp_path / "transfer.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert not any(".venv" in n or n.endswith(".env") or "build.log" in n for n in names)
        assert "Bert_lab/TRANSFER.json" in names
        tar.extractall(tmp_path / "unpack", filter="data")
    assert verify_transfer(tmp_path / "unpack/Bert_lab")["ok"]
    (tmp_path / "unpack/Bert_lab/training/src/pkg/core.py").write_text("changed")
    with pytest.raises(ValueError):
        verify_transfer(tmp_path / "unpack/Bert_lab")


def test_full_experiment_freezes_only_winning_lr_seeds(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.io import read_json
    from watch_intents_training.workflow import experiment
    c = load_config(fixture_project)
    result = experiment(c, "tiny")
    assert len(result["candidates"]) == 3
    assert len(result["search_runs"]) == 3
    assert len(list(c.output_dir.glob("tiny-*/run.json"))) == 5
    frozen = read_json(result["freeze"])
    assert set(frozen["candidates"]) == set(result["candidates"])
    assert result["selected_run"] in result["candidates"]
    assert experiment(c, "tiny", resume=True)["selected_run"] == result["selected_run"]
