import json

import pytest


def test_config_paths_relative_to_file(fixture_project, monkeypatch, tmp_path):
    from watch_intents_training.config import load_config
    monkeypatch.chdir(tmp_path.parent)
    c = load_config(fixture_project)
    assert c.data_dir == fixture_project.parent / "data"
    assert c.micro_batch * c.gradient_accumulation == 4


def test_frozen_exports_and_multihot(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.data import FrozenData
    d = FrozenData(load_config(fixture_project))
    rows, targets = d.read_split("train")
    assert targets.shape == (9, 25)
    assert targets[0].sum() == 0
    assert targets[2].sum() == 2
    assert rows[1]["text"] == "open sport"


def test_tampered_file_and_mapping_rejected(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.data import FrozenData
    path = fixture_project.parent / "data/label2id.json"
    path.write_text(json.dumps({"L0": 7}))
    with pytest.raises(ValueError, match="mapping"):
        FrozenData(load_config(fixture_project))


def test_tampered_export_rejected(fixture_project):
    from watch_intents_training.config import load_config
    from watch_intents_training.data import FrozenData
    path = fixture_project.parent / "data/train.jsonl"
    path.write_text(path.read_text().replace("open sport", "close music"))
    with pytest.raises(ValueError, match="asset"):
        FrozenData(load_config(fixture_project))


@pytest.mark.parametrize("setting", ['micro_batch=0', 'learning_rate=nan',
                                     'unknown=1', 'epochs=true', 'max_length=513'])
def test_invalid_config_rejected(tmp_path, setting):
    from watch_intents_training.config import load_config
    path = tmp_path / "bad.toml"
    path.write_text('data_dir="d"\nmodel_dir="m"\noutput_dir="o"\n' + setting)
    with pytest.raises(ValueError):
        load_config(path)


def test_no_truncation(fixture_project):
    from transformers import AutoTokenizer

    from watch_intents_training.data import encode_texts
    tokenizer = AutoTokenizer.from_pretrained(fixture_project.parent / "base")
    with pytest.raises(ValueError, match="length"):
        encode_texts(tokenizer, ["open " * 130], 128)
