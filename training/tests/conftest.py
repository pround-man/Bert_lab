import json
from pathlib import Path

import pytest


@pytest.fixture
def fixture_project(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    from watch_intents_training.io import digest, file_hash, write_json

    torch.set_num_threads(1)
    model = tmp_path / "base"
    model.mkdir()
    vocab = {word: i for i, word in enumerate(
        ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "open", "close", "sport", "music", "hello"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>",
                                                  special_tokens=[("<s>", 0), ("</s>", 2)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>",
                                  eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
                                  mask_token="<mask>", model_max_length=128)
    fast.save_pretrained(model)
    config = XLMRobertaConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=24,
                             max_position_embeddings=130, bos_token_id=0, pad_token_id=1,
                             eos_token_id=2)
    XLMRobertaModel(config, add_pooling_layer=False).save_pretrained(model)
    data = tmp_path / "data"
    data.mkdir()
    catalog = {"version": "0.1", "labels": [{"id": i, "name": f"L{i}"} for i in range(25)]}
    write_json(data / "labels.json", catalog)
    write_json(data / "label2id.json", {f"L{i}": i for i in range(25)})
    write_json(data / "id2label.json", {str(i): f"L{i}" for i in range(25)})
    files = {}
    for split in ("train", "validation", "test"):
        rows = []
        for i in range(9):
            labels = [[], ["L0"], ["L0", "L1"]][i % 3]
            rows.append({"sample_id": f"{split}-{i}", "group_id": f"g-{split}-{i}",
                             "text": ["hello", "open sport", "open sport music"][i % 3],
                             "labels": labels, "locale": ["en", "zh-Hans", "th"][i // 3],
                             "kind": ["zero", "single", "multi"][i % 3], "scenario": "direct",
                             "split": split, "label_version": "0.1", "review_status": "machine_accepted"})
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[path.name] = {"rows": len(rows), "sha256": file_hash(path)}
    identity = {"catalog_sha256": digest(catalog), "assets": {
        "tokenizer_sha256": file_hash(model / "tokenizer.json"),
        "model_config_sha256": file_hash(model / "config.json"),
        "weights_size_bytes": (model / "model.safetensors").stat().st_size}}
    write_json(data / "manifest.json", {"stage": "full", "complete": True, "accepted": 27,
               "expected": 27, "files": files, "identity": identity,
               "fingerprint": digest(identity), "sample_signature": "fixture"})
    write_json(data / "verification.json", {"ok": True, "errors": [], "accepted": 27,
               "expected": 27, "stage": "full", "sample_signature": "fixture"})
    config_path = tmp_path / "run.toml"
    config_path.write_text('data_dir="data"\nmodel_dir="base"\noutput_dir="out"\n'
                           'device="cpu"\nprecision="fp32"\nepochs=2\n'
                           'micro_batch=2\ngradient_accumulation=2\neval_batch=4\n')
    return Path(config_path)
