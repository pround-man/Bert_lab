"""XLM-R encoder with the same simple head used in the design document."""

import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .io import directory_hashes, read_json, verify_files, write_json


class IntentModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_base(cls, directory, num_labels):
        encoder = AutoModel.from_pretrained(directory, local_files_only=True,
                                            add_pooling_layer=False, use_safetensors=True,
                                            attn_implementation="eager")
        if encoder.config.model_type != "xlm-roberta":
            raise ValueError("This baseline requires XLM-R")
        return cls(encoder, num_labels)

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(hidden))


def save_bundle(directory, model, tokenizer, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.encoder.save_pretrained(directory / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.classifier.state_dict().items()},
              str(directory / "classifier.safetensors"))
    write_json(directory / "bundle.json", {"format_version": 1, "head": "first_token_dropout_linear",
                                           **metadata})
    labels = metadata["labels"]
    write_json(directory / "label2id.json", {name: i for i, name in enumerate(labels)})
    write_json(directory / "id2label.json", {str(i): name for i, name in enumerate(labels)})
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    if lock.is_file():
        shutil.copyfile(lock, directory / "environment.uv.lock")
    write_json(directory / "checksums.json", directory_hashes(directory))


def load_bundle(directory, device="cpu"):
    directory = Path(directory)
    verify_files(directory, read_json(directory / "checksums.json"))
    meta = read_json(directory / "bundle.json")
    labels, thresholds = meta["labels"], meta["thresholds"]
    if (read_json(directory / "label2id.json") != {name: i for i, name in enumerate(labels)}
            or read_json(directory / "id2label.json") != {str(i): name for i, name in enumerate(labels)}):
        raise ValueError("Bundle label mapping mismatch")
    if (meta.get("format_version") != 1 or len(labels) != 25 or len(set(labels)) != 25
            or len(thresholds) != len(labels) or any(not 0 < x < 1 for x in thresholds)
            or type(meta["max_length"]) is not int or not 1 <= meta["max_length"] <= 512):
        raise ValueError("Invalid model bundle contract")
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    model = IntentModel.from_base(directory / "encoder", len(labels))
    if (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) != (0, 1, 2):
        raise ValueError("Unexpected tokenizer special tokens")
    if len(tokenizer) != model.encoder.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match encoder")
    model.classifier.load_state_dict(load_file(str(directory / "classifier.safetensors")))
    return model.to(device).eval(), tokenizer, meta
