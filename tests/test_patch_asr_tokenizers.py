"""CPU smoke tests for scripts/patch_asr_tokenizers.py (no models, no network)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_asr_tokenizers as p  # noqa: E402


def test_list_moves_to_additional_special_tokens():
    out = p.patch_tokenizer_config({"tokenizer_class": "WhisperTokenizer", "extra_special_tokens": ["<|a|>", "<|b|>"]})
    assert "extra_special_tokens" not in out
    assert out["additional_special_tokens"] == ["<|a|>", "<|b|>"]
    assert out["tokenizer_class"] == "WhisperTokenizer"


def test_dict_and_absent_left_alone():
    assert p.patch_tokenizer_config({"extra_special_tokens": {"x": "<|x|>"}}) == {"extra_special_tokens": {"x": "<|x|>"}}
    assert p.patch_tokenizer_config({"a": 1}) == {"a": 1}


def test_clash_raises():
    with pytest.raises(ValueError):
        p.patch_tokenizer_config({"extra_special_tokens": ["<|a|>"], "additional_special_tokens": ["<|b|>"]})


def test_input_not_mutated():
    cfg = {"extra_special_tokens": ["<|a|>"]}
    p.patch_tokenizer_config(cfg)
    assert cfg == {"extra_special_tokens": ["<|a|>"]}


def test_build_patched_dir_symlinks_weights_and_rewrites_config(tmp_path):
    snap, dest = tmp_path / "snap", tmp_path / "dest"
    (snap / "sub").mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "sub" / "x.json").write_text("{}")
    (snap / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": ["<|a|>"]}))

    p.build_patched_dir(snap, dest)

    assert (dest / "model.safetensors").is_symlink()
    assert (dest / "model.safetensors").read_bytes() == b"weights"
    assert (dest / "sub" / "x.json").read_text() == "{}"
    assert not (dest / "tokenizer_config.json").is_symlink()
    assert json.loads((dest / "tokenizer_config.json").read_text()) == {"additional_special_tokens": ["<|a|>"]}
    assert json.loads((dest / "tokenizer_config.json.orig").read_text()) == {"extra_special_tokens": ["<|a|>"]}
    p.build_patched_dir(snap, dest)  # idempotent rerun
