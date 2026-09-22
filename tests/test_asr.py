"""CPU-only smoke test for src/asr.py's ASRBundle.transcribe(). Never loads a
real Whisper checkpoint or touches the network -- fakes stand in for the
processor and model. load()/_build_ceb()/_build_native() are not exercised
here since they require a real HF download, which the laptop must not do:
no downloading or loading real models on HK's laptop, real GPU runs happen
on JOJIE only."""

from __future__ import annotations

import numpy as np
import torch

from src import asr


class FakeFeatures:
    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device, dtype=None):
        return self.tensor


class FakeFeatureExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, sampling_rate, return_tensors="pt"):
        self.calls.append({"sampling_rate": sampling_rate, "n_samples": len(audio)})
        return type("Batch", (), {"input_features": FakeFeatures(torch.zeros(1, 80, 3000))})()


class FakeTokenizer:
    def __init__(self, text):
        self.text = text
        self.decode_calls = []

    def batch_decode(self, generated, skip_special_tokens=True):
        self.decode_calls.append({"skip_special_tokens": skip_special_tokens})
        return [self.text]


class FakeProcessor:
    def __init__(self, text):
        self.feature_extractor = FakeFeatureExtractor()
        self.tokenizer = FakeTokenizer(text)


class FakeModel:
    def __init__(self):
        self.dtype = torch.float32
        self.generate_calls = []

    def generate(self, features, max_length=None):
        self.generate_calls.append({"max_length": max_length})
        return "GENERATED_PLACEHOLDER"


def make_bundle(lang="eng", text="fake transcript") -> asr.ASRBundle:
    return asr.ASRBundle(
        lang=lang,
        processor=FakeProcessor(text),
        model=FakeModel(),
        device="cpu",
    )


def test_transcribe_returns_decoded_text_stripped():
    bundle = make_bundle(text="  hello world  ")
    result = bundle.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "hello world"


def test_transcribe_passes_sample_rate_and_max_length():
    bundle = make_bundle()
    bundle.transcribe(np.zeros(16000, dtype=np.float32))

    assert bundle.processor.feature_extractor.calls[0]["sampling_rate"] == asr.SAMPLE_RATE
    assert bundle.model.generate_calls[0]["max_length"] == asr.MAX_LENGTH


def test_transcribe_decode_skips_special_tokens():
    bundle = make_bundle()
    bundle.transcribe(np.zeros(16000, dtype=np.float32))
    assert bundle.processor.tokenizer.decode_calls[0]["skip_special_tokens"] is True


def test_unload_removes_model_attribute():
    bundle = make_bundle()
    bundle.unload()
    assert not hasattr(bundle, "model")


def test_checkpoints_cover_all_three_languages():
    assert set(asr.CHECKPOINTS) == {"ceb", "fil", "eng"}


def test_whisper_lang_mapping():
    assert asr.WHISPER_LANG == {"fil": "tagalog", "eng": "english"}


def test_get_dtype_cpu_is_float32():
    assert asr.get_dtype("cpu") == torch.float32


def test_load_unknown_lang_raises():
    import pytest

    with pytest.raises(ValueError):
        asr.load("xyz", device="cpu")
