"""CPU-only smoke test for src/lid_audio.py. Never downloads whisper-medium
or the trained head -- FakeFeatureExtractor/FakeEncoder stand in for the
Whisper encoder. The classification head is lid_audio.build_head() with its
default (untrained, random-init) weights, since build_head() is plain
PyTorch and needs no download; this exercises the real chunking/masking/
pooling logic in head_forward(), just not a trained result."""

from __future__ import annotations

import numpy as np
import torch

from src import lid_audio


class FakeFeatures:
    def __init__(self, tensor):
        self.input_features = tensor

    def to(self, device):
        return self


class FakeFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors="pt"):
        return FakeFeatures(torch.zeros(1, 80, 3000))


class FakeEncoderOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class FakeEncoder:
    """Stands in for the whisper encoder: returns a fixed-shape hidden state
    (1, MAX_FRAMES, 1024), matching the real encoder's output shape so the
    chunking math in AudioLID.predict runs unchanged."""

    def __init__(self, hidden=None):
        self.hidden = hidden if hidden is not None else torch.randn(1, lid_audio.MAX_FRAMES, 1024)

    def __call__(self, features):
        return FakeEncoderOutput(self.hidden)


def make_model(hidden=None) -> lid_audio.AudioLID:
    torch.manual_seed(0)
    return lid_audio.AudioLID(
        feature_extractor=FakeFeatureExtractor(),
        encoder=FakeEncoder(hidden),
        head=lid_audio.build_head(),
        device="cpu",
    )


def test_predict_returns_one_of_three_labels():
    model = make_model()
    audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s clip
    result = model.predict(audio)
    assert result.lang in lid_audio.LABELS
    assert set(result.probs) == set(lid_audio.LABELS)
    assert abs(sum(result.probs.values()) - 1.0) < 1e-5


def test_valid_chunks_short_clip_clips_to_one():
    assert lid_audio.valid_chunks(0.01) == 1


def test_valid_chunks_long_clip_clips_to_n_chunks():
    assert lid_audio.valid_chunks(60.0) == lid_audio.N_CHUNKS


def test_valid_chunks_matches_duration_math():
    # 15s at 50 frames/sec = 750 frames = 25 chunks of 30 frames each
    assert lid_audio.valid_chunks(15.0) == 25


def test_head_forward_output_shape():
    torch.manual_seed(0)
    head = lid_audio.build_head()
    chunks = torch.randn(2, lid_audio.N_CHUNKS, 1024)
    num_valid = torch.tensor([lid_audio.N_CHUNKS, 1])
    logits = lid_audio.head_forward(head, chunks, num_valid)
    assert logits.shape == (2, 3)


def test_head_forward_masks_invalid_chunks_not_nan():
    # a clip with only 1 valid chunk out of N_CHUNKS must not produce NaN --
    # that would mean the softmax mask let every chunk get -inf.
    torch.manual_seed(0)
    head = lid_audio.build_head()
    chunks = torch.randn(1, lid_audio.N_CHUNKS, 1024)
    num_valid = torch.tensor([1])
    logits = lid_audio.head_forward(head, chunks, num_valid)
    assert not torch.isnan(logits).any()


def test_unload_deletes_encoder_and_head():
    model = make_model()
    model.unload()
    assert not hasattr(model, "encoder")
    assert not hasattr(model, "head")
