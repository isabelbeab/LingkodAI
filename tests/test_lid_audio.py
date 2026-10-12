"""CPU-only smoke test for src/lid_audio.py. Never downloads whisper-medium
or the trained head -- FakeFeatureExtractor/FakeEncoder stand in for the
Whisper encoder. The classification head is lid_audio.build_head() with its
default (untrained, random-init) weights, since build_head() is plain
PyTorch and needs no download; this exercises the real chunking/masking/
pooling logic in head_forward(), just not a trained result."""

from __future__ import annotations

import numpy as np
import pytest
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


# --- Head weights from the Hub (resolve_head_path) ---------------------------
# hf_hub_download is replaced with a function returning a local temp file, so
# nothing is downloaded.


def _fake_download(path):
    def fake(repo_id, filename, revision):
        assert (repo_id, filename, revision) == (
            lid_audio.HEAD_REPO, lid_audio.HEAD_FILENAME, lid_audio.HEAD_REVISION)
        return str(path)
    return fake


def test_head_downloads_from_the_pinned_hub_file_and_checks_it(tmp_path, monkeypatch):
    head = tmp_path / "deeper_50chunks.pt"
    head.write_bytes(b"head weights")
    monkeypatch.setattr(lid_audio, "HEAD_PATH", None)
    monkeypatch.setattr(lid_audio, "HEAD_SHA256", lid_audio.sha256_file(head))
    monkeypatch.setattr(lid_audio, "hf_hub_download", _fake_download(head))
    assert lid_audio.resolve_head_path() == str(head)


def test_head_with_a_different_checksum_raises(tmp_path, monkeypatch):
    head = tmp_path / "deeper_50chunks.pt"
    head.write_bytes(b"some other head")
    monkeypatch.setattr(lid_audio, "HEAD_PATH", None)
    monkeypatch.setattr(lid_audio, "hf_hub_download", _fake_download(head))
    with pytest.raises(RuntimeError, match="Not loading a different audio LID head"):
        lid_audio.resolve_head_path()


def test_lid_head_override_skips_the_hub(tmp_path, monkeypatch):
    head = tmp_path / "my_head.pt"
    head.write_bytes(b"local")
    monkeypatch.setattr(lid_audio, "HEAD_PATH", str(head))
    monkeypatch.setattr(lid_audio, "hf_hub_download", lambda **kw: pytest.fail("downloaded"))
    assert lid_audio.resolve_head_path() == str(head)


def test_lid_head_override_that_is_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(lid_audio, "HEAD_PATH", str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="LID_HEAD"):
        lid_audio.resolve_head_path()


def test_pinned_checksum_is_the_formerly_committed_head():
    # models/deeper_50chunks.pt's SHA-256 before it moved to the Hub
    # (CHANGELOG.md, 2026-09-28 and 2026-10-12).
    assert lid_audio.HEAD_SHA256 == "101c32f02a254a1596617d44a98c9eccbc0f761862232b787ddf31e2b066dc99"
