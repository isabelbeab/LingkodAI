"""CPU-only smoke test for src/audio.py. subprocess.run and librosa.load are
monkeypatched so this runs even when ffmpeg is not installed locally (it is a
system dependency per CLAUDE.md, not guaranteed on HK's laptop) -- this checks
the ffmpeg command construction and tempdir plumbing, not real decoding."""

from __future__ import annotations

import numpy as np

from src import audio


def test_load_audio_calls_ffmpeg_with_expected_args(monkeypatch):
    calls = []

    def fake_run(cmd, check, capture_output):
        calls.append(cmd)
        assert check is True
        assert capture_output is True

    def fake_librosa_load(wav_path, sr, mono):
        assert wav_path.endswith("audio_16k.wav")
        assert sr == audio.SAMPLE_RATE
        assert mono is True
        return np.zeros(1000, dtype=np.float32), sr

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(audio.librosa, "load", fake_librosa_load)

    result = audio.load_audio("some_input.wav")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-i") + 1] == "some_input.wav"
    assert cmd[cmd.index("-ar") + 1] == str(audio.SAMPLE_RATE)
    assert cmd[cmd.index("-ac") + 1] == "1"


def test_sample_rate_constant():
    # locked at 16 kHz -- every downstream stage (LID encoder, ASR, both
    # trained on this rate) assumes it.
    assert audio.SAMPLE_RATE == 16000
