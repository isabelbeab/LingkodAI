"""CPU-only smoke test for src/tts.py. Never loads Qwen3-TTS for real -- a
FakeQwen3TTSModel stands in for the qwen_tts object tts.py calls. Uses the
real, vendored tts_frontend modules (pure Python, no torch at import time)
against plain text with no digits or acronyms, so nothing there raises."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src import tts


# --- fakes -------------------------------------------------------------


class FakeQwen3TTSModel:
    """call_plan: list of None (succeed) or an Exception instance to raise,
    consumed in order across .generate_voice_clone() calls. Once exhausted,
    calls succeed. Returns a 1-second 24kHz sine-ish wav per call."""

    def __init__(self, call_plan=None, sr=24000):
        self.calls = []
        self._call_plan = list(call_plan) if call_plan is not None else []
        self._sr = sr

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        if self._call_plan:
            behavior = self._call_plan.pop(0)
            if behavior is not None:
                raise behavior
        wav = np.ones(self._sr, dtype=np.float32) * 0.1
        return [wav], self._sr


def make_bundle(tmp_path, call_plan=None, langs=("eng", "fil", "ceb")) -> tts.TTSBundle:
    ref_config = {}
    for lang in langs:
        p = tmp_path / f"{lang}_ref.wav"
        sf.write(str(p), np.zeros(2400, dtype=np.float32), 24000)
        ref_config[lang] = {
            "ref_audio": p,
            "ref_text": f"{lang} reference text",
            "language": "english" if lang == "eng" else None,
        }
    return tts.TTSBundle(
        model=FakeQwen3TTSModel(call_plan=call_plan),
        device="cpu",
        ref_sr=24000,
        ref_config=ref_config,
    )


# --- _pad_ref_audio_with_silence ---------------------------------------


def test_pad_ref_audio_appends_silence(tmp_path):
    src = tmp_path / "clip.wav"
    sf.write(str(src), np.ones(1000, dtype=np.float32), 1000)  # 1.0s @ 1000Hz

    out_path = tts._pad_ref_audio_with_silence(src, silence_sec=0.5, out_dir=tmp_path / "out")

    padded, sr = sf.read(str(out_path))
    assert sr == 1000
    assert len(padded) == 1500  # 1.0s audio + 0.5s silence
    assert np.all(padded[1000:] == 0.0)


# --- load_init_set -------------------------------------------------------


def test_load_init_set_reads_tokens(tmp_path):
    path = tmp_path / "init_set_ds6.json"
    path.write_text('{"tokens": ["PRC", "COR", "CPA"]}', encoding="utf-8")

    result = tts.load_init_set(path)

    assert result == {"PRC", "COR", "CPA"}


# --- _gen_with_oom_retry --------------------------------------------------


def test_gen_once_uses_ref_config_for_lang(tmp_path):
    bundle = make_bundle(tmp_path)

    wav, sr, wall = tts._gen_once(bundle, "hello", "eng", tts.MAX_NEW_TOKENS_PRIMARY)

    assert sr == 24000
    assert len(wav) == 24000
    call = bundle.model.calls[0]
    assert call["ref_text"] == "eng reference text"
    assert call["language"] == "english"
    assert call["x_vector_only_mode"] is False
    assert call["non_streaming_mode"] is True
    assert call["max_new_tokens"] == tts.MAX_NEW_TOKENS_PRIMARY


def test_oom_falls_back_to_smaller_token_budget(tmp_path):
    oom = torch_oom_error()
    bundle = make_bundle(tmp_path, call_plan=[oom])

    wav, sr, wall = tts._gen_with_oom_retry(bundle, "hello", "eng")

    assert len(bundle.model.calls) == 2
    assert bundle.model.calls[0]["max_new_tokens"] == tts.MAX_NEW_TOKENS_PRIMARY
    assert bundle.model.calls[1]["max_new_tokens"] == tts.MAX_NEW_TOKENS_FALLBACK


def test_non_oom_error_propagates(tmp_path):
    bundle = make_bundle(tmp_path, call_plan=[RuntimeError("something else broke")])

    with pytest.raises(RuntimeError, match="something else broke"):
        tts._gen_with_oom_retry(bundle, "hello", "eng")


def torch_oom_error():
    import torch

    return torch.cuda.OutOfMemoryError("simulated OOM")


# --- synthesize ------------------------------------------------------------


def test_synthesize_eng_calls_generate_once(tmp_path):
    bundle = make_bundle(tmp_path)

    audio, sr, wall, rtf = tts.synthesize(bundle, "Bring your PRC ID.", "eng", set())

    assert sr == 24000
    assert len(bundle.model.calls) == 1
    assert bundle.model.calls[0]["language"] == "english"


def test_synthesize_fil_calls_generate_once(tmp_path):
    bundle = make_bundle(tmp_path)

    audio, sr, wall, rtf = tts.synthesize(bundle, "Kumusta ka", "fil", set())

    assert len(bundle.model.calls) == 1
    assert bundle.model.calls[0]["language"] is None


def test_synthesize_ceb_generates_per_segment(tmp_path):
    bundle = make_bundle(tmp_path)

    audio, sr, wall, rtf = tts.synthesize(bundle, "Kumusta ka karon", "ceb", set())

    assert sr == bundle.ref_sr
    assert len(bundle.model.calls) >= 1
    assert all(c["language"] is None for c in bundle.model.calls)


def test_synthesize_unknown_lang_raises(tmp_path):
    bundle = make_bundle(tmp_path)

    with pytest.raises(ValueError, match="Unknown lang"):
        tts.synthesize(bundle, "text", "xyz", set())


# --- synthesize_turn: the non-crashing entry point ------------------------


def test_synthesize_turn_success(tmp_path):
    bundle = make_bundle(tmp_path)

    result = tts.synthesize_turn(bundle, "turn-1", "Bring your PRC ID.", "eng", set())

    assert result.ok is True
    assert result.audio is not None
    assert result.failure_reason is None


def test_synthesize_turn_catches_raise_and_keeps_text_path(tmp_path):
    bundle = make_bundle(tmp_path, langs=("ceb",))

    # CEB raises on any digit until CEB_DIGIT_READINGS has evidence-backed
    # entries -- exactly the "raises by design" case the module docstring
    # describes.
    result = tts.synthesize_turn(bundle, "turn-2", "Kita ta sa 3:00", "ceb", set())

    assert result.ok is False
    assert result.audio is None
    assert result.sr is None
    assert result.failure_reason is not None
    assert result.id == "turn-2"
    assert result.lang == "ceb"


# --- reference clip config (DECISIONS.md #8) ---------------------------


def test_find_ref_clip_finds_nested_and_raises_when_missing(tmp_path):
    nested = tmp_path / "ENG" / "1904"
    nested.mkdir(parents=True)
    clip = nested / tts.REF_CLIP_FILENAMES["eng"]
    clip.write_bytes(b"x")
    assert tts.find_ref_clip(tmp_path, tts.REF_CLIP_FILENAMES["eng"]) == clip
    with pytest.raises(FileNotFoundError, match=str(tmp_path.resolve())):
        tts.find_ref_clip(tmp_path, tts.REF_CLIP_FILENAMES["fil"])


def test_find_ref_clip_ambiguous_raises(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / tts.REF_CLIP_FILENAMES["ceb"]).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="ambiguous"):
        tts.find_ref_clip(tmp_path, tts.REF_CLIP_FILENAMES["ceb"])


def test_ref_text_config_locked():
    assert tts.REF_CLIP_FILENAMES["fil"].endswith("0443.wav")
    assert "siyete" in tts.REF_TEXT_CONFIG["ceb"]["ref_text"]
    assert tts.REF_TEXT_CONFIG["eng"]["language"] == "english"
    assert tts.REF_TEXT_CONFIG["fil"]["language"] is None
    assert tts.REF_TEXT_CONFIG["ceb"]["language"] is None
