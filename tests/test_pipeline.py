"""CPU-only smoke test for src/pipeline.py. Never loads a real model for any
stage -- each stage module already has its own smoke test (test_mt.py,
test_rag.py, test_tts.py, and lid_audio/asr/lid_text are Bea's, untested
here). This test only checks the wiring run_staged is responsible for:
audio/text LID running on turn 0 only, the locked final_lang propagating to
every turn, per-turn ASR, RAG history accumulation, and a mid-run MT failure
raising instead of silently continuing (the only designed fallbacks are
RAG's no-match message and TTS's text-only turn -- a failed MT stage is not
one of them)."""

from __future__ import annotations

import numpy as np
import pytest

from src import pipeline


class FakeUnloadable:
    def __init__(self):
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class FakeLIDResult:
    def __init__(self, lang, confidence=0.9):
        self.lang = lang
        self.confidence = confidence


class FakeAudioLID(FakeUnloadable):
    def __init__(self, lang):
        super().__init__()
        self.lang = lang
        self.calls = 0

    def predict(self, audio):
        self.calls += 1
        return FakeLIDResult(self.lang)


class FakeASRBundle(FakeUnloadable):
    def __init__(self, lang):
        super().__init__()
        self.lang = lang
        self.calls: list = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return f"transcript-{self.lang}-{len(self.calls)}"


class FakeTextLIDResult:
    def __init__(self, lang, confidence=0.8):
        self.lang = lang
        self.confidence = confidence


class FakeTextLID:
    def __init__(self, lang):
        self.lang = lang
        self.calls: list = []

    def predict(self, text):
        self.calls.append(text)
        return FakeTextLIDResult(self.lang)


class FakeMTResult:
    def __init__(self, id, text, ok=True, failure_reason=None):
        self.id = id
        self.text = text
        self.ok = ok
        self.failure_reason = failure_reason


def make_translate(prefix, ok_ids=None):
    def _translate(bundle, items):
        results = {}
        for item in items:
            ok = ok_ids is None or item.id in ok_ids
            text = f"{prefix}-{item.text}" if ok else None
            reason = None if ok else "simulated MT failure"
            results[item.id] = FakeMTResult(item.id, text, ok, reason)
        return results

    return _translate


class FakeRAGResult:
    def __init__(self, answer, resolved_query, chunk_ids=None, was_dependent=False):
        self.answer = answer
        self.resolved_query = resolved_query
        self.chunk_ids = chunk_ids or []
        self.was_dependent = was_dependent
        self.retrieval_s = 0.01
        self.select_s = 0.01
        self.generate_s = 0.01


class FakeTTSResult:
    def __init__(self, id, lang, ok=True, failure_reason=None):
        self.id = id
        self.lang = lang
        self.audio = np.zeros(10) if ok else None
        self.sr = 24000 if ok else None
        self.wall_sec = 0.1
        self.rtf = 0.1
        self.ok = ok
        self.failure_reason = failure_reason


def _default_answer_turn(bundle, query, history):
    return FakeRAGResult(answer=f"answer-to-{query}", resolved_query=query, chunk_ids=["c1"])


def _default_synthesize_turn(bundle, id, text, lang, initialism_set):
    return FakeTTSResult(id, lang)


@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    monkeypatch.setattr(pipeline.audio, "load_audio", lambda path: np.zeros(1600))

    monkeypatch.setattr(pipeline.lid_audio, "load", lambda: FakeAudioLID("ceb"))
    monkeypatch.setattr(pipeline.asr, "load", lambda lang: FakeASRBundle(lang))
    monkeypatch.setattr(pipeline.lid_text, "load", lambda: FakeTextLID("ceb"))

    monkeypatch.setattr(pipeline.mt, "load", lambda: FakeUnloadable())
    monkeypatch.setattr(pipeline.mt, "translate_to_english", make_translate("EN"))
    monkeypatch.setattr(pipeline.mt, "translate_from_english", make_translate("NATIVE"))

    monkeypatch.setattr(pipeline.rag, "load", lambda data_dir=None: FakeUnloadable())
    monkeypatch.setattr(pipeline.rag, "answer_turn", _default_answer_turn)

    monkeypatch.setattr(pipeline.tts, "load", lambda: FakeUnloadable())
    monkeypatch.setattr(pipeline.tts, "synthesize_turn", _default_synthesize_turn)


def test_run_staged_two_turns_happy_path():
    result = pipeline.run_staged(["turn1.wav", "turn2.wav"], init_set=set())

    assert len(result.turns) == 2
    assert result.routing.final_lang == "ceb"

    t0, t1 = result.turns
    assert t0.pred_lang == "ceb"
    assert t0.text_lang == "ceb"
    assert t0.was_overridden is False
    assert t1.pred_lang is None  # turn 1 never runs audio/text LID
    assert t1.text_lang is None

    assert t0.final_lang == "ceb"  # but the lock propagates to every turn
    assert t1.final_lang == "ceb"

    assert t0.transcript == "transcript-ceb-1"
    assert t1.transcript == "transcript-ceb-2"

    assert t0.mt_in_ok is True
    assert t0.english_query.startswith("EN-")
    assert t0.english_answer.startswith("answer-to-")
    assert t0.mt_out_ok is True
    assert t0.native_answer.startswith("NATIVE-")
    assert t0.tts_ok is True
    assert t0.audio_out is not None


def test_run_staged_locks_language_from_turn_one_only():
    result = pipeline.run_staged(["a.wav", "b.wav", "c.wav"], init_set=set())
    assert {t.final_lang for t in result.turns} == {"ceb"}


def test_run_staged_audio_and_text_lid_run_once(monkeypatch):
    audio_lid = FakeAudioLID("ceb")
    text_lid = FakeTextLID("ceb")
    monkeypatch.setattr(pipeline.lid_audio, "load", lambda: audio_lid)
    monkeypatch.setattr(pipeline.lid_text, "load", lambda: text_lid)

    pipeline.run_staged(["a.wav", "b.wav", "c.wav"], init_set=set())

    assert audio_lid.calls == 1
    assert len(text_lid.calls) == 1


def test_run_staged_rag_history_accumulates(monkeypatch):
    seen_history_lengths = []

    def fake_answer_turn(bundle, query, history):
        seen_history_lengths.append(len(history))
        return FakeRAGResult(answer=f"answer-{query}", resolved_query=query)

    monkeypatch.setattr(pipeline.rag, "answer_turn", fake_answer_turn)

    pipeline.run_staged(["a.wav", "b.wav", "c.wav"], init_set=set())

    assert seen_history_lengths == [0, 2, 4]


def test_run_staged_raises_on_mt_in_failure(monkeypatch):
    monkeypatch.setattr(pipeline.mt, "translate_to_english", make_translate("EN", ok_ids={0}))

    with pytest.raises(RuntimeError, match="stage 4"):
        pipeline.run_staged(["a.wav", "b.wav"], init_set=set())


def test_run_staged_raises_on_mt_out_failure(monkeypatch):
    monkeypatch.setattr(pipeline.mt, "translate_from_english", make_translate("NATIVE", ok_ids={0}))

    with pytest.raises(RuntimeError, match="stage 6"):
        pipeline.run_staged(["a.wav", "b.wav"], init_set=set())


def test_run_staged_tts_failure_keeps_text_answer(monkeypatch):
    def fake_synthesize_turn(bundle, id, text, lang, initialism_set):
        if id == 1:
            return FakeTTSResult(id, lang, ok=False, failure_reason="frontend raised")
        return FakeTTSResult(id, lang)

    monkeypatch.setattr(pipeline.tts, "synthesize_turn", fake_synthesize_turn)

    result = pipeline.run_staged(["a.wav", "b.wav"], init_set=set())

    assert result.turns[1].tts_ok is False
    assert result.turns[1].audio_out is None
    assert result.turns[1].native_answer is not None  # text path unaffected


def test_run_staged_empty_audio_paths_raises():
    with pytest.raises(ValueError, match="empty"):
        pipeline.run_staged([], init_set=set())


def test_to_json_dict_excludes_audio_array():
    turn = pipeline.TurnRecord(turn_idx=0, audio_path="a.wav")
    turn.audio_out = np.zeros(10)
    turn.audio_out_sr = 24000

    d = turn.to_json_dict()

    assert "audio_out" not in d
    assert d["has_audio"] is True
    assert d["audio_out_sr"] == 24000
