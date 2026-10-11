"""CPU-only tests for the demo notebooks' helpers: src/demo.py (DemoSession,
with every stage faked) and src/demo_ui.py (rows, saved sessions, .env
loading, golden clips). No model is loaded."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src import demo, demo_ui

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_pipeline import (FakeASRBundle, FakeAudioLID, FakeRAGResult, FakeTextLID,  # noqa: E402
                           FakeTTSResult, FakeUnloadable, make_translate)


class Loader:
    """Counts loads and hands out one fake per load."""

    def __init__(self, factory):
        self.factory = factory
        self.made: list = []

    def __call__(self, *args, **kwargs):
        self.made.append(self.factory(*args, **kwargs))
        return self.made[-1]


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    ns = type("Fakes", (), {})()
    ns.audio_lang, ns.text_lang = "fil", "fil"
    ns.audio_lid = Loader(lambda: FakeAudioLID(ns.audio_lang))
    ns.asr = Loader(lambda lang: FakeASRBundle(lang))
    ns.text_lid = Loader(lambda: FakeTextLID(ns.text_lang))
    ns.mt = Loader(FakeUnloadable)
    ns.rag = Loader(lambda data_dir=None: FakeUnloadable())
    ns.tts = Loader(FakeUnloadable)
    ns.rag_calls, ns.tts_calls = [], []
    ns.mt_in_ok = True

    def translate_in(bundle, items):
        fn = make_translate("en", ok_ids=None if ns.mt_in_ok else set())
        out = fn(bundle, items)
        for r in out.values():
            r.translate_time_sec = 0.5
        return out

    def translate_out(bundle, items):
        out = make_translate("native")(bundle, items)
        for r in out.values():
            r.translate_time_sec = 0.5
        return out

    def answer_turn(bundle, query, history):
        ns.rag_calls.append((query, list(history)))
        return FakeRAGResult(answer=f"answer-to-{query}", resolved_query=query, chunk_ids=["EXT-001"])

    def synthesize_turn(bundle, id, text, lang, init_set):
        ns.tts_calls.append((id, text, lang))
        return FakeTTSResult(id, lang, ok=not text.startswith("native-answer-to-en-silent"),
                             failure_reason="frontend raised")

    monkeypatch.setattr(demo.audio, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(demo.lid_audio, "load", ns.audio_lid)
    monkeypatch.setattr(demo.asr, "load", ns.asr)
    ckpts = {lang: tmp_path / f"asr_{lang}" for lang in ("ceb", "fil", "eng")}
    for p in ckpts.values():
        p.mkdir()
    monkeypatch.setattr(demo.asr, "CHECKPOINTS", {k: str(v) for k, v in ckpts.items()})
    monkeypatch.setattr(demo.lid_text, "load", ns.text_lid)
    monkeypatch.setattr(demo.mt, "load", ns.mt)
    monkeypatch.setattr(demo.mt, "translate_to_english", translate_in)
    monkeypatch.setattr(demo.mt, "translate_from_english", translate_out)
    monkeypatch.setattr(demo.rag, "load", ns.rag)
    monkeypatch.setattr(demo.rag, "answer_turn", answer_turn)
    monkeypatch.setattr(demo.tts, "load", ns.tts)
    monkeypatch.setattr(demo.tts, "synthesize_turn", synthesize_turn)
    monkeypatch.setattr(demo, "gpu_memory", lambda: "n/a")
    ns.clip = tmp_path / "q.wav"
    ns.clip.write_bytes(b"")
    return ns


# --- DemoSession -------------------------------------------------------------------


def test_text_turn_detects_language_once_and_locks_it(fakes):
    s = demo.DemoSession(tts_backend="hf")
    t1 = s.ask_text("Unsa ang requirements?")
    t2 = s.ask_text("Pila ang bayad?")

    assert s.final_lang == "fil" and t1.record.final_lang == t2.record.final_lang == "fil"
    assert fakes.text_lid.made[0].calls == ["Unsa ang requirements?"]  # turn 1 only
    assert t1.lang_source == "GlotLID 0.80" and t2.lang_source == "locked from the first turn"
    assert t1.record.english_query == "en-Unsa ang requirements?"
    assert t1.record.native_answer == "native-answer-to-en-Unsa ang requirements?"
    assert t1.record.tts_ok is None and t1.audio is None  # speak defaults to False for text
    assert set(t1.stage_s) == {"text LID", "MT in", "RAG", "MT out"}


def test_text_turn_in_another_language_needs_reset(fakes):
    s = demo.DemoSession(tts_backend="hf")
    s.ask_text("first", lang="ceb")
    with pytest.raises(ValueError, match="call reset"):
        s.ask_text("second", lang="eng")
    s.reset()
    assert s.ask_text("second", lang="eng").record.final_lang == "eng"
    assert fakes.text_lid.made == []  # lang= skips GlotLID
    with pytest.raises(ValueError, match="one of"):
        s.ask_text("third", lang="tl")


def test_audio_first_turn_routes_like_run_staged(fakes):
    fakes.audio_lang, fakes.text_lang = "fil", "ceb"  # text LID overrides audio LID
    s = demo.DemoSession(tts_backend="hf")
    t1 = s.ask_audio(fakes.clip)
    t2 = s.ask_audio(fakes.clip)

    assert (t1.record.pred_lang, t1.record.text_lang, t1.record.final_lang) == ("fil", "ceb", "ceb")
    assert t1.record.was_overridden is True
    assert t2.record.pred_lang is None and t2.record.final_lang == "ceb"
    # ASR keeps audio LID's checkpoint for every turn; each LID runs once.
    assert [b.lang for b in fakes.asr.made] == ["fil"] and len(fakes.asr.made[0].calls) == 2
    assert fakes.audio_lid.made[0].calls == 1 and len(fakes.text_lid.made[0].calls) == 1
    assert "text LID overrode audio LID" in t1.lang_source
    assert t1.question_audio is not None and t1.record.tts_ok is True  # speak defaults to True for audio


def test_audio_lang_skips_both_lids(fakes):
    s = demo.DemoSession(tts_backend="hf")
    t = s.ask_audio(fakes.clip, lang="eng", speak=False)
    assert t.record.final_lang == "eng" and t.lang_source == "set by you"
    assert fakes.audio_lid.made == [] and fakes.text_lid.made == []
    assert [b.lang for b in fakes.asr.made] == ["eng"]


def test_audio_refuses_an_asr_checkpoint_that_is_not_a_local_folder(fakes, monkeypatch):
    monkeypatch.setitem(demo.asr.CHECKPOINTS, "fil", "lingkodai/lingkodai-whisper-fil")
    s = demo.DemoSession(tts_backend="hf")
    with pytest.raises(RuntimeError, match="patch_asr_tokenizers"):
        s.ask_audio(fakes.clip)


def test_a_failed_turn_records_nothing_and_locks_nothing(fakes):
    fakes.mt_in_ok = False
    s = demo.DemoSession(tts_backend="hf")
    with pytest.raises(RuntimeError, match="MT in"):
        s.ask_text("hello")
    assert s.turns == [] and s.history == [] and s.final_lang is None
    fakes.mt_in_ok = True
    s.ask_text("hello", lang="eng")
    assert s.final_lang == "eng"


def test_history_feeds_follow_ups_in_english(fakes):
    s = demo.DemoSession(tts_backend="hf")
    s.ask_text("one", lang="ceb")
    s.ask_text("two")
    query, history = fakes.rag_calls[1]
    assert query == "en-two"
    assert [(h.type, h.content) for h in history] == [("human", "en-one"), ("ai", "answer-to-en-one")]


def test_models_stay_loaded_unless_keep_loaded_is_false(fakes):
    s = demo.DemoSession(tts_backend="hf")
    s.ask_text("a", lang="fil", speak=True)
    s.ask_text("b", speak=True)
    assert (len(fakes.mt.made), len(fakes.rag.made), len(fakes.tts.made)) == (1, 1, 1)
    assert sorted(s.loaded) == ["MT (NLLB)", "RAG", "TTS (hf)"]
    s.unload()
    assert s.loaded == [] and all(m.unloaded for m in fakes.mt.made + fakes.rag.made + fakes.tts.made)

    lean = demo.DemoSession(tts_backend="hf", keep_loaded=False)
    lean.ask_text("c", lang="fil", speak=True)
    assert len(fakes.mt.made) == 3  # loaded for MT in and again for MT out
    assert lean.loaded == [] and all(m.unloaded for m in fakes.mt.made[1:])


def test_speech_uses_the_native_answer_and_a_text_only_turn_is_not_an_error(fakes):
    s = demo.DemoSession(tts_backend="hf")
    t = s.ask_text("hi", lang="fil", speak=True)
    assert fakes.tts_calls[-1] == (0, "native-answer-to-en-hi", "fil")
    assert t.audio is not None and t.record.tts_backend == "hf" and "TTS" in t.stage_s

    t2 = s.ask_text("silent", speak=True)
    assert t2.record.tts_ok is False and t2.audio is None
    assert demo_ui.audio_note(t2.to_json_dict()) == "text only: frontend raised"
    assert len(s.turns) == 2


def test_session_rejects_bad_backend_options():
    with pytest.raises(ValueError, match="Unknown tts_backend"):
        demo.DemoSession(tts_backend="onnx")
    with pytest.raises(ValueError, match="cpp backend only"):
        demo.DemoSession(tts_backend="hf", tts_options={"mode": "segmented"})


def test_save_writes_the_run_conversation_layout_and_replays(fakes, tmp_path):
    s = demo.DemoSession(tts_backend="hf")
    s.ask_audio(fakes.clip, lang="ceb")
    s.ask_audio(fakes.clip, speak=False)
    out = s.save(tmp_path / "session")

    assert sorted(p.name for p in out.iterdir()) == [
        "manifest.json", "turn_0.json", "turn_0.wav", "turn_0_question.wav", "turn_1.json", "turn_1_question.wav"]
    manifest, turns = demo_ui.load_session(out)
    assert manifest["source"] == "demo" and manifest["final_lang"] == "ceb"
    assert turns[0]["input_mode"] == "audio" and turns[0]["chunk_titles"]["EXT-001"].startswith("ISSUANCE")
    assert turns[0]["_wav"].name == "turn_0.wav" and turns[1]["_wav"] is None
    assert turns[1]["_question_audio"].name == "turn_1_question.wav"
    assert demo_ui.find_sessions(tmp_path) == [out]
    with pytest.raises(FileExistsError):
        s.save(out)


# --- demo_ui -------------------------------------------------------------------------


def test_turn_rows_for_a_text_turn_and_a_golden_check_turn():
    text_turn = {"input_mode": "text", "typed_text": "Pila?", "final_lang": "fil", "lang_source": "GlotLID 0.97",
                 "english_query": "How much?", "chunk_ids": ["EXT-001"], "english_answer": "PHP 75.",
                 "native_answer": "PHP 75.", "tts_ok": None, "stage_s": {"MT in": 0.4, "RAG": 8.0}}
    rows = dict(demo_ui.turn_rows(text_turn))
    assert rows["You typed"] == "Pila?" and rows["Language"] == "Filipino (GlotLID 0.97)"
    assert rows["Sources"].startswith("EXT-001 ISSUANCE OF CERTIFIED TRUE COPY")
    assert rows["Audio"] == "not requested" and rows["Time"] == "MT in 0.4 s | RAG 8.0 s"

    golden = {"clip_id": "eng_01_mt_001", "turn_idx": 0, "transcript": "How do I renew?", "pred_lang": "eng",
              "lid_confidence": 0.99, "text_lang": "eng", "text_lid_confidence": 0.95, "final_lang": "eng",
              "english_query": "How do I renew?", "chunk_ids": [], "english_answer": "x", "native_answer": "x",
              "tts_ok": True, "tts_backend": "cpp", "tts_wall_sec": 3.0, "tts_rtf": 0.4}
    rows = dict(demo_ui.turn_rows(golden))
    assert rows["Heard (ASR)"] == "How do I renew?"
    assert rows["Language"] == "English (audio LID eng 0.99, text LID eng 0.95)"
    assert "In English" not in rows and "Answer in English" not in rows  # eng needs no pivot
    assert rows["Sources"] == "none" and rows["Audio"] == "cpp TTS, 3.0 s to synthesize (RTF 0.40)"


def test_html_and_chat_bubbles_escape_text():
    turn = {"typed_text": "<script>x</script>", "native_answer": "a & b"}
    out = demo_ui.turn_html(turn, title="T<1>")
    assert "<script>" not in out and "&lt;script&gt;" in out and "a &amp; b" in out and "T&lt;1&gt;" in out
    bubble = demo_ui.bubble_html("LingkodAI", "<b>hi</b>", meta="ceb | 3.0 s")
    assert "&lt;b&gt;hi&lt;/b&gt;" in bubble and "ceb | 3.0 s" in bubble


def test_answer_meta_summarizes_language_sources_and_time():
    turn = {"final_lang": "ceb", "chunk_ids": ["EXT-001", "EXT-002"], "stage_s": {"MT in": 0.5, "RAG": 2.5},
            "tts_ok": False, "tts_failure_reason": "cap"}
    assert demo_ui.answer_meta(turn) == "Cebuano | sources EXT-001, EXT-002 | 3.0 s | no audio: cap"


def test_load_env_file_sets_only_unset_non_empty_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nDEMO_A=1\nexport DEMO_B='two'\nDEMO_C=\nDEMO_D=already\n\nnot a line\n")
    monkeypatch.setenv("DEMO_D", "kept")
    for k in ("DEMO_A", "DEMO_B", "DEMO_C"):
        monkeypatch.delenv(k, raising=False)
    assert demo_ui.load_env_file(env) == ["DEMO_A", "DEMO_B"]
    import os

    assert (os.environ["DEMO_A"], os.environ["DEMO_B"], os.environ["DEMO_D"]) == ("1", "two", "kept")
    assert "DEMO_C" not in os.environ
    assert demo_ui.load_env_file(tmp_path / "missing.env") == []


def test_find_and_load_a_golden_check_folder_in_turn_order(tmp_path):
    folder = tmp_path / "golden"
    folder.mkdir()
    for cid in ("eng_01_mt_002", "ceb_01_mt_10", "ceb_01_mt_02", "eng_01_mt_001"):
        (folder / f"{cid}.json").write_text(json.dumps({"clip_id": cid, "final_lang": cid[:3]}))
    (folder / "eng_01_mt_001.wav").write_bytes(b"")
    (folder / "report.json").write_text("{}")
    assert demo_ui.find_sessions(tmp_path) == [folder]
    manifest, turns = demo_ui.load_session(folder)
    assert manifest == {}
    assert [t["clip_id"] for t in turns] == ["ceb_01_mt_02", "ceb_01_mt_10", "eng_01_mt_001", "eng_01_mt_002"]
    assert turns[2]["_wav"] is not None and turns[3]["_wav"] is None


def test_golden_audio_root_matches_the_fetch_script():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import fetch_eval_assets

    assert demo_ui.DS6_AUDIO_ROOT == fetch_eval_assets.DS6_AUDIO_ROOT
    assert demo_ui.GOLDEN_CONVERSATION == fetch_eval_assets.CONVERSATION


def test_golden_clips_maps_ds6_paths_into_the_assets(tmp_path):
    pytest.importorskip("openpyxl")
    import pandas as pd

    rows = []
    for lang, cid_fmt in (("eng", "eng_01_mt_{:03d}"), ("ceb", "ceb_01_mt_{:02d}")):
        for n in (2, 1):
            cid = cid_fmt.format(n)
            rel = f"multi_turn/01_{lang}_mt_audio/{cid}.m4a"
            rows.append({"clip_id": cid, "conversation_id": "CONV-MT-001", "turn_id": f"CONV-MT-001-T0{n}",
                         "source_audio_path": f"{demo_ui.DS6_AUDIO_ROOT}/{rel}", "reference_transcript": f"q {cid}",
                         "english_reference": f"e {cid}"})
            (tmp_path / "golden" / "audio" / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "golden" / "audio" / rel).write_bytes(b"")
    rows.append({**rows[0], "clip_id": "eng_02_mt_001", "conversation_id": "CONV-MT-002"})
    pd.DataFrame(rows).to_excel(tmp_path / "ds6.xlsx", index=False)

    clips = demo_ui.golden_clips(tmp_path / "ds6.xlsx", tmp_path)
    assert [c["clip_id"] for c in clips] == ["ceb_01_mt_01", "ceb_01_mt_02", "eng_01_mt_001", "eng_01_mt_002"]
    assert clips[0]["turn"] == 1 and clips[0]["question"] == "q ceb_01_mt_01" and clips[0]["path"].is_file()

    (tmp_path / "golden" / "audio" / "multi_turn" / "01_ceb_mt_audio" / "ceb_01_mt_02.m4a").unlink()
    with pytest.raises(FileNotFoundError, match="1 golden clip"):
        demo_ui.golden_clips(tmp_path / "ds6.xlsx", tmp_path)


def _fake_notebook_modules(monkeypatch):
    """Minimal ipywidgets and IPython.display stand-ins that record what is shown."""
    import types

    shown: list = []

    class Widget:
        def __init__(self, *children, **kwargs):
            self.children, self.value, self.disabled = children, kwargs.get("value", ""), False
            self._clicks = []

        def on_click(self, fn):
            self._clicks.append(fn)

    class Output(Widget):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    w = types.ModuleType("ipywidgets")
    for name in ("Text", "Button", "Checkbox", "HTML", "HBox", "VBox"):
        setattr(w, name, type(name, (Widget,), {}))
    w.Output, w.Layout = Output, lambda **kw: kw
    ipy, disp = types.ModuleType("IPython"), types.ModuleType("IPython.display")
    disp.HTML = lambda s: ("html", s)
    disp.Audio = lambda *a, **kw: ("audio", a, kw)
    disp.display = shown.append
    ipy.display = disp
    for name, mod in (("ipywidgets", w), ("IPython", ipy), ("IPython.display", disp)):
        monkeypatch.setitem(sys.modules, name, mod)
    return shown


def test_chat_box_shows_question_answer_audio_and_errors(fakes, monkeypatch):
    shown = _fake_notebook_modules(monkeypatch)
    s = demo.DemoSession(tts_backend="hf")
    box = demo_ui.ChatBox(s, speak=True)

    box.submit("  Pila ang bayad?  ")
    kinds = [item[0] for item in shown]
    assert kinds == ["html", "html", "audio"]
    assert "Pila ang bayad?" in shown[0][1] and "native-answer-to-en-Pila ang bayad?" in shown[1][1]
    assert box.text.value == "" and not box.send.disabled and len(s.turns) == 1

    shown.clear()
    box.speak.value, box.details.value = False, True
    box.submit("next")
    assert [item[0] for item in shown] == ["html", "html", "html"]  # bubble, answer, every-stage table

    shown.clear()
    fakes.mt_in_ok = False
    box.text.value = "broken"
    box.submit("broken")
    assert "RuntimeError: stage 4 (MT in) failed" in shown[-1][1]
    assert box.text.value == "broken" and not box.send.disabled and len(s.turns) == 2

    box.new_conversation()
    assert s.turns == [] and s.final_lang is None and "New conversation" in shown[-1][1]
    box.submit("   ")  # empty: ignored
    assert s.turns == []


def test_show_session_plays_a_saved_demo_session(fakes, monkeypatch, tmp_path):
    s = demo.DemoSession(tts_backend="hf")
    s.ask_audio(fakes.clip, lang="ceb")
    folder = s.save(tmp_path / "saved")
    shown = _fake_notebook_modules(monkeypatch)
    demo_ui.show_session(folder)
    kinds = [item[0] for item in shown]
    assert kinds == ["html", "html", "html", "html", "audio", "html", "audio"]
    assert "saved: 1 turn(s), TTS hf" in shown[0][1] and "Cebuano" in shown[1][1]
    assert shown[4][2]["filename"].endswith("turn_0_question.wav")
    assert shown[6][2]["filename"].endswith("turn_0.wav")
