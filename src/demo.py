"""Resident-mode sessions for the demo notebooks (notebooks/demo_*.ipynb).

src/pipeline.run_staged loads and frees each model family per conversation
(it has to: the models do not fit together on JOJIE's 11 GB cards), so every
demo question would wait minutes for model loads. A DemoSession loads each
model the first time a turn needs it and keeps it, which needs a larger GPU;
the combined footprint is not measured yet, so the notebooks print free GPU
memory after every load. keep_loaded=False frees each model right after use
instead, for a smaller GPU, at the cost of reloading it on every turn.

Per turn it calls the same stage functions, in the same order and with the
same conversation rules, as run_staged:

  ask_audio  stages 1-7. On a conversation's first turn, audio LID picks the
             ASR checkpoint and text LID on the transcript decides final_lang
             (src/routing.py); later turns reuse both.
  ask_text   typed text replaces stages 1-2. On the first turn GlotLID (the
             stage 3 text LID) picks final_lang; later turns reuse it. Then
             stages 4-7.

lang= on a first turn skips LID and sets the language directly (as the
Streamlit GPU demo's language picker does); on a later turn it must match
the locked language, and reset() starts a new conversation. Follow-up
history is every earlier turn's English query and answer, as in run_staged.
A turn that raises changes nothing: no turn is recorded and the language is
not locked.

No silent fallbacks, same as run_staged: a failed translation raises; the
designed fallbacks are RAG's no-match message and TTS's text-only turn.
"""

from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import asr, audio, demo_ui, lid_audio, lid_text, mt, rag, routing, tts
from .pipeline import TTS_BACKENDS, TurnRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INIT_SET = REPO_ROOT / "data" / "init_set_ds6.json"


def gpu_memory() -> str:
    """"x GB free of y GB" for GPU 0, or "no GPU"."""
    import torch

    if not torch.cuda.is_available():
        return "no GPU"
    free_b, total_b = torch.cuda.mem_get_info()
    return f"{free_b / 1e9:.1f} GB free of {total_b / 1e9:.1f} GB"


def _unload(model) -> None:
    unload = getattr(model, "unload", None)  # GlotLID (fastText, CPU) has none
    if unload is not None:
        unload()


@dataclass
class DemoTurn:
    """One answered turn: the pipeline's TurnRecord plus what the demo shows."""

    record: TurnRecord
    input_mode: str  # "audio" or "text"
    typed_text: str | None = None
    lang_source: str = ""
    chunk_titles: dict = field(default_factory=dict)
    stage_s: dict = field(default_factory=dict)
    question_audio: np.ndarray | None = None  # 16 kHz, audio mode only

    @property
    def audio(self):
        """(array, sr) of the spoken answer, or None."""
        if self.record.audio_out is None:
            return None
        return self.record.audio_out, self.record.audio_out_sr

    def to_json_dict(self) -> dict:
        d = self.record.to_json_dict()
        d.update(input_mode=self.input_mode, typed_text=self.typed_text, lang_source=self.lang_source,
                 chunk_titles=self.chunk_titles, stage_s=self.stage_s)
        return d

    def show(self, title: str | None = None) -> None:
        question = (self.question_audio, audio.SAMPLE_RATE) if self.question_audio is not None else None
        demo_ui.show_turn(self.to_json_dict(), audio=self.audio, question_audio=question,
                          title=title or f"Turn {self.record.turn_idx + 1}")


class DemoSession:
    """One conversation at a time, models kept loaded between turns."""

    def __init__(self, *, tts_backend: str = "cpp", tts_options: dict | None = None, keep_loaded: bool = True,
                 init_set_path=DEFAULT_INIT_SET, rag_data_dir=None):
        if tts_backend not in TTS_BACKENDS:
            raise ValueError(f"Unknown tts_backend {tts_backend!r}, expected one of {TTS_BACKENDS}")
        if tts_backend == "hf" and tts_options:
            raise ValueError(f"tts_options apply to the cpp backend only, got {sorted(tts_options)}")
        self.tts_backend = tts_backend
        self.tts_options = dict(tts_options or {})
        self.keep_loaded = keep_loaded
        self.init_set = tts.load_init_set(init_set_path)
        self.rag_data_dir = rag_data_dir
        self.tts_info: dict | None = None
        self._models: dict[str, object] = {}
        self._n_speak = 0
        self.reset()

    # -- conversation state ---------------------------------------------------

    def reset(self) -> None:
        """Start a new conversation. Loaded models stay loaded."""
        self.turns: list[DemoTurn] = []
        self.history: list[rag.HistoryTurn] = []
        self.final_lang: str | None = None
        self.asr_lang: str | None = None

    def _check_lang(self, lang: str | None) -> None:
        if lang is not None and lang not in routing.KNOWN_LANGS:
            raise ValueError(f"lang must be one of {routing.KNOWN_LANGS}, got {lang!r}")
        if lang is not None and self.final_lang is not None and lang != self.final_lang:
            raise ValueError(f"This conversation is in {self.final_lang}; call reset() to start a new one in {lang}")

    # -- models -----------------------------------------------------------------

    @property
    def loaded(self) -> list[str]:
        return list(self._models)

    @contextmanager
    def _model(self, key: str, loader):
        model = self._models.get(key)
        if model is None:
            print(f"  loading {key} ...", flush=True)
            t0 = time.perf_counter()
            model = loader()
            print(f"  loaded {key} in {time.perf_counter() - t0:.0f} s; GPU {gpu_memory()}", flush=True)
            if self.keep_loaded:
                self._models[key] = model
        try:
            yield model
        finally:
            if not self.keep_loaded:
                _unload(model)

    def unload(self) -> None:
        """Free every loaded model (and stop the TTS server)."""
        for key in list(self._models):
            _unload(self._models.pop(key))
            print(f"  unloaded {key}")
        gc.collect()
        print(f"  GPU {gpu_memory()}")

    def _tts_module(self):
        if self.tts_backend == "cpp":
            from . import tts_cpp

            return tts_cpp
        return tts

    def _load_tts(self):
        if self.tts_backend == "cpp":
            bundle = self._tts_module().load(**self.tts_options)
            self.tts_info = dict(bundle.info)
            return bundle
        return tts.load()

    @staticmethod
    def _check_asr_checkpoint(lang: str) -> None:
        src = asr.CHECKPOINTS[lang]
        if not Path(src).is_dir():
            raise RuntimeError(
                f"ASR_{lang.upper()} is {src!r}, not a local folder. The Hub checkpoints do not load under "
                "transformers 4.57.3 (CHANGELOG.md, 2026-09-22): run scripts/patch_asr_tokenizers.py and set "
                "ASR_CEB, ASR_FIL and ASR_ENG before importing src.demo.")

    # -- turns ------------------------------------------------------------------

    def ask_text(self, text: str, *, lang: str | None = None, speak: bool = False) -> DemoTurn:
        """A typed question through stages 3 (first turn) and 4-7."""
        text = str(text).strip()
        if not text:
            raise ValueError("The question is empty")
        self._check_lang(lang)
        rec = TurnRecord(turn_idx=len(self.turns), audio_path=None)
        stage_s: dict = {}
        lock = None
        if self.final_lang is not None:
            final, source = self.final_lang, "locked from the first turn"
        elif lang is not None:
            final, source, lock = lang, "set by you", (lang, lang)
        else:
            with self._model("text LID (GlotLID)", lid_text.load) as model:
                t0 = time.perf_counter()
                result = model.predict(text)
                stage_s["text LID"] = time.perf_counter() - t0
            if result.lang is None:
                raise ValueError("GlotLID found no language in that text; pass lang='ceb', 'fil' or 'eng'")
            rec.text_lang, rec.text_lid_confidence = result.lang, result.confidence
            final, source, lock = result.lang, f"GlotLID{demo_ui._conf(result.confidence)}", (result.lang, result.lang)
        rec.final_lang = final
        turn = DemoTurn(rec, "text", typed_text=text, lang_source=source, stage_s=stage_s)
        return self._answer(turn, text, speak, lock)

    def ask_audio(self, path, *, lang: str | None = None, speak: bool = True) -> DemoTurn:
        """A spoken question through all seven stages."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._check_lang(lang)
        rec = TurnRecord(turn_idx=len(self.turns), audio_path=str(path))
        stage_s: dict = {}
        question = audio.load_audio(str(path))
        first = self.final_lang is None

        if not first:
            asr_lang = self.asr_lang
        elif lang is not None:
            asr_lang = lang
        else:
            with self._model("audio LID", lid_audio.load) as model:
                t0 = time.perf_counter()
                result = model.predict(question)
                stage_s["audio LID"] = time.perf_counter() - t0
            rec.pred_lang, rec.lid_confidence = result.lang, result.confidence
            asr_lang = result.lang

        self._check_asr_checkpoint(asr_lang)
        with self._model(f"ASR ({asr_lang})", lambda: asr.load(asr_lang)) as model:
            t0 = time.perf_counter()
            rec.transcript = model.transcribe(question)
            stage_s["ASR"] = time.perf_counter() - t0

        lock = None
        if not first:
            final, source = self.final_lang, "locked from the first turn"
        elif lang is not None:
            final, source, lock = lang, "set by you", (lang, lang)
        else:
            with self._model("text LID (GlotLID)", lid_text.load) as model:
                t0 = time.perf_counter()
                result = model.predict(rec.transcript)
                stage_s["text LID"] = time.perf_counter() - t0
            rec.text_lang, rec.text_lid_confidence = result.lang, result.confidence
            conv = routing.ConversationRouting.from_first_turn(rec.pred_lang, result.lang)
            rec.was_overridden = conv.was_overridden
            final, lock = conv.final_lang, (conv.final_lang, asr_lang)
            source = demo_ui.lang_source(rec.to_json_dict())
        rec.final_lang = final
        turn = DemoTurn(rec, "audio", lang_source=source, stage_s=stage_s, question_audio=question)
        return self._answer(turn, rec.transcript, speak, lock)

    def _answer(self, turn: DemoTurn, question: str, speak: bool, lock) -> DemoTurn:
        """Stages 4-7 for one turn; records the turn only if nothing raised."""
        rec, stage_s, lang = turn.record, turn.stage_s, turn.record.final_lang

        with self._model("MT (NLLB)", mt.load) as model:
            r = mt.translate_to_english(model, [mt.TurnText(id=0, text=question, lang=lang)])[0]
        rec.english_query, rec.mt_in_ok, rec.mt_in_failure_reason = r.text, r.ok, r.failure_reason
        stage_s["MT in"] = r.translate_time_sec
        if not r.ok:
            raise RuntimeError(f"stage 4 (MT in) failed: {r.failure_reason}; no designed fallback")

        with self._model("RAG", lambda: rag.load(data_dir=self.rag_data_dir)) as model:
            t0 = time.perf_counter()
            result = rag.answer_turn(model, rec.english_query, self.history)
            stage_s["RAG"] = time.perf_counter() - t0
        rec.resolved_query, rec.english_answer = result.resolved_query, result.answer
        rec.chunk_ids, rec.was_dependent = result.chunk_ids, result.was_dependent
        turn.chunk_titles = demo_ui.chunk_titles(result.chunk_ids, self.rag_data_dir or rag.DATA_DIR)

        with self._model("MT (NLLB)", mt.load) as model:
            r = mt.translate_from_english(model, [mt.TurnText(id=0, text=rec.english_answer, lang=lang)])[0]
        rec.native_answer, rec.mt_out_ok, rec.mt_out_failure_reason = r.text, r.ok, r.failure_reason
        stage_s["MT out"] = r.translate_time_sec
        if not r.ok:
            raise RuntimeError(f"stage 6 (MT out) failed: {r.failure_reason}; no designed fallback")

        if speak:
            result = self._synthesize(rec.turn_idx, rec.native_answer, lang)
            rec.audio_out, rec.audio_out_sr = result.audio, result.sr
            rec.tts_ok, rec.tts_failure_reason = result.ok, result.failure_reason
            rec.tts_backend, rec.tts_wall_sec, rec.tts_rtf = self.tts_backend, result.wall_sec, result.rtf
            if self.tts_backend == "cpp":
                rec.tts_segments, rec.tts_timeline = result.segments, result.timeline
            stage_s["TTS"] = result.wall_sec

        self.history += [rag.HistoryTurn("human", rec.english_query), rag.HistoryTurn("ai", rec.english_answer)]
        if lock is not None:
            self.final_lang, self.asr_lang = lock
        self.turns.append(turn)
        return turn

    def _synthesize(self, turn_id, text: str, lang: str):
        with self._model(f"TTS ({self.tts_backend})", self._load_tts) as model:
            return self._tts_module().synthesize_turn(model, turn_id, text, lang, self.init_set)

    def speak(self, text: str, lang: str):
        """Any text in the cloned voice, outside the conversation. Returns the
        backend's TTSResult (audio is None, with failure_reason, on a
        text-only result)."""
        if lang not in routing.KNOWN_LANGS:
            raise ValueError(f"lang must be one of {routing.KNOWN_LANGS}, got {lang!r}")
        self._n_speak += 1
        return self._synthesize(f"speak-{self._n_speak}", text, lang)

    # -- saving -----------------------------------------------------------------

    def save(self, folder) -> Path:
        """Write the conversation in run_conversation.py's layout (plus each
        spoken question as turn_N_question.wav) for the replay notebook.
        Refuses a folder that already holds a saved conversation."""
        import soundfile as sf

        if not self.turns:
            raise ValueError("Nothing to save: no turns yet")
        folder = Path(folder)
        if folder.is_dir() and any(folder.glob("turn_*.json")):
            raise FileExistsError(f"{folder} already holds a saved conversation; pick another folder")
        folder.mkdir(parents=True, exist_ok=True)
        for turn in self.turns:
            n = turn.record.turn_idx
            (folder / f"turn_{n}.json").write_text(
                json.dumps(turn.to_json_dict(), indent=2, ensure_ascii=False, default=float), encoding="utf-8")
            if turn.audio is not None:
                sf.write(str(folder / f"turn_{n}.wav"), *turn.audio)
            if turn.question_audio is not None:
                sf.write(str(folder / f"turn_{n}_question.wav"), turn.question_audio, audio.SAMPLE_RATE)
        manifest = {
            "source": "demo",
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_turns": len(self.turns),
            "final_lang": self.final_lang,
            "turn_files": [f"turn_{t.record.turn_idx}.json" for t in self.turns],
            "tts_backend": self.tts_backend,
            "tts": self.tts_info,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
        print(f"Saved {len(self.turns)} turn(s) to {folder}")
        return folder
