"""Conversation state and the staged pipeline runner.

Ties stages 1-7 together for one scripted, multi-turn conversation, the way
the evaluated DS6 run was produced. See CLAUDE.md's "Execution modes":
`staged` (this module's only runner so far) is phase-major over the whole
conversation -- each phase loads its models once, processes every turn, then
frees them, because the models do not fit together on an 11 GB GPU. Phase
order:

  1. audio LID (turn 1 only) + ASR (every turn) + text LID (turn 1 only) +
     routing, all turns
  2. MT in, all turns
  3. RAG, turn by turn in order (follow-up rewriting needs earlier English
     answers)
  4. MT out, all turns
  5. TTS, all turns

`resident` mode (everything loaded once, per-turn API) is a stretch goal per
CLAUDE.md and is not implemented here -- build it only after `staged` passes
the golden check.

No silent fallbacks. The only designed fallbacks anywhere in this pipeline
are RAG's no-match message (src/rag.py) and TTS's text-only turn
(src/tts.py's synthesize_turn). A failed MT stage has no designed fallback,
so run_staged raises rather than guessing what to feed the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import asr, audio, lid_audio, lid_text, mt, rag, routing, tts


def _print_gpu_mem(label: str) -> None:
    """Free/total CUDA memory at a phase boundary, for diagnosing OOMs across
    phases without guessing -- see DECISIONS.md's TTS OOM investigation."""
    if not torch.cuda.is_available():
        return
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"  [gpu] {label}: {free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total")


@dataclass
class TurnRecord:
    """Everything known about one conversation turn, filled in phase by
    phase as run_staged proceeds. Fields stay None until their stage runs."""

    turn_idx: int
    audio_path: str

    # Stage 1 -- audio LID. Only turn 0 gets real values; every other turn
    # inherits the conversation's locked final_lang instead.
    pred_lang: str | None = None
    lid_confidence: float | None = None

    # Stage 2 -- ASR, every turn.
    transcript: str | None = None

    # Stage 3 -- text LID (turn 0 only) and routing (locked from turn 0,
    # copied onto every turn for convenience).
    text_lang: str | None = None
    text_lid_confidence: float | None = None
    final_lang: str | None = None
    was_overridden: bool | None = None

    # Stage 4 -- MT in, every turn.
    english_query: str | None = None
    mt_in_ok: bool | None = None
    mt_in_failure_reason: str | None = None

    # Stage 5 -- RAG, every turn.
    resolved_query: str | None = None
    english_answer: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    was_dependent: bool | None = None

    # Stage 6 -- MT out, every turn.
    native_answer: str | None = None
    mt_out_ok: bool | None = None
    mt_out_failure_reason: str | None = None

    # Stage 7 -- TTS, every turn. audio_out is None whenever tts_ok is False
    # (the frontend raised by design) -- the turn still has its native_answer.
    audio_out: np.ndarray | None = None
    audio_out_sr: int | None = None
    tts_ok: bool | None = None
    tts_failure_reason: str | None = None

    def to_json_dict(self) -> dict:
        """Everything except the raw audio array, which the caller writes to
        its own WAV file instead of embedding in JSON."""
        d = {k: v for k, v in self.__dict__.items() if k not in ("audio_out",)}
        d["has_audio"] = self.audio_out is not None
        return d


@dataclass
class ConversationResult:
    """The full output of one run_staged call."""

    turns: list[TurnRecord]
    routing: routing.ConversationRouting


def run_staged(
    audio_paths: list[str],
    init_set: set[str],
    *,
    rag_data_dir=None,
) -> ConversationResult:
    """Run one scripted conversation through all seven stages, phase-major.

    init_set is the frozen TTS initialism set (see src/tts.load_init_set);
    never derive it from the conversation's own answers.
    """
    if not audio_paths:
        raise ValueError("audio_paths is empty; a conversation needs at least one turn")

    turns = [TurnRecord(turn_idx=i, audio_path=p) for i, p in enumerate(audio_paths)]
    loaded_audio = [audio.load_audio(t.audio_path) for t in turns]

    # --- Phase 1: audio LID (turn 0 only) + ASR (every turn) + text LID
    # (turn 0 only) + routing -----------------------------------------
    print(f"[phase 1/5] audio LID, ASR, text LID, routing -- {len(turns)} turn(s)")
    _print_gpu_mem("start of phase 1")

    lid_model = lid_audio.load()
    try:
        lid_result = lid_model.predict(loaded_audio[0])
    finally:
        lid_model.unload()
    turns[0].pred_lang = lid_result.lang
    turns[0].lid_confidence = lid_result.confidence

    asr_bundle = asr.load(lid_result.lang)
    try:
        for t, aud in zip(turns, loaded_audio):
            t.transcript = asr_bundle.transcribe(aud)
    finally:
        asr_bundle.unload()

    text_lid_model = lid_text.load()
    text_lid_result = text_lid_model.predict(turns[0].transcript)
    turns[0].text_lang = text_lid_result.lang
    turns[0].text_lid_confidence = text_lid_result.confidence

    conv_routing = routing.ConversationRouting.from_first_turn(lid_result.lang, text_lid_result.lang)
    for t in turns:
        t.final_lang = conv_routing.final_lang
    turns[0].was_overridden = conv_routing.was_overridden

    # --- Phase 2: MT in, all turns -------------------------------------
    print("[phase 2/5] MT in (native -> English)")
    _print_gpu_mem("start of phase 2")

    mt_bundle = mt.load()
    try:
        items = [mt.TurnText(id=t.turn_idx, text=t.transcript, lang=conv_routing.final_lang) for t in turns]
        results = mt.translate_to_english(mt_bundle, items)
    finally:
        mt_bundle.unload()
    for t in turns:
        r = results[t.turn_idx]
        t.english_query = r.text
        t.mt_in_ok = r.ok
        t.mt_in_failure_reason = r.failure_reason

    # --- Phase 3: RAG, turn by turn in order ----------------------------
    print("[phase 3/5] RAG")
    _print_gpu_mem("start of phase 3")

    rag_bundle = rag.load(data_dir=rag_data_dir)
    try:
        history: list[rag.HistoryTurn] = []
        for t in turns:
            if not t.mt_in_ok:
                raise RuntimeError(
                    f"turn {t.turn_idx}: stage 4 (MT in) failed ({t.mt_in_failure_reason}); "
                    "no designed fallback for a failed translation, see module docstring"
                )
            result = rag.answer_turn(rag_bundle, t.english_query, history)
            t.resolved_query = result.resolved_query
            t.english_answer = result.answer
            t.chunk_ids = result.chunk_ids
            t.was_dependent = result.was_dependent
            history.append(rag.HistoryTurn("human", t.english_query))
            history.append(rag.HistoryTurn("ai", t.english_answer))
    finally:
        rag_bundle.unload()

    # --- Phase 4: MT out, all turns --------------------------------------
    print("[phase 4/5] MT out (English -> native)")
    _print_gpu_mem("start of phase 4")

    mt_bundle = mt.load()
    try:
        items = [mt.TurnText(id=t.turn_idx, text=t.english_answer, lang=conv_routing.final_lang) for t in turns]
        results = mt.translate_from_english(mt_bundle, items)
    finally:
        mt_bundle.unload()
    for t in turns:
        r = results[t.turn_idx]
        t.native_answer = r.text
        t.mt_out_ok = r.ok
        t.mt_out_failure_reason = r.failure_reason

    # --- Phase 5: TTS, all turns ------------------------------------------
    print("[phase 5/5] TTS")
    _print_gpu_mem("start of phase 5")

    tts_bundle = tts.load()
    try:
        for t in turns:
            if not t.mt_out_ok:
                raise RuntimeError(
                    f"turn {t.turn_idx}: stage 6 (MT out) failed ({t.mt_out_failure_reason}); "
                    "no designed fallback for a failed translation, see module docstring"
                )
            result = tts.synthesize_turn(
                tts_bundle, t.turn_idx, t.native_answer, conv_routing.final_lang, init_set
            )
            t.audio_out = result.audio
            t.audio_out_sr = result.sr
            t.tts_ok = result.ok
            t.tts_failure_reason = result.failure_reason
    finally:
        tts_bundle.unload()

    return ConversationResult(turns=turns, routing=conv_routing)
