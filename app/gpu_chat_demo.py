"""Live GPU demo. GPU required; torch and transformers are imported directly
(unlike app/streamlit_app.py, which stays torch-free so the CPU image can
build without them).

Not part of docker/Dockerfile.cpu or docker/Dockerfile.gpu and not one of the
shipped artifacts: a standalone Streamlit app for showing the pipeline live on
a large-VRAM GPU host (it was run on a 40GB A100), started with
`streamlit run app/gpu_chat_demo.py` and torn down after use.

Two input modes:
  - Text: stages 4-6 only (MT in -> RAG -> MT out), typed text in, text
    answer back.
  - Audio: stages 2 + 4-7 (ASR, MT in, RAG, MT out, TTS), an audio question
    in (either a pre-staged file from a dropdown, or a live microphone
    recording via st.audio_input), transcript + text answer + synthesized
    answer audio back. No audio LID (stage 1): the language is picked
    manually in a language selector, the same simplification text mode makes
    for GlotLID (optional auto-detect there, manual only here). This also
    keeps models/deeper_50chunks.pt off the demo host.

The dropdown of known-good pre-staged files is the lower-risk default for a
live audience (no room noise, no mic permission prompts, no silent takes);
the microphone is there when that risk is acceptable. Either way the compute
(ASR, translation, retrieval, generation, translation back, synthesis) runs
for real when the button is clicked. Point DEMO_INPUT_AUDIO_DIR at a folder
of audio files and the dropdown lists whatever is in it.

Resident where it matters, lazy where it doesn't: mt.load() and rag.load()
load once at first use and stay resident (st.cache_resource). asr.load(lang)
is cached per language (a language never selected is never loaded), and
tts.load() is cached once, on first audio-mode use. The combined worst case
(NLLB + RAG + all 3 ASR checkpoints + TTS resident at once) is unmeasured;
check `nvidia-smi` after exercising all three languages once.

Environment:
  HF_TOKEN            required: the ASR checkpoints are private.
  ASR_CEB/FIL/ENG     required: point at scripts/patch_asr_tokenizers.py's
                      output; that script must be run first (it produces
                      local checkpoint copies; the patch is not in the HF
                      repos). See CHANGELOG.md, 2026-09-22.
  TTS_REF_DIR         required for audio mode: the three locked reference
                      clips.
  DEMO_INPUT_AUDIO_DIR required for audio mode: a folder of question audio
                      files for the dropdown.
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import asr, audio, lid_text, mt, rag, tts  # noqa: E402

LANG_NAMES = {"ceb": "Cebuano", "fil": "Filipino", "eng": "English"}

DEFAULT_INIT_SET_PATH = Path(__file__).resolve().parents[1] / "data" / "init_set_ds6.json"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}


# --- Resident / lazily-cached model loaders ---------------------------------
@st.cache_resource(show_spinner="Loading GlotLID (text language ID)...")
def load_text_lid() -> lid_text.TextLID:
    return lid_text.load()


@st.cache_resource(show_spinner="Loading NLLB (translation, ~6.7GB, stays resident)...")
def load_mt() -> mt.MTBundle:
    return mt.load()


@st.cache_resource(show_spinner="Loading RAG (retrieval + generation, stays resident)...")
def load_rag() -> rag.RAGBundle:
    return rag.load()


@st.cache_resource(show_spinner="Loading ASR checkpoint (first use for this language only)...")
def load_asr(lang: str) -> asr.ASRBundle:
    return asr.load(lang)


@st.cache_resource(show_spinner="Loading Qwen3-TTS (first audio-mode use, loads reference clips)...")
def load_tts() -> tts.TTSBundle:
    return tts.load()


@st.cache_resource(show_spinner=False)
def load_init_set() -> set[str]:
    return tts.load_init_set(DEFAULT_INIT_SET_PATH)


def print_environment() -> None:
    import torch
    import transformers

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_note = f"{torch.cuda.get_device_name(0)}, {vram_gb:.0f}GB"
    else:
        gpu_note = "none (CPU only)"
    st.caption(f"torch {torch.__version__} | transformers {transformers.__version__} | GPU {gpu_note}")


# --- Text mode: stages 4-6 ---------------------------------------------------
def run_turn(text: str, lang: str, history: list[rag.HistoryTurn]) -> dict:
    """One question through stages 4-6, using the resident models loaded
    once at first use. Each stage's actual output prints inline (expanded
    status, not just a timing label) as soon as that stage finishes, to
    show how the pipeline works, not just its final answer. Returns a dict of everything worth showing; raises only if a
    stage itself raises unexpectedly (caught by the caller)."""
    result: dict = {"native_query": text, "final_lang": lang}
    mt_bundle = load_mt()
    rag_bundle = load_rag()
    lang_label = LANG_NAMES.get(lang, lang)

    with st.status("Stage 4: translating to English...", expanded=True) as status:
        mt_in = mt.translate_to_english(mt_bundle, [mt.TurnText(id=0, text=text, lang=lang)])[0]
        if not mt_in.ok:
            status.update(label="Stage 4 failed", state="error")
            raise RuntimeError(f"MT in failed: {mt_in.failure_reason}")
        result["english_query"] = mt_in.text
        st.write(f"**English query:** {mt_in.text}")
        status.update(
            label=f"Stage 4 done: translated to English ({mt_in.translate_time_sec:.1f}s)",
            state="complete", expanded=True,
        )

    with st.status("Stage 5: retrieving + generating answer...", expanded=True) as status:
        rag_result = rag.answer_turn(rag_bundle, mt_in.text, history)
        result["resolved_query"] = rag_result.resolved_query
        result["was_dependent"] = rag_result.was_dependent
        result["english_answer"] = rag_result.answer
        result["chunk_ids"] = rag_result.chunk_ids
        if rag_result.was_dependent:
            st.write(f"**Resolved follow-up to:** {rag_result.resolved_query}")
        st.write(f"**English answer:** {rag_result.answer}")
        st.caption("Chunks used: " + (", ".join(rag_result.chunk_ids) if rag_result.chunk_ids else "none"))
        status.update(label="Stage 5 done: answer generated", state="complete", expanded=True)

    with st.status("Stage 6: translating answer back...", expanded=True) as status:
        mt_out = mt.translate_from_english(
            mt_bundle, [mt.TurnText(id=0, text=rag_result.answer, lang=lang)]
        )[0]
        if not mt_out.ok:
            status.update(label="Stage 6 failed", state="error")
            raise RuntimeError(f"MT out failed: {mt_out.failure_reason}")
        result["native_answer"] = mt_out.text
        if lang != "eng":
            st.write(f"**{lang_label} answer:** {mt_out.text}")
        status.update(
            label=f"Stage 6 done: translated back ({mt_out.translate_time_sec:.1f}s)",
            state="complete", expanded=True,
        )

    return result


# --- Audio mode: stages 2 + 4-7 ----------------------------------------------
def discover_input_audio_files() -> dict[str, Path]:
    """filename -> path, for every audio file under DEMO_INPUT_AUDIO_DIR.
    Empty dict (not an error) when unset or empty -- the audio-mode UI
    explains what to do instead of erroring."""
    import os

    dir_str = os.environ.get("DEMO_INPUT_AUDIO_DIR")
    if not dir_str:
        return {}
    d = Path(dir_str)
    if not d.is_dir():
        return {}
    return {p.name: p for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_EXTS}


def _wav_bytes(audio_np, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV")
    return buf.getvalue()


def run_audio_turn(audio_path: Path, lang: str, history: list[rag.HistoryTurn]) -> dict:
    """One question from a pre-staged audio file through stages 2 + 4-7.
    Reuses run_turn for the MT/RAG/MT chain rather than duplicating it --
    its own stage-by-stage inline output (English query, answer, native
    answer) all appears here too, well before this function reaches TTS,
    since TTS is by far the slowest stage. Text results are never gated
    behind audio synthesis finishing."""
    with st.status("Stage 2: transcribing (ASR)...", expanded=True) as status:
        aud = audio.load_audio(str(audio_path))
        transcript = load_asr(lang).transcribe(aud)
        st.write(f"**Transcript:** {transcript}")
        status.update(label="Stage 2 done: transcribed", state="complete", expanded=True)

    result = run_turn(transcript, lang, history)
    result["transcript"] = transcript

    with st.status("Stage 7: synthesizing answer audio (TTS)...", expanded=True) as status:
        tts_result = tts.synthesize_turn(load_tts(), 0, result["native_answer"], lang, load_init_set())
        result["tts_ok"] = tts_result.ok
        result["tts_failure_reason"] = tts_result.failure_reason
        result["audio_out"] = tts_result.audio
        result["audio_out_sr"] = tts_result.sr
        # The frontend raises on some inputs by design (see src/tts.py's
        # module docstring); synthesize_turn already turns that into
        # ok=False rather than crashing -- the text answer above is
        # unaffected either way.
        status.update(
            label="Stage 7 done: answer audio ready" if tts_result.ok else f"Stage 7 failed: {tts_result.failure_reason}",
            state="complete" if tts_result.ok else "error",
            expanded=True,
        )

    return result


def main() -> None:
    st.set_page_config(page_title="LingkodAI (live)", page_icon="🇵🇭", layout="wide")
    st.title("LingkodAI -- live demo")
    st.caption(
        "Live demo on a GPU host. Text or "
        "audio in, real translation + retrieval + generation (+ synthesis for "
        "audio) back. Models load once at first use and stay resident, so "
        "only the first use of each language/mode is slow."
    )
    print_environment()

    if "history" not in st.session_state:
        st.session_state.history = []  # list[rag.HistoryTurn], English only
    if "display_history" not in st.session_state:
        st.session_state.display_history = []  # list[dict], for rendering

    with st.sidebar:
        if st.button("New conversation"):
            st.session_state.history = []
            st.session_state.display_history = []
            st.rerun()
        if st.button("Warm up text mode now"):
            load_mt()
            load_rag()
            load_text_lid()
            st.success("Text mode loaded and resident.")
        st.caption(
            "ASR/TTS load lazily on first audio-mode use per language -- "
            "click through each dropdown option once before presenting to "
            "warm them up ahead of time."
        )

    for turn in st.session_state.display_history:
        with st.chat_message("user"):
            st.write(turn["native_query"])
        with st.chat_message("assistant"):
            lang_label = LANG_NAMES.get(turn["final_lang"], turn["final_lang"])
            st.markdown(f"**English:** {turn['english_answer']}")
            if turn["final_lang"] != "eng":
                st.markdown(f"**{lang_label}:** {turn['native_answer']}")
            if turn["chunk_ids"]:
                st.caption("Chunks used: " + ", ".join(turn["chunk_ids"]))
            else:
                st.caption("No matching PRC service found.")
            if turn.get("audio_out") is not None:
                st.audio(_wav_bytes(turn["audio_out"], turn["audio_out_sr"]))
            elif turn.get("tts_ok") is False:
                st.caption(f"No answer audio: {turn['tts_failure_reason']}")

    mode = st.radio("Input mode", ["Text", "Audio"], horizontal=True)

    if mode == "Text":
        lang_choice = st.radio(
            "Language", ["Auto-detect (GlotLID)", "Cebuano", "Filipino", "English"], horizontal=True
        )
        query = st.chat_input("Type your question here...")

        if query:
            with st.chat_message("user"):
                st.write(query)

            if lang_choice == "Auto-detect (GlotLID)":
                lid_result = load_text_lid().predict(query)
                if lid_result.lang is None:
                    st.error("GlotLID could not identify a language for this text.")
                    st.stop()
                lang = lid_result.lang
                st.caption(f"Detected: {LANG_NAMES[lang]} (confidence {lid_result.confidence:.3f})")
            else:
                lang = {"Cebuano": "ceb", "Filipino": "fil", "English": "eng"}[lang_choice]

            with st.chat_message("assistant"):
                t0 = time.time()
                try:
                    turn = run_turn(query, lang, st.session_state.history)
                except Exception as exc:
                    st.error(f"This turn failed: {exc}")
                    st.stop()
                wall = time.time() - t0

                lang_label = LANG_NAMES.get(lang, lang)
                st.markdown(f"**English:** {turn['english_answer']}")
                if lang != "eng":
                    st.markdown(f"**{lang_label}:** {turn['native_answer']}")
                if turn["chunk_ids"]:
                    st.caption("Chunks used: " + ", ".join(turn["chunk_ids"]))
                else:
                    st.caption("No matching PRC service found.")
                st.caption(f"Total: {wall:.1f}s")

            st.session_state.history.append(rag.HistoryTurn("human", turn["english_query"]))
            st.session_state.history.append(rag.HistoryTurn("ai", turn["english_answer"]))
            st.session_state.display_history.append(turn)

    else:  # Audio mode
        lang_choice = st.radio("Language", ["Cebuano", "Filipino", "English"], horizontal=True, key="audio_lang")
        audio_source = st.radio(
            "Audio source", ["Pre-recorded file", "Record with microphone"], horizontal=True, key="audio_source"
        )

        audio_path: Path | None = None
        audio_label: str | None = None

        if audio_source == "Pre-recorded file":
            files = discover_input_audio_files()
            if not files:
                st.info(
                    "No input audio files found. Set DEMO_INPUT_AUDIO_DIR to a folder "
                    "of audio files (a mounted Google Drive folder works well) and "
                    "restart this app."
                )
            else:
                filename = st.selectbox("Question audio", sorted(files))
                audio_path = files[filename]
                audio_label = filename
        else:
            recorded = st.audio_input("Record your question", key="mic_recording")
            if recorded is not None:
                # st.audio_input always returns WAV-encoded bytes.
                tmp_dir = Path(tempfile.gettempdir()) / "lingkodai_mic_recordings"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                audio_path = tmp_dir / f"recording_{int(time.time())}.wav"
                audio_path.write_bytes(recorded.getvalue())
                audio_label = "microphone recording"

        if audio_path is not None and st.button("Ask", type="primary"):
            lang = {"Cebuano": "ceb", "Filipino": "fil", "English": "eng"}[lang_choice]

            with st.chat_message("user"):
                st.audio(str(audio_path))
                st.caption(audio_label)

            with st.chat_message("assistant"):
                t0 = time.time()
                try:
                    turn = run_audio_turn(audio_path, lang, st.session_state.history)
                except Exception as exc:
                    st.error(f"This turn failed: {exc}")
                    st.stop()
                wall = time.time() - t0

                lang_label = LANG_NAMES.get(lang, lang)
                st.caption(f"Transcript: {turn['transcript']}")
                st.markdown(f"**English:** {turn['english_answer']}")
                if lang != "eng":
                    st.markdown(f"**{lang_label}:** {turn['native_answer']}")
                if turn["chunk_ids"]:
                    st.caption("Chunks used: " + ", ".join(turn["chunk_ids"]))
                else:
                    st.caption("No matching PRC service found.")
                if turn.get("audio_out") is not None:
                    st.audio(_wav_bytes(turn["audio_out"], turn["audio_out_sr"]))
                elif not turn.get("tts_ok"):
                    st.caption(f"No answer audio: {turn['tts_failure_reason']}")
                st.caption(f"Total: {wall:.1f}s")

            turn["native_query"] = turn["transcript"]
            st.session_state.history.append(rag.HistoryTurn("human", turn["english_query"]))
            st.session_state.history.append(rag.HistoryTurn("ai", turn["english_answer"]))
            st.session_state.display_history.append(turn)


if __name__ == "__main__":
    main()
