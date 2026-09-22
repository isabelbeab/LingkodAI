"""Presentation-only live demo. GPU required, torch/transformers imported
directly (unlike app/streamlit_app.py, which must stay torch-free so the
CPU/Fly image can build without them).

NOT part of docker/Dockerfile.cpu or docker/Dockerfile.gpu, not referenced by
fly.toml. Run directly on a real GPU host (Colab A100) with `streamlit run
app/gpu_chat_demo.py`, tunneled out with cloudflared (see
reference/colab_streamlit_launcher.ipynb for the tunnel pattern this
mirrors), torn down after use. Never describe this as a fourth shipped
artifact.

Standalone Streamlit app -- no Fly frontend, no GPU_BACKEND_URL. That split
architecture was tried and abandoned the same day it was built (Fly kept
restarting; not worth chasing down under deadline pressure) in favor of
this simpler, single-process design: one Streamlit app IS the whole demo,
same as the original version of this file.

Two input modes:
  - Text: stages 4-6 only (MT in -> RAG -> MT out), typed text in, text
    answer back. This is the original version of this file, unchanged.
  - Audio: stages 2 + 4-7 (ASR, MT in, RAG, MT out, TTS), a pre-staged
    audio file selected from a dropdown in, transcript + text answer +
    synthesized answer audio back. No audio LID (stage 1) -- language is
    picked manually in the dropdown's language selector, same
    simplification the text mode already makes for GlotLID (there it's
    optional auto-detect; here there is no live audio to run LID on
    ahead of time in a way that would be safe to skip re-verifying, so
    it is manual only). This also avoids needing to transfer
    models/deeper_50chunks.pt (Bea's file) to this host.

The audio dropdown is deliberately not a live microphone recorder
(st.audio_input): a dropdown of known-good files is lower-risk for a live
presentation (no room noise, no mic permission prompts, no surprise silent
takes) while the actual compute -- ASR, translation, retrieval, generation,
translation back, synthesis -- still runs for real, live, when the button
is clicked. Point DEMO_INPUT_AUDIO_DIR at a folder of audio files (a
mounted Google Drive folder works well) and the dropdown lists whatever is
in it -- add a new demo question by adding a file to Drive, no code change.

Resident where it matters, lazy where it doesn't: mt.load() and rag.load()
load once at first use and stay resident (st.cache_resource), matching the
already-verified fix for the latency that broke the very first JOJIE live
attempt. asr.load(lang) is cached per language (lazy: a language never
selected in this session is never loaded), and tts.load() is cached once,
lazy on first audio-mode use. On a 40GB A100 the combined worst case (NLLB
+ RAG + all 3 ASR checkpoints + TTS all resident at once) is the one
genuinely unmeasured number in this design -- check `nvidia-smi` after
exercising all three languages once, same discipline as the JOJIE OOM
investigation earlier this project.

New environment requirements beyond the text-only version:
  HF_TOKEN            required -- the ASR checkpoints are private.
  ASR_CEB/FIL/ENG     required -- point at scripts/patch_asr_tokenizers.py's
                      output; that script must be run first (produces local
                      checkpoint copies, the patch is not baked into the HF
                      repos). See DECISIONS.md #9.
  TTS_REF_DIR         required for audio mode -- the three locked reference
                      clips, staged via Drive (HK's choice; download from
                      JOJIE's TTS_REF_DIR, upload to Drive, mount it here).
  DEMO_INPUT_AUDIO_DIR required for audio mode -- a folder of question audio
                      files for the dropdown, same Drive-staging idea.
"""

from __future__ import annotations

import io
import sys
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
    once at first use. Returns a dict of everything worth showing; raises
    only if a stage itself raises unexpectedly (caught by the caller)."""
    result: dict = {"native_query": text, "final_lang": lang}
    mt_bundle = load_mt()
    rag_bundle = load_rag()

    with st.status("Translating to English...", expanded=False) as status:
        mt_in = mt.translate_to_english(mt_bundle, [mt.TurnText(id=0, text=text, lang=lang)])[0]
        if not mt_in.ok:
            status.update(label="MT in failed", state="error")
            raise RuntimeError(f"MT in failed: {mt_in.failure_reason}")
        result["english_query"] = mt_in.text
        status.update(label=f"Translated to English ({mt_in.translate_time_sec:.1f}s)", state="complete")

    with st.status("Retrieving and generating answer...", expanded=False) as status:
        rag_result = rag.answer_turn(rag_bundle, mt_in.text, history)
        result["resolved_query"] = rag_result.resolved_query
        result["was_dependent"] = rag_result.was_dependent
        result["english_answer"] = rag_result.answer
        result["chunk_ids"] = rag_result.chunk_ids
        status.update(label="Answer generated", state="complete")

    with st.status("Translating answer back...", expanded=False) as status:
        mt_out = mt.translate_from_english(
            mt_bundle, [mt.TurnText(id=0, text=rag_result.answer, lang=lang)]
        )[0]
        if not mt_out.ok:
            status.update(label="MT out failed", state="error")
            raise RuntimeError(f"MT out failed: {mt_out.failure_reason}")
        result["native_answer"] = mt_out.text
        status.update(label=f"Translated back ({mt_out.translate_time_sec:.1f}s)", state="complete")

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
    Reuses run_turn for the MT/RAG/MT chain rather than duplicating it."""
    with st.status("Transcribing...", expanded=False) as status:
        aud = audio.load_audio(str(audio_path))
        transcript = load_asr(lang).transcribe(aud)
        status.update(label=f"Transcribed ({len(transcript)} chars)", state="complete")

    result = run_turn(transcript, lang, history)
    result["transcript"] = transcript

    with st.status("Synthesizing answer audio...", expanded=False) as status:
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
            label="Answer audio ready" if tts_result.ok else f"TTS failed: {tts_result.failure_reason}",
            state="complete" if tts_result.ok else "error",
        )

    return result


def main() -> None:
    st.set_page_config(page_title="LingkodAI (live)", page_icon="🇵🇭", layout="wide")
    st.title("LingkodAI -- live demo")
    st.caption(
        "Presentation-only live demo, standalone on a real GPU host. Text or "
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
        files = discover_input_audio_files()

        if not files:
            st.info(
                "No input audio files found. Set DEMO_INPUT_AUDIO_DIR to a folder "
                "of audio files (a mounted Google Drive folder works well) and "
                "restart this app."
            )
        else:
            filename = st.selectbox("Question audio", sorted(files))
            if st.button("Ask", type="primary"):
                lang = {"Cebuano": "ceb", "Filipino": "fil", "English": "eng"}[lang_choice]
                audio_path = files[filename]

                with st.chat_message("user"):
                    st.audio(str(audio_path))
                    st.caption(filename)

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
