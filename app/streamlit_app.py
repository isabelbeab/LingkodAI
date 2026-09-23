"""LingkodAI CPU demo (Streamlit). No torch imports anywhere in this file or
anything it imports -- this is what makes the CPU image possible (target
size around 200 MB, no CUDA, no model weights baked in). See
docker/Dockerfile.cpu and README.md's "What ships" #2.

What runs live in here, on CPU:
  - GlotLID text language identification (src/lid_text.py -- fastText, CPU
    is instant)
  - the TTS text frontend (src/tts_frontend/ -- verbalizers, initialism
    spell-out, CEB segmentation and pause planning), previewing the text
    LingkodAI would hand to the voice model, without actually running it

What does NOT run in here directly: audio LID, ASR, NLLB, the RAG LLM,
Qwen3-TTS. Those need a GPU. This page browses pre-rendered audio from real
end-to-end conversations recorded on JOJIE (demo_audio/), previews the text
frontend live, and -- when GPU_BACKEND_URL is set -- calls a real GPU
backend (app/gpu_backend_api.py, run separately on a real GPU host, e.g.
Colab) over HTTP for a live pipeline. No torch import here either way: the
call is a plain requests.post(), same as any other HTTP client. This Fly
app is still the CPU-only frontend, the GPU work happens elsewhere
entirely.

demo_audio/<conversation_name>/ is exactly one scripts/run_conversation.py
output directory (manifest.json + turn_N.json + turn_N.wav), copied in
after being generated for real on JOJIE -- no separate manifest format was
invented for this page.

The demo plays synthesized speech in a cloned PLD research speaker's voice
(CC-BY-NC, research-only licence), which is why the page is password-gated
and discloses the voice below (see README.md's Fly deployment section).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
import streamlit as st

# The vendored TTS frontend modules import each other by top-level name
# (import shared_domain_frontend, import ceb_respell_lab, ...), so their
# directory goes on sys.path rather than rewriting their imports -- same
# pattern as src/tts.py, but this file must not import src.tts itself: that
# module imports torch at module level (needed for the real GPU synthesis
# it does), which would break the whole point of this being a CPU image.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TTS_FRONTEND_DIR = _REPO_ROOT / "src" / "tts_frontend"
if str(_TTS_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_TTS_FRONTEND_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ceb_full_pipeline  # noqa: E402
from eng_prc_digit_verbalizer import verbalize_eng_prc  # noqa: E402
from fil_prc_digit_verbalizer import verbalize_fil_prc  # noqa: E402
from lingkod_tts_utils import prep_tts_text  # noqa: E402

from src import lid_text  # noqa: E402  -- pure Python + fasttext, no torch

DEMO_AUDIO_DIR = _REPO_ROOT / "demo_audio"
INIT_SET_PATH = _REPO_ROOT / "data" / "init_set_ds6.json"

LANG_NAMES = {"ceb": "Cebuano", "fil": "Filipino", "eng": "English"}

VOICE_DISCLOSURE = (
    "Synthesized audio on this page uses a cloned voice from a Philippine "
    "Languages Database research speaker, under a CC-BY-NC, research-only "
    "licence. It is not a real PRC official."
)


# --- Frozen initialism set ---------------------------------------------------
# Duplicated (not imported) from src/tts.load_init_set: that function is
# two lines of json.load, but it lives in src/tts.py, which imports torch at
# module level -- importing it here for two lines would defeat the point of
# this file staying torch-free. Never derive this set at runtime; see
# scripts/build_init_set.py's docstring for why per-answer derivation
# silently breaks acronym spelling.
@st.cache_resource(show_spinner=False)
def load_init_set() -> set[str]:
    if not INIT_SET_PATH.exists():
        raise FileNotFoundError(
            f"frozen initialism set not found: {INIT_SET_PATH}. "
            "Run scripts/build_init_set.py or check the image build copied data/."
        )
    with open(INIT_SET_PATH, encoding="utf-8") as fh:
        return set(json.load(fh)["tokens"])


# --- Text LID -----------------------------------------------------------
@st.cache_resource(show_spinner="Loading GlotLID (first use downloads ~1.2 GB)...")
def load_text_lid() -> lid_text.TextLID:
    return lid_text.load()


# --- Demo conversations ---------------------------------------------------
def discover_demo_conversations() -> dict[str, Path]:
    """conversation name -> its directory, for every demo_audio/*/manifest.json
    found. Empty dict (not an error) when none exist yet -- demo_audio/ is
    gitignored, so a fresh clone or a laptop that never received the
    recorded conversations legitimately has none."""
    if not DEMO_AUDIO_DIR.is_dir():
        return {}
    found = {}
    for manifest_path in sorted(DEMO_AUDIO_DIR.glob("*/manifest.json")):
        found[manifest_path.parent.name] = manifest_path.parent
    return found


def load_conversation(conv_dir: Path) -> tuple[dict, list[dict]]:
    with open(conv_dir / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    turns = []
    for turn_file in manifest["turn_files"]:
        with open(conv_dir / turn_file, encoding="utf-8") as fh:
            turns.append(json.load(fh))
    return manifest, turns


def render_demo_tab() -> None:
    st.caption(VOICE_DISCLOSURE)
    conversations = discover_demo_conversations()

    if not conversations:
        st.info(
            "No pre-rendered conversations are bundled with this deploy yet. "
            "These come from real scripts/run_conversation.py runs on JOJIE, "
            "copied into demo_audio/ before the image is built."
        )
        return

    name = st.selectbox("Conversation", sorted(conversations))
    manifest, turns = load_conversation(conversations[name])

    lang_label = LANG_NAMES.get(manifest["final_lang"], manifest["final_lang"])
    st.write(
        f"**Language:** {lang_label}"
        + (" (audio LID and text LID disagreed on turn 1)" if manifest["was_overridden"] else "")
    )

    for turn in turns:
        with st.chat_message("user"):
            st.write(turn["transcript"])
            st.caption(f"turn {turn['turn_idx']}")
        with st.chat_message("assistant"):
            st.markdown(f"**English:** {turn['english_answer']}")
            if manifest["final_lang"] != "eng":
                st.markdown(f"**{lang_label}:** {turn['native_answer']}")
            if turn["has_audio"]:
                wav_path = conversations[name] / f"turn_{turn['turn_idx']}.wav"
                if wav_path.exists():
                    st.audio(str(wav_path))
                else:
                    st.warning(f"manifest says turn {turn['turn_idx']} has audio, but {wav_path.name} is missing.")
            elif turn["tts_ok"] is False:
                st.caption(f"No audio for this turn: {turn['tts_failure_reason']}")


# --- Live text-frontend preview -------------------------------------------
def render_frontend_tab() -> None:
    st.caption(
        "Runs the real text-preprocessing stage 7 uses before handing text to "
        "the voice model: initialism spell-out, digit/date/money verbalization, "
        "and (for Cebuano) segmentation with inter-segment pauses. This is what "
        "would be spoken, not audio -- synthesis itself needs a GPU."
    )

    if os.environ.get("GPU_BACKEND_URL"):
        st.caption("A GPU backend is configured -- see the \"Live pipeline\" tab for real translation and answers.")

    text = st.text_area("Text to preview", placeholder="Type a PRC-related answer here...", height=120)
    lang_choice = st.radio(
        "Language", ["Auto-detect (GlotLID)", "Cebuano", "Filipino", "English"], horizontal=True
    )

    if st.button("Preview", type="primary", disabled=not text.strip()):
        if lang_choice == "Auto-detect (GlotLID)":
            model = load_text_lid()
            result = model.predict(text)
            if result.lang is None:
                st.error("GlotLID could not identify a language for this text.")
                return
            lang = result.lang
            st.write(f"**Detected:** {LANG_NAMES[lang]} (confidence {result.confidence:.3f}, raw label `{result.raw_label}`)")
        else:
            lang = {"Cebuano": "ceb", "Filipino": "fil", "English": "eng"}[lang_choice]

        init_set = load_init_set()
        prepped = prep_tts_text(text)

        try:
            if lang == "ceb":
                segments = ceb_full_pipeline.full_ceb_pipeline_segments(prepped, init_set, demo_fallback=False)
                st.write("**Segments the voice model would receive, in order:**")
                for i, (seg_text, pause) in enumerate(segments, 1):
                    st.markdown(f"{i}. {seg_text}")
                    if pause:
                        st.caption(f"↳ {pause:.2f}s pause after this segment")
            elif lang == "fil":
                st.write("**Text the voice model would receive:**")
                st.markdown(verbalize_fil_prc(prepped, initialism_set=init_set))
            else:
                st.write("**Text the voice model would receive:**")
                st.markdown(verbalize_eng_prc(prepped, initialism_set=init_set))
        except Exception as exc:
            # The frontend raises on some inputs by design (see src/tts.py's
            # module docstring) -- a text-only turn's answer is unaffected in
            # the real pipeline, and this preview just shows why.
            st.warning(f"The frontend raised on this text: {exc}")


# --- Live pipeline, via GPU_BACKEND_URL --------------------------------------
def render_live_tab() -> None:
    """Calls a real GPU backend (app/gpu_backend_api.py) over plain HTTP for
    real translation + retrieval + generation + translation back. No torch
    import here -- requests is a plain HTTP client, same as any other. Only
    meaningful when GPU_BACKEND_URL is set to a reachable backend; otherwise
    shows why not, same "no silent fallback" spirit as the rest of this app.
    """
    gpu_backend = os.environ.get("GPU_BACKEND_URL")
    if not gpu_backend:
        st.info(
            "GPU_BACKEND_URL is not set, so this tab has nothing to call. "
            "This Fly app is CPU-only by design -- see app/gpu_backend_api.py "
            "for the backend this would reach, run separately on a real GPU host."
        )
        return

    try:
        health = requests.get(f"{gpu_backend}/health", timeout=5)
        health.raise_for_status()
        loaded = health.json().get("loaded", False)
    except requests.exceptions.RequestException as exc:
        st.error(f"GPU backend at {gpu_backend} is not reachable: {exc}")
        return

    if not loaded:
        st.warning("GPU backend is reachable but still loading its models -- try again shortly.")
        return

    st.caption(f"Connected to a live GPU backend at {gpu_backend}. Text in, real answer back -- no audio yet.")

    text = st.text_area("Your question", placeholder="Type a question here...", height=100, key="live_text")
    lang_choice = st.radio("Language", ["Cebuano", "Filipino", "English"], horizontal=True, key="live_lang")

    if st.button("Ask", type="primary", disabled=not text.strip()):
        lang = {"Cebuano": "ceb", "Filipino": "fil", "English": "eng"}[lang_choice]
        try:
            with st.spinner("Translating, retrieving, generating, translating back..."):
                resp = requests.post(
                    f"{gpu_backend}/chat", data={"lang": lang, "text": text}, timeout=60
                )
                resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            st.error(f"Request to the GPU backend failed: {exc}")
            return

        result = resp.json()
        st.markdown(f"**English:** {result['english_answer']}")
        if lang != "eng":
            st.markdown(f"**{LANG_NAMES[lang]}:** {result['native_answer']}")
        if result["chunk_ids"]:
            st.caption("Chunks used: " + ", ".join(result["chunk_ids"]))
        else:
            st.caption("No matching PRC service found.")


# --- Password gate ---------------------------------------------------------
def require_password() -> bool:
    app_password = os.environ.get("APP_PASSWORD")
    if not app_password:
        st.error(
            "APP_PASSWORD is not set. Refusing to start rather than serving this "
            "page unprotected -- see .env.example."
        )
        return False

    if st.session_state.get("authenticated"):
        return True

    st.title("LingkodAI")
    st.caption(VOICE_DISCLOSURE)
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pw == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(page_title="LingkodAI", page_icon="🇵🇭", layout="wide")

    if not require_password():
        st.stop()

    st.title("LingkodAI")
    st.caption(
        "A voice assistant for Philippine Professional Regulation Commission (PRC) "
        "services, in Cebuano, Filipino, or English. This is a containerized, "
        "reproducible pipeline with a hosted front end -- not a production system."
    )

    demo_tab, frontend_tab, live_tab = st.tabs(
        ["Recorded conversations", "Try the text frontend", "Live pipeline"]
    )
    with demo_tab:
        render_demo_tab()
    with frontend_tab:
        render_frontend_tab()
    with live_tab:
        render_live_tab()


if __name__ == "__main__":
    main()
