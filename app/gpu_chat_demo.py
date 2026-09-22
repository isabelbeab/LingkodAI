"""JOJIE-only live demo for the presentation. GPU required, torch/transformers
imported directly (unlike app/streamlit_app.py, which must stay torch-free
so the CPU/Fly image can build without them).

NOT part of docker/Dockerfile.cpu or docker/Dockerfile.gpu, not referenced by
fly.toml. This is presentation tooling: run directly on JOJIE with
`streamlit run app/gpu_chat_demo.py`, tunneled out with cloudflared (see
reference/colab_streamlit_launcher.ipynb for the exact tunnel command this
mirrors), and torn down after use. Never describe this as a fourth shipped
artifact.

Text in, stages 4-6 only (MT in -> RAG -> MT out). No audio LID, ASR, or TTS
-- no audio in or out anywhere in this file, by design (HK: "we can skip ASR
and TTS if we really can't... at the very least we should try to show the
text chatbot component").

Per-question, not per-conversation: loads each stage's model, uses it, and
unloads it (src/mt.py's and src/rag.py's existing load()/unload(), unchanged)
for every single question, the same staged pattern src/pipeline.py's
run_staged already proves fits on JOJIE's 11 GB card -- just scoped to one
question's three stages instead of a whole conversation's five phases. This
trades a per-question wait (model load from a warm HF cache) for near
certainty that it fits, rather than gambling on NLLB and RAG's generation
model coexisting resident (unmeasured, likely tight to over budget on 11 GB:
NLLB fp16 alone is 6.7 GB).

No password gate: unlike the CPU demo, nothing here touches the cloned
research-speaker voice (no TTS at all), and the RAG knowledge base is public
PRC citizen's-charter content. Nothing here needs gating.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import lid_text, mt, rag  # noqa: E402

LANG_NAMES = {"ceb": "Cebuano", "fil": "Filipino", "eng": "English"}


@st.cache_resource(show_spinner="Loading GlotLID (text language ID)...")
def load_text_lid() -> lid_text.TextLID:
    return lid_text.load()


def print_environment() -> None:
    import torch
    import transformers

    st.caption(
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only)'}"
    )


def run_turn(text: str, lang: str, history: list[rag.HistoryTurn]) -> dict:
    """One question through stages 4-6, loading/using/unloading each stage's
    model in turn. Returns a dict of everything worth showing; raises only
    if a stage itself raises unexpectedly (caught by the caller)."""
    result: dict = {"native_query": text, "final_lang": lang}

    with st.status("Translating to English...", expanded=False) as status:
        mt_bundle = mt.load()
        try:
            mt_in = mt.translate_to_english(mt_bundle, [mt.TurnText(id=0, text=text, lang=lang)])[0]
        finally:
            mt_bundle.unload()
        if not mt_in.ok:
            status.update(label="MT in failed", state="error")
            raise RuntimeError(f"MT in failed: {mt_in.failure_reason}")
        result["english_query"] = mt_in.text
        status.update(label=f"Translated to English ({mt_in.translate_time_sec:.1f}s)", state="complete")

    with st.status("Retrieving and generating answer...", expanded=False) as status:
        rag_bundle = rag.load()
        try:
            rag_result = rag.answer_turn(rag_bundle, mt_in.text, history)
        finally:
            rag_bundle.unload()
        result["resolved_query"] = rag_result.resolved_query
        result["was_dependent"] = rag_result.was_dependent
        result["english_answer"] = rag_result.answer
        result["chunk_ids"] = rag_result.chunk_ids
        status.update(label="Answer generated", state="complete")

    with st.status("Translating answer back...", expanded=False) as status:
        mt_bundle = mt.load()
        try:
            mt_out = mt.translate_from_english(
                mt_bundle, [mt.TurnText(id=0, text=rag_result.answer, lang=lang)]
            )[0]
        finally:
            mt_bundle.unload()
        if not mt_out.ok:
            status.update(label="MT out failed", state="error")
            raise RuntimeError(f"MT out failed: {mt_out.failure_reason}")
        result["native_answer"] = mt_out.text
        status.update(label=f"Translated back ({mt_out.translate_time_sec:.1f}s)", state="complete")

    return result


def main() -> None:
    st.set_page_config(page_title="LingkodAI (live, JOJIE)", page_icon="🇵🇭", layout="wide")
    st.title("LingkodAI -- live text chatbot (JOJIE)")
    st.caption(
        "Presentation-only live demo. Text in, real translation + retrieval + "
        "generation + translation back -- no audio anywhere. Each question "
        "loads and unloads its own models, so expect a real wait per turn."
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


if __name__ == "__main__":
    main()
