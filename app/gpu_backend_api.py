"""GPU backend for the GPU_BACKEND_URL architecture. Runs on a real GPU host
(Colab), tunneled out with cloudflared, called by the already-live Fly CPU
app (app/streamlit_app.py) over HTTP -- see reference/colab_streamlit_
launcher.ipynb for the tunnel pattern this reuses.

NOT part of docker/Dockerfile.cpu or docker/Dockerfile.gpu, not referenced
by fly.toml. Presentation tooling, not a shipped artifact. FastAPI, not
Streamlit: this backend has no UI of its own, the Fly app is the UI.

Phase 1 (this file, initially): stages 4-6 (MT in -> RAG -> MT out), text
only, reusing the exact same resident-loading pattern as
app/gpu_chat_demo.py -- st.cache_resource there, module-level globals
loaded once at startup here.

Phase 2 (added after Phase 1 is verified live): stages 2 and 7 (ASR, TTS)
too, audio in -> audio out. No audio LID (stage 1) -- language is a
required request field the caller picks manually, the same simplification
gpu_chat_demo.py already makes for text LID; this also avoids needing to
transfer models/deeper_50chunks.pt (Bea's file) to this host.

Run with:
    uvicorn app.gpu_backend_api:app --host 0.0.0.0 --port 8500
Loads eagerly at startup (not on first request) -- there is no "warm up"
button possible for a headless API, so the tunnel should not even be
considered usable until startup finishes. Watch the log for
"Application startup complete" before tunneling.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import mt, rag  # noqa: E402

app = FastAPI(title="LingkodAI GPU backend (presentation tooling, not a shipped artifact)")

_mt_bundle: mt.MTBundle | None = None
_rag_bundle: rag.RAGBundle | None = None


@app.on_event("startup")
def _load_models() -> None:
    global _mt_bundle, _rag_bundle
    print("Loading NLLB (translation)...")
    _mt_bundle = mt.load()
    print("Loading RAG (retrieval + generation)...")
    _rag_bundle = rag.load()
    print("Startup loading complete -- resident and ready.")


@app.get("/health")
def health() -> dict:
    """Reachability + readiness check. The Fly frontend calls this before
    showing its live-pipeline tab's input form, so a caller can tell
    'unreachable' apart from 'reachable but still loading'."""
    return {"status": "ok", "loaded": _mt_bundle is not None and _rag_bundle is not None}


@app.post("/chat")
async def chat(lang: str = Form(...), text: str = Form(...)) -> dict:
    """Stages 4-6 on typed text. lang is one of ceb/fil/eng, picked by the
    caller (no audio LID or text LID here -- the caller already knows)."""
    if _mt_bundle is None or _rag_bundle is None:
        raise HTTPException(503, "Models still loading; try again shortly.")

    mt_in = mt.translate_to_english(_mt_bundle, [mt.TurnText(id=0, text=text, lang=lang)])[0]
    if not mt_in.ok:
        raise HTTPException(500, f"MT in failed: {mt_in.failure_reason}")

    # No cross-request history yet: each /chat call is stateless. A
    # follow-up-aware version would need the caller to pass prior turns
    # back in, which the Fly UI doesn't do yet -- fine for Phase 1.
    rag_result = rag.answer_turn(_rag_bundle, mt_in.text, [])

    mt_out = mt.translate_from_english(
        _mt_bundle, [mt.TurnText(id=0, text=rag_result.answer, lang=lang)]
    )[0]
    if not mt_out.ok:
        raise HTTPException(500, f"MT out failed: {mt_out.failure_reason}")

    return {
        "english_query": mt_in.text,
        "resolved_query": rag_result.resolved_query,
        "was_dependent": rag_result.was_dependent,
        "english_answer": rag_result.answer,
        "native_answer": mt_out.text,
        "chunk_ids": rag_result.chunk_ids,
    }
