"""Stage 5 -- Retrieval-augmented generation.

Hybrid retrieval (Harrier-oss-v1-0.6b dense + a hand-rolled BM25 sparse
vector, fused via Qdrant DBSF) over the PRC chunk corpus, an LLM
(Qwen3-4B-Instruct-2507, 4-bit) that selects 0-N candidates, and the same
model generating the English answer.

Ported from reference/francis_rag_pipeline.ipynb. LangChain's
`RunnableWithMessageHistory` is dropped for a plain per-conversation history
list -- it only ever stored a human/ai message list, which a plain list
reproduces directly.

Prompts, the system prompt, the character-budget constants, NO_MATCH_SENTINEL
and PRC_CONTACT_FALLBACK are byte-identical to the notebook. Streaming and
time-to-first-token instrumentation are Colab demo/eval scaffolding and are
not ported -- the pipeline only needs the plain, greedy `generate_answer`.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_DIR = os.environ.get(
    "RAG_CHUNKS_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "prc_chunks"),
)
EXPECTED_CHUNK_FILES = 13
EXPECTED_TOTAL_CHUNKS = 547

HARRIER_MODEL = "microsoft/harrier-oss-v1-0.6b"
HARRIER_TASK = "Retrieve relevant passages that answer the query"
GEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
QDRANT_COLLECTION_NAME = "prc_knowledge_base_chunks"

VERIFY_TOP_K = 10  # Number of hybrid-search candidates retrieved and passed to selection.
BM25_K1 = 1.5
BM25_B = 0.75

# MAX_GENERATION_CONTEXT_CHARS is the primary safeguard, applied per-selection
# before generation ever runs. MAX_CONTEXT_CHARS is a secondary backstop
# inside build_prompt, set above the primary cap so it should rarely engage.
MAX_CANDIDATES_TOTAL_CHARS = 10000
MAX_GENERATION_CONTEXT_CHARS = 12000
MAX_CONTEXT_CHARS = 13000

REWRITE_MAX_NEW_TOKENS = 100
SELECT_MAX_NEW_TOKENS = 40
GENERATE_MAX_NEW_TOKENS = 512

NO_MATCH_SENTINEL = "NO_MATCHING_SERVICE_FOUND"
PRC_CONTACT_FALLBACK = "No matching PRC service was found for this question. Please verify with the relevant government agency or PRC directly."

SYSTEM_PROMPT = """You are LingkodAI, an assistant that answers questions about \
Professional Regulation Commission (PRC) services using ONLY the information \
provided below. Do not use any outside knowledge about PRC or government \
processes -- only what appears in the provided service information.

Follow these rules:
1. Answer only what the citizen actually asked. Do not include additional steps, requirements, fees, or processing times beyond what's needed to answer the question, unless the citizen specifically asks for more.
2. Write your answer as complete sentences in prose -- do not use bullet points, numbered lists, or headers.
3. You may be given more than one candidate document, ordered most to least likely relevant. Decide which document(s) genuinely and specifically answer the citizen's question -- most questions need only one, but occasionally two related documents are both needed.
4. Ignore any candidate document that does not match what the citizen actually described, even if it looks similar or uses nearly identical wording.
5. If NONE of the provided candidate documents answer the question -- for example, if they all describe a different document, card, or service than what the citizen asked about -- respond with exactly this text and nothing else: NO_MATCHING_SERVICE_FOUND
6. When asked for a total processing time or fee, always prefer an explicitly labeled total over summing or selecting an individual step's value yourself.
7. Some candidate services use nearly identical wording and differ only in one key detail -- the reason or trigger stated (e.g. lost/damaged vs. expiring/expired), or the specific document, card, or certificate type named (e.g. an ID card/PIC vs. a Certificate of Registration/COR). Pay close attention to these details: only the document(s) matching what the citizen actually described should be used."""


@dataclass
class HistoryTurn:
    """One turn of conversation history. type is "human" or "ai"."""

    type: str
    content: str


@dataclass
class RAGTurnResult:
    """Everything a caller needs from one conversation turn."""

    answer: str
    resolved_query: str
    chunk_ids: list[str]
    was_dependent: bool
    retrieval_s: float | None
    select_s: float | None
    generate_s: float | None


@dataclass
class RAGBundle:
    """Everything the RAG stage needs, loaded once per staged-mode phase."""

    chunks: list[dict]
    chunk_by_id: dict[str, dict]
    embed_model: SentenceTransformer
    vectorizer: CountVectorizer
    chunk_bm25: np.ndarray
    qdrant_client: QdrantClient
    gen_model: AutoModelForCausalLM
    gen_tokenizer: AutoTokenizer
    device: str

    def unload(self) -> None:
        del self.embed_model, self.gen_model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load_chunks(
    data_dir: str | Path | None = None,
    expected_files: int = EXPECTED_CHUNK_FILES,
    expected_total: int = EXPECTED_TOTAL_CHUNKS,
) -> list[dict]:
    """Load and combine all PRC chunk files, asserting the expected shape.

    Globs `*_chunks.json` under data_dir. Filenames must already have any
    download suffix stripped (e.g. `charter_chunks (7).json` ->
    `charter_chunks.json`), or the glob silently skips them.
    """
    data_dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    paths = sorted(data_dir.glob("*_chunks.json"))
    if len(paths) != expected_files:
        raise RuntimeError(
            f"Expected {expected_files} chunk files under {data_dir}, found {len(paths)}: "
            f"{[p.name for p in paths]}"
        )

    chunks: list[dict] = []
    for path in paths:
        with open(path) as f:
            chunks.extend(json.load(f))

    if len(chunks) != expected_total:
        raise RuntimeError(f"Expected {expected_total} total chunks, found {len(chunks)}")

    ids = [c["chunk_id"] for c in chunks]
    duplicates = {cid for cid in ids if ids.count(cid) > 1}
    if duplicates:
        raise RuntimeError(f"Duplicate chunk_id(s) across documents: {duplicates}")

    return chunks


def harrier_instruct(query: str, task: str = HARRIER_TASK) -> str:
    """Wraps a query (not a document) with Harrier's instruction prefix."""
    return f"Instruct: {task}\nQuery: {query}"


def bm25_weight_matrix(tf_matrix: np.ndarray, k1: float = BM25_K1, b: float = BM25_B) -> np.ndarray:
    """Hand-rolled BM25 term-weight matrix over a term-frequency matrix."""
    n_docs, n_terms = tf_matrix.shape
    df = (tf_matrix > 0).sum(axis=0)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
    doc_len = tf_matrix.sum(axis=1, keepdims=True)
    avgdl = doc_len.mean()
    denom = tf_matrix + k1 * (1 - b + b * doc_len / avgdl)
    return idf * (tf_matrix * (k1 + 1)) / np.clip(denom, 1e-9, None)


def _build_qdrant_collection(
    chunk_embeddings: np.ndarray,
    chunk_bm25: np.ndarray,
    chunks: list[dict],
    name: str = QDRANT_COLLECTION_NAME,
) -> QdrantClient:
    """Builds an in-memory Qdrant collection with a dense vector
    (Harrier-oss-v1-0.6b, cosine) and a sparse vector (BM25) per chunk, so
    hybrid_search can fuse the two at query time with DBSF."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=chunk_embeddings.shape[1], distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"bm25_sparse": models.SparseVectorParams()},
    )

    points = []
    for i, c in enumerate(chunks):
        nonzero = np.nonzero(chunk_bm25[i])[0]
        points.append(
            models.PointStruct(
                id=i,
                vector={
                    "dense": chunk_embeddings[i].tolist(),
                    "bm25_sparse": models.SparseVector(
                        indices=nonzero.tolist(), values=chunk_bm25[i][nonzero].tolist()
                    ),
                },
                payload={"chunk_id": c["chunk_id"], "chunk_title": c["chunk_title"]},
            )
        )
    client.upsert(collection_name=name, points=points)
    return client


def load(data_dir: str | Path | None = None, device: str | None = None) -> RAGBundle:
    """Load the chunk corpus, Harrier + BM25 + Qdrant index, and Qwen3-4B."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    chunks = load_chunks(data_dir)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    embed_model = SentenceTransformer(HARRIER_MODEL, model_kwargs={"dtype": "auto"})
    chunk_embeddings = embed_model.encode(
        [c["retrieval_text"] for c in chunks], normalize_embeddings=True
    ).astype(np.float32)

    vectorizer = CountVectorizer(token_pattern=r"[a-zA-Z0-9]+", lowercase=True)
    chunk_countvec = (
        vectorizer.fit_transform([c["retrieval_text"] for c in chunks]).toarray().astype(np.float32)
    )
    chunk_bm25 = bm25_weight_matrix(chunk_countvec).astype(np.float32)

    # The GPU is only needed to embed the chunks once; move the embedding
    # model to CPU to free its GPU memory for the generation model below.
    embed_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qdrant_client = _build_qdrant_collection(chunk_embeddings, chunk_bm25, chunks)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    return RAGBundle(
        chunks=chunks,
        chunk_by_id=chunk_by_id,
        embed_model=embed_model,
        vectorizer=vectorizer,
        chunk_bm25=chunk_bm25,
        qdrant_client=qdrant_client,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        device=device,
    )


def hybrid_search(bundle: RAGBundle, query: str, k: int = VERIFY_TOP_K) -> list[tuple[str, str, float]]:
    """Encodes the query (dense Harrier embedding + sparse BM25) and returns
    the top-k chunks by DBSF-fused hybrid search via Qdrant, as
    (chunk_id, chunk_title, score) tuples."""
    q_dense = bundle.embed_model.encode(
        [harrier_instruct(query)], normalize_embeddings=True
    )[0].astype(np.float32)
    q_sparse_vec = bundle.vectorizer.transform([query]).toarray().astype(np.float32)[0]
    q_nz = np.nonzero(q_sparse_vec)[0]

    result = bundle.qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=q_dense.tolist(), using="dense", limit=k),
            models.Prefetch(
                query=models.SparseVector(indices=q_nz.tolist(), values=[1.0] * len(q_nz)),
                using="bm25_sparse",
                limit=k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        limit=k,
    )
    return [(p.payload["chunk_id"], p.payload["chunk_title"], p.score) for p in result.points]


def generate_answer(
    bundle: RAGBundle, messages: list[dict], max_new_tokens: int = GENERATE_MAX_NEW_TOKENS
) -> str:
    """Qwen3-4B-Instruct-2507 is non-thinking-only, so no thinking-mode flag
    is needed. Deterministic (greedy) decoding, for reproducibility across
    selection and generation calls."""
    text = bundle.gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = bundle.gen_tokenizer([text], return_tensors="pt").to(bundle.gen_model.device)
    outputs = bundle.gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return bundle.gen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def rewrite_followup(bundle: RAGBundle, query: str, history: list[HistoryTurn]) -> str:
    """Rewrites a follow-up question into a standalone question using the
    prior conversation, if the question depends on it; otherwise returns
    the question unchanged."""
    history_text = "\n".join(
        f"{'Citizen' if turn.type == 'human' else 'LingkodAI'}: {turn.content}" for turn in history
    )
    prompt = f"""Conversation so far:
{history_text}

New question: {query}

If this question contains a vague reference (e.g. "that", "it", "this") that \
can ONLY be understood using the conversation above, rewrite it into a fully \
standalone question by replacing that reference with the specific topic from \
the conversation above. If the question already fully specifies what it's \
asking about within itself -- even if it happens to contain a word like \
"it" -- return it completely unchanged. Do not pull in details from the \
conversation above unless genuinely necessary.

Respond with ONLY the resulting question, nothing else."""

    messages = [
        {"role": "system", "content": "You rewrite follow-up questions into standalone questions."},
        {"role": "user", "content": prompt},
    ]
    return generate_answer(bundle, messages, max_new_tokens=REWRITE_MAX_NEW_TOKENS).strip()


def resolve_followup(bundle: RAGBundle, query: str, history: list[HistoryTurn]) -> tuple[str, bool]:
    """Determines whether a query depends on the prior conversation, and
    resolves it into a standalone question if so.

    Returns (resolved_query, was_dependent), where was_dependent is True
    if and only if the query was rewritten.
    """
    if not history:
        return query, False
    rewritten = rewrite_followup(bundle, query, history)
    return rewritten, rewritten != query


def build_candidate_lines(
    items: list, get_text_fn, max_total_chars: int = MAX_CANDIDATES_TOTAL_CHARS
) -> list[str]:
    """Builds numbered candidate lines for a selection prompt, truncating
    individual candidates only as needed to keep the combined total under
    budget. Budget is redistributed evenly across remaining candidates, so a
    few short candidates do not waste budget that a later long one could
    use. Truncation only engages when the combined total would exceed the
    budget; most queries' candidate sets never approach this cap."""
    lines = []
    remaining_budget = max_total_chars
    n_remaining = len(items)
    for i, item in enumerate(items, 1):
        text = get_text_fn(item)
        share = max(200, remaining_budget // max(1, n_remaining))  # floor so nothing gets truncated to nothing
        if len(text) > share:
            text = text[:share].rsplit(" ", 1)[0] + "..."
        lines.append(f"{i}. {text}")
        remaining_budget -= len(text)
        n_remaining -= 1
    return lines


def select_candidates_for_generation(
    bundle: RAGBundle, query: str, candidates: list[tuple[str, str, float]]
) -> list[str]:
    """Selects 0-10 candidates from the VERIFY_TOP_K retrieval results
    using only retrieval_text. An empty result means no match and skips
    generation. This is the only stage where candidates may be rejected.

    candidates is the output of hybrid_search(bundle, query, k=VERIFY_TOP_K):
    (chunk_id, chunk_title, score) tuples, ordered by the DBSF-fused
    retrieval score.
    """
    candidate_ids = [c[0] for c in candidates]
    candidate_lines = build_candidate_lines(
        candidate_ids, lambda cid: bundle.chunk_by_id[cid]["retrieval_text"]
    )

    prompt = f"""Citizen's Question: {query}

Candidate Services (ordered by retrieval relevance):
{chr(10).join(candidate_lines)}

Decide which of these numbered candidates you would need to read in full to answer the citizen's question. Apply these rules:
a. Every candidate is a PRC (Professional Regulation Commission) service for a PRC-regulated profession. If the citizen's question is about a license, ID, or document from a different government agency (e.g. driver's license/LTO, passport/DFA, SSS, PhilHealth, business permit, barangay clearance), say NONE even if the wording looks similar -- PRC does not handle it.
b. Most questions need only one candidate; occasionally two related candidates are both needed.
c. Select as many or as few as are genuinely necessary. Do not include a candidate just because it appeared on this list.
d. If none of them plausibly answer the question, say so.

Respond with ONLY a comma-separated list of the numbers you need, most relevant first (e.g. "3,1"), or "NONE" if none apply. No explanation."""

    messages = [
        {
            "role": "system",
            "content": "You help decide which government service "
            "listings, from a ranked list of candidates, "
            "are actually needed to answer a citizen's question.",
        },
        {"role": "user", "content": prompt},
    ]
    response = generate_answer(bundle, messages, max_new_tokens=SELECT_MAX_NEW_TOKENS).strip()

    if "NONE" in response.upper():
        return []

    seen = set()
    selected_ids = []
    for tok in re.findall(r"\d+", response):
        idx = int(tok) - 1
        if 0 <= idx < len(candidate_ids) and candidate_ids[idx] not in seen:
            seen.add(candidate_ids[idx])
            selected_ids.append(candidate_ids[idx])

    return selected_ids


def build_totals_note(chunk: dict) -> str:
    """Builds an unambiguous totals summary from the service's total
    processing time and fee fields."""
    total_time = chunk.get("total_processing_time", "")
    total_fee = chunk.get("total_fee", "")

    if not total_time and not total_fee:
        return ""

    lines = [
        "IMPORTANT -- Official totals for this service. Use these values when "
        "answering questions about total processing time or total fee -- "
        "do NOT use an individual step's time or fee as the total."
    ]

    if total_time:
        lines.append(f"- Total Processing Time: {total_time}")
    if total_fee:
        # Strip the redundant leading "Total Standard Fee: " label where present
        # (single-condition services); multi-condition services embed their own
        # per-line labels and are left as-is.
        FEE_PREFIX = "Total Standard Fee: "
        display_fee = total_fee[len(FEE_PREFIX):] if total_fee.startswith(FEE_PREFIX) else total_fee
        lines.append(f"- Total Fee: {display_fee}")

    return "\n".join(lines)


def cap_selected_chunks_by_length(
    selected_ids: list[str], chunk_by_id: dict[str, dict], max_chars: int = MAX_GENERATION_CONTEXT_CHARS
) -> list[str]:
    """Keeps chunks in order (most-relevant-first, from
    select_candidates_for_generation) until combined full_entry_text length
    would exceed max_chars. Always keeps at least the top chunk, even if it
    alone exceeds the cap."""
    capped = []
    total = 0
    for cid in selected_ids:
        length = len(chunk_by_id[cid]["full_entry_text"])
        if capped and total + length > max_chars:
            break
        capped.append(cid)
        total += length
    return capped


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Assembles the final system + user messages sent to the generation
    model. Combines the selected chunks' full_entry_text (plus each
    chunk's totals note) into a single context block, then wraps it with
    the citizen's question and instructions in the user message."""
    sections = []
    for i, chunk in enumerate(chunks, 1):
        totals_note = build_totals_note(chunk)
        piece = f"--- Candidate Document {i} ---\n{chunk['full_entry_text']}"
        if totals_note:
            piece += f"\n{totals_note}"
        sections.append(piece)

    context = "\n\n".join(sections)
    truncated = len(context) > MAX_CONTEXT_CHARS
    if truncated:
        context = context[:MAX_CONTEXT_CHARS]

    user_prompt = f"""PRC Service Information ({len(chunks)} candidate document(s), ordered most to least likely relevant):
{context}
{"[... additional content truncated for length ...]" if truncated else ""}

Citizen's Question: {query}

Answer the question using only the service information above. If more than \
one candidate document is provided, use only the one(s) that are actually \
relevant -- ignore any that don't apply."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def retrieve_and_select(bundle: RAGBundle, query: str) -> tuple[list[dict], float | None, float | None]:
    """Bridges retrieval and selection for use inside a multi-turn
    conversation. Returns (chunks, retrieval_s, select_s): chunks is an
    ordered list of chunk dicts, already length-capped for generation. An
    empty list means nothing passed selection, and the caller falls back to
    PRC_CONTACT_FALLBACK. retrieval_s/select_s are None for whichever stage
    never ran.

    CUDA cache is cleared between each stage, since memory fragmentation and
    reference-cycle buildup across these back-to-back heavy calls is what
    drives OOM risk here, especially across a multi-turn conversation's
    several turns run in sequence."""
    t0 = time.perf_counter()
    top_candidates = hybrid_search(bundle, query, k=VERIFY_TOP_K)
    retrieval_s = time.perf_counter() - t0
    if bundle.device == "cuda":
        torch.cuda.empty_cache()
    if not top_candidates:
        return [], retrieval_s, None

    t0 = time.perf_counter()
    selected_ids = select_candidates_for_generation(bundle, query, top_candidates)
    select_s = time.perf_counter() - t0
    if bundle.device == "cuda":
        torch.cuda.empty_cache()
    if not selected_ids:
        return [], retrieval_s, select_s

    selected_ids = cap_selected_chunks_by_length(selected_ids, bundle.chunk_by_id)
    return [bundle.chunk_by_id[cid] for cid in selected_ids], retrieval_s, select_s


def answer_turn(bundle: RAGBundle, query: str, history: list[HistoryTurn]) -> RAGTurnResult:
    """Runs one conversation turn: follow-up resolution, hybrid retrieval,
    candidate selection, and generation.

    Caller appends HistoryTurn("human", query) and
    HistoryTurn("ai", result.answer) to history afterward -- history stores
    the original query, not resolved_query, matching what the notebook's
    LangChain history stored.
    """
    resolved_query, was_dependent = resolve_followup(bundle, query, history)

    chunks, retrieval_s, select_s = retrieve_and_select(bundle, resolved_query)
    generate_s = None
    if not chunks:
        answer = PRC_CONTACT_FALLBACK
    else:
        messages = build_prompt(resolved_query, chunks)
        t0 = time.perf_counter()
        raw_answer = generate_answer(bundle, messages, max_new_tokens=GENERATE_MAX_NEW_TOKENS)
        generate_s = time.perf_counter() - t0
        answer = PRC_CONTACT_FALLBACK if NO_MATCH_SENTINEL in raw_answer else raw_answer

    return RAGTurnResult(
        answer=answer,
        resolved_query=resolved_query,
        chunk_ids=[c["chunk_id"] for c in chunks],
        was_dependent=was_dependent,
        retrieval_s=retrieval_s,
        select_s=select_s,
        generate_s=generate_s,
    )
