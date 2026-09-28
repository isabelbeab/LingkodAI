"""CPU-only smoke test for src/rag.py. Never loads Harrier, Qdrant, or
Qwen3-4B for real -- fakes stand in for each, matching the shapes rag.py
actually calls."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src import rag


# --- fakes -----------------------------------------------------------------


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeEmbedModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.ones((len(texts), 4), dtype=np.float32)

    def to(self, device):
        return self


class FakeVectorizer:
    def transform(self, texts):
        class _Arr:
            def toarray(self_inner):
                return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        return _Arr()


class FakePoint:
    def __init__(self, chunk_id, title, score):
        self.payload = {"chunk_id": chunk_id, "chunk_title": title}
        self.score = score


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    def __init__(self, points):
        self._points = points
        self.last_kwargs = None

    def query_points(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeQueryResult(self._points)


class FakeGenTokenizer:
    def __init__(self, responses):
        self._responses = list(responses)
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.chat_calls.append(messages)
        return "PROMPT"

    def __call__(self, texts, return_tensors="pt"):
        return FakeBatch(input_ids=np.zeros((1, 3), dtype=np.int64))

    def decode(self, ids, skip_special_tokens=True):
        return self._responses.pop(0)


class FakeGenModel:
    device = "cpu"

    def generate(self, **kwargs):
        return np.zeros((1, 5), dtype=np.int64)


def make_bundle(gen_responses=None, hybrid_points=None) -> rag.RAGBundle:
    return rag.RAGBundle(
        chunks=[],
        chunk_by_id={},
        embed_model=FakeEmbedModel(),
        vectorizer=FakeVectorizer(),
        chunk_bm25=np.zeros((0, 0), dtype=np.float32),
        qdrant_client=FakeQdrantClient(hybrid_points or []),
        gen_model=FakeGenModel(),
        gen_tokenizer=FakeGenTokenizer(gen_responses or []),
        device="cpu",
    )


def write_chunk_file(path, chunks):
    path.write_text(json.dumps(chunks))


# --- load_chunks -------------------------------------------------------


def test_load_chunks_happy_path(tmp_path):
    write_chunk_file(
        tmp_path / "alpha_chunks.json",
        [{"chunk_id": "a1", "chunk_title": "A", "retrieval_text": "rt", "full_entry_text": "fe"}],
    )
    write_chunk_file(
        tmp_path / "beta_chunks.json",
        [
            {"chunk_id": "b1", "chunk_title": "B1", "retrieval_text": "rt", "full_entry_text": "fe"},
            {"chunk_id": "b2", "chunk_title": "B2", "retrieval_text": "rt", "full_entry_text": "fe"},
        ],
    )
    chunks = rag.load_chunks(tmp_path, expected_files=2, expected_total=3)
    assert len(chunks) == 3
    assert {c["chunk_id"] for c in chunks} == {"a1", "b1", "b2"}


def test_load_chunks_wrong_file_count_raises(tmp_path):
    write_chunk_file(tmp_path / "alpha_chunks.json", [{"chunk_id": "a1"}])
    with pytest.raises(RuntimeError, match="Expected 2 chunk files"):
        rag.load_chunks(tmp_path, expected_files=2, expected_total=1)


def test_load_chunks_wrong_total_raises(tmp_path):
    write_chunk_file(tmp_path / "alpha_chunks.json", [{"chunk_id": "a1"}])
    with pytest.raises(RuntimeError, match="Expected 5 total chunks"):
        rag.load_chunks(tmp_path, expected_files=1, expected_total=5)


def test_load_chunks_duplicate_id_raises(tmp_path):
    write_chunk_file(tmp_path / "alpha_chunks.json", [{"chunk_id": "dup"}])
    write_chunk_file(tmp_path / "beta_chunks.json", [{"chunk_id": "dup"}])
    with pytest.raises(RuntimeError, match="Duplicate chunk_id"):
        rag.load_chunks(tmp_path, expected_files=2, expected_total=2)


def test_load_chunks_skips_files_with_download_suffix(tmp_path):
    # "beta_chunks (1).json" does not end in "_chunks.json", so the glob
    # silently skips it -- download suffixes must be stripped before loading.
    write_chunk_file(tmp_path / "alpha_chunks.json", [{"chunk_id": "a1"}])
    write_chunk_file(tmp_path / "beta_chunks (1).json", [{"chunk_id": "b1"}])
    with pytest.raises(RuntimeError, match="Expected 2 chunk files"):
        rag.load_chunks(tmp_path, expected_files=2, expected_total=2)


# --- bm25_weight_matrix ------------------------------------------------


def test_bm25_weight_matrix_known_values():
    tf = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    weights = rag.bm25_weight_matrix(tf, k1=1.5, b=0.75)

    expected_idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1.0)
    expected_weight = expected_idf * (1 * 2.5) / 2.5

    assert weights.shape == (2, 2)
    assert np.isclose(weights[0, 0], expected_weight)
    assert np.isclose(weights[1, 1], expected_weight)
    assert np.isclose(weights[0, 1], 0.0)
    assert np.isclose(weights[1, 0], 0.0)


# --- build_candidate_lines -----------------------------------------------


def test_build_candidate_lines_redistributes_unused_budget():
    texts = {"a": "short", "b": "x" * 2000}
    lines = rag.build_candidate_lines(["a", "b"], texts.get, max_total_chars=1000)

    assert lines[0] == "1. short"
    assert lines[1].startswith("2. ")
    assert lines[1].endswith("...")
    # "a" only used 5 of its ~500-char share, so "b" gets far more than an
    # even 500/500 split would give it.
    assert len(lines[1]) > 500


def test_build_candidate_lines_no_truncation_under_budget():
    texts = {"a": "short one", "b": "short two"}
    lines = rag.build_candidate_lines(["a", "b"], texts.get, max_total_chars=10000)
    assert lines == ["1. short one", "2. short two"]


# --- cap_selected_chunks_by_length -----------------------------------------


def test_cap_selected_chunks_stops_before_exceeding_budget():
    chunk_by_id = {
        "c1": {"full_entry_text": "x" * 5000},
        "c2": {"full_entry_text": "y" * 8000},
        "c3": {"full_entry_text": "z" * 100},
    }
    capped = rag.cap_selected_chunks_by_length(["c1", "c2", "c3"], chunk_by_id, max_chars=6000)
    # c2 alone would push the total over budget, so the loop stops there --
    # c3 is never considered even though it alone would have fit.
    assert capped == ["c1"]


def test_cap_selected_chunks_always_keeps_top_chunk():
    chunk_by_id = {"big": {"full_entry_text": "z" * 20000}}
    capped = rag.cap_selected_chunks_by_length(["big"], chunk_by_id, max_chars=6000)
    assert capped == ["big"]


# --- build_totals_note -----------------------------------------------------


def test_build_totals_note_strips_fee_prefix():
    chunk = {"total_processing_time": "3 days", "total_fee": "Total Standard Fee: PHP 100"}
    note = rag.build_totals_note(chunk)
    assert "Total Processing Time: 3 days" in note
    assert "- Total Fee: PHP 100" in note
    assert "Total Standard Fee:" not in note


def test_build_totals_note_empty_when_no_totals():
    assert rag.build_totals_note({}) == ""


# --- select_candidates_for_generation ---------------------------------------


def test_select_candidates_parses_comma_list_most_relevant_first():
    bundle = make_bundle(gen_responses=["3,1"])
    bundle.chunk_by_id = {
        "id1": {"retrieval_text": "text one"},
        "id2": {"retrieval_text": "text two"},
        "id3": {"retrieval_text": "text three"},
    }
    candidates = [("id1", "t1", 0.9), ("id2", "t2", 0.8), ("id3", "t3", 0.7)]
    selected = rag.select_candidates_for_generation(bundle, "some query", candidates)
    assert selected == ["id3", "id1"]


def test_select_candidates_none_response_returns_empty():
    bundle = make_bundle(gen_responses=["NONE"])
    bundle.chunk_by_id = {"id1": {"retrieval_text": "text one"}}
    selected = rag.select_candidates_for_generation(bundle, "q", [("id1", "t1", 0.9)])
    assert selected == []


# --- hybrid_search -----------------------------------------------------------


def test_hybrid_search_returns_points_in_order():
    points = [FakePoint("id1", "Title 1", 0.9), FakePoint("id2", "Title 2", 0.7)]
    bundle = make_bundle(hybrid_points=points)
    result = rag.hybrid_search(bundle, "query text", k=2)
    assert result == [("id1", "Title 1", 0.9), ("id2", "Title 2", 0.7)]
    assert bundle.qdrant_client.last_kwargs["limit"] == 2
    assert bundle.qdrant_client.last_kwargs["query"].fusion == rag.models.Fusion.DBSF


# --- resolve_followup / rewrite_followup ------------------------------------


def test_resolve_followup_empty_history_skips_model_call():
    bundle = make_bundle(gen_responses=[])
    resolved, was_dependent = rag.resolve_followup(bundle, "What are the requirements?", [])
    assert resolved == "What are the requirements?"
    assert was_dependent is False
    assert bundle.gen_tokenizer.chat_calls == []


def test_resolve_followup_rewrites_when_dependent():
    bundle = make_bundle(gen_responses=["What are the requirements for a PRC ID?"])
    history = [rag.HistoryTurn("human", "How do I get a PRC ID?"), rag.HistoryTurn("ai", "...")]
    resolved, was_dependent = rag.resolve_followup(bundle, "What are the requirements for that?", history)
    assert resolved == "What are the requirements for a PRC ID?"
    assert was_dependent is True


def test_resolve_followup_unchanged_when_self_contained():
    bundle = make_bundle(gen_responses=["Is PRC open on weekends?"])
    history = [rag.HistoryTurn("human", "How do I renew my license?"), rag.HistoryTurn("ai", "...")]
    resolved, was_dependent = rag.resolve_followup(bundle, "Is PRC open on weekends?", history)
    assert resolved == "Is PRC open on weekends?"
    assert was_dependent is False


# --- answer_turn end-to-end (all fakes) -------------------------------------


def test_answer_turn_no_candidates_falls_back():
    bundle = make_bundle(hybrid_points=[])
    result = rag.answer_turn(bundle, "Some question", [])
    assert result.answer == rag.PRC_CONTACT_FALLBACK
    assert result.chunk_ids == []
    assert result.select_s is None
    assert result.generate_s is None


def test_answer_turn_selection_none_falls_back():
    points = [FakePoint("id1", "Title", 0.9)]
    bundle = make_bundle(gen_responses=["NONE"], hybrid_points=points)
    bundle.chunk_by_id = {"id1": {"retrieval_text": "text", "full_entry_text": "full text", "chunk_id": "id1"}}
    result = rag.answer_turn(bundle, "question", [])
    assert result.answer == rag.PRC_CONTACT_FALLBACK
    assert result.generate_s is None


def test_answer_turn_normal_path_returns_answer_and_chunk_ids():
    points = [FakePoint("id1", "Title", 0.9)]
    bundle = make_bundle(gen_responses=["1", "The answer is X."], hybrid_points=points)
    bundle.chunk_by_id = {"id1": {"retrieval_text": "text", "full_entry_text": "full text", "chunk_id": "id1"}}
    result = rag.answer_turn(bundle, "question", [])
    assert result.answer == "The answer is X."
    assert result.chunk_ids == ["id1"]
    assert result.generate_s is not None


def test_answer_turn_no_match_sentinel_substituted():
    points = [FakePoint("id1", "Title", 0.9)]
    bundle = make_bundle(gen_responses=["1", rag.NO_MATCH_SENTINEL], hybrid_points=points)
    bundle.chunk_by_id = {"id1": {"retrieval_text": "text", "full_entry_text": "full text", "chunk_id": "id1"}}
    result = rag.answer_turn(bundle, "question", [])
    assert result.answer == rag.PRC_CONTACT_FALLBACK
