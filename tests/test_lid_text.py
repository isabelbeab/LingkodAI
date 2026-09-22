"""CPU-only smoke test for src/lid_text.py. Never downloads GlotLID -- a
FakeFastText stands in for the fasttext model."""

from __future__ import annotations

import pytest

from src import lid_text


class FakeFastText:
    def __init__(self, label_probs):
        self._label_probs = label_probs
        self.predict_calls = []

    def predict(self, text, k=-1):
        self.predict_calls.append(text)
        labels = tuple(self._label_probs)
        probs = tuple(self._label_probs[label] for label in labels)
        return labels, probs


def make_model(label_probs) -> lid_text.TextLID:
    return lid_text.TextLID(model=FakeFastText(label_probs))


def test_predict_empty_text_returns_all_none():
    model = make_model({})
    result = model.predict("")
    assert result.lang is None
    assert result.raw_label is None
    assert result.confidence is None
    assert model.model.predict_calls == []  # early return, model never touched


def test_predict_whitespace_only_also_returns_all_none():
    model = make_model({})
    result = model.predict("   \n  ")
    assert result.lang is None


def test_predict_ceb_highest_prob_wins():
    model = make_model({
        "__label__ceb_Latn": 0.9,
        "__label__fil_Latn": 0.05,
        "__label__eng_Latn": 0.05,
    })
    result = model.predict("Kumusta ka?")
    assert result.lang == "ceb"
    assert result.raw_label == "ceb_Latn"
    assert result.confidence == pytest.approx(0.9)


def test_predict_tgl_maps_to_fil():
    model = make_model({"__label__tgl_Latn": 0.8})
    result = model.predict("Kamusta ka?")
    assert result.lang == "fil"
    assert result.raw_label == "tgl_Latn"


def test_predict_missing_candidate_labels_default_to_zero():
    # fasttext's predict(k=-1) only returns labels it scored above zero; any
    # CANDIDATE_LABELS absent from that set must score 0.0, not raise.
    model = make_model({"__label__ceb_Latn": 0.3})
    result = model.predict("some text")
    assert result.lang == "ceb"


def test_predict_strips_linebreaks():
    model = make_model({"__label__eng_Latn": 1.0})
    model.predict("line one\nline two")
    assert model.model.predict_calls[0] == "line one line two"


def test_label_map_covers_all_candidate_labels():
    for label in lid_text.CANDIDATE_LABELS:
        raw = label.replace("__label__", "")
        assert raw in lid_text.LABEL_MAP
