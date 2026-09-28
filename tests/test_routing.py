"""Smoke test for src/routing.py. No external dependencies -- routing.py is
pure Python, so this needs no fakes at all."""

from __future__ import annotations

import pytest

from src import routing


def test_agreement_not_overridden():
    result = routing.decide(pred_lang="ceb", text_lang="ceb")
    assert result.final_lang == "ceb"
    assert result.was_overridden is False


def test_disagreement_text_lid_wins():
    result = routing.decide(pred_lang="ceb", text_lang="fil")
    # text LID's label wins on disagreement -- final_lang is text_lang, not pred_lang
    assert result.final_lang == "fil"
    assert result.pred_lang == "ceb"
    assert result.text_lang == "fil"
    assert result.was_overridden is True


def test_empty_transcript_text_lang_none_not_overridden():
    # lid_text.TextLIDResult.lang is None for an empty transcript -- that is
    # not a disagreement, it's an absence of a signal.
    result = routing.decide(pred_lang="eng", text_lang=None)
    assert result.final_lang == "eng"
    assert result.was_overridden is False


@pytest.mark.parametrize("lang", routing.KNOWN_LANGS)
def test_all_known_langs_pass_through_unchanged(lang):
    result = routing.decide(pred_lang=lang, text_lang=lang)
    assert result.final_lang == lang


def test_unknown_pred_lang_raises():
    with pytest.raises(ValueError):
        routing.decide(pred_lang="tgl", text_lang="ceb")


def test_unknown_text_lang_raises():
    with pytest.raises(ValueError):
        routing.decide(pred_lang="ceb", text_lang="tgl")


def test_conversation_routing_locks_final_lang_from_first_turn():
    conv = routing.ConversationRouting.from_first_turn(pred_lang="fil", text_lang="eng")
    assert conv.final_lang == "eng"
    assert conv.was_overridden is True

    # the lock is just a stored value -- nothing re-derives it from later input
    assert conv.result.pred_lang == "fil"
    assert conv.result.text_lang == "eng"
