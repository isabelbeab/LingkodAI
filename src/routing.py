"""Stage 3 -- Routing decision and conversation language lock.

Audio LID (stage 1) runs on the first turn's audio and picks the ASR
checkpoint for every turn, which routes the audio to the corresponding
STT model.

Text LID (src/lid_text.py) runs on the first turn's transcript and wins on
disagreement: `final_lang` locks to its label and the disagreement is
recorded as `was_overridden`. ASR is not re-run; the override applies from
MT in onward.

The decision is made only on the first turn of each conversation and is
used for the rest of the turns.
"""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_LANGS = ("ceb", "fil", "eng")


@dataclass
class RoutingResult:
    """The one-time stage 3 decision, made from turn 1's audio LID label and
    text LID label."""

    pred_lang: str
    text_lang: str | None
    final_lang: str
    was_overridden: bool


def decide(pred_lang: str, text_lang: str | None) -> RoutingResult:
    """Text LID's label wins on disagreement with audio LID; the disagreement
    is recorded via `was_overridden`. `text_lang` of None (an empty
    transcript, see lid_text.TextLIDResult) never counts as a disagreement,
    and `final_lang` falls back to `pred_lang`."""
    if pred_lang not in KNOWN_LANGS:
        raise ValueError(f"Unknown pred_lang {pred_lang!r}; expected one of {KNOWN_LANGS}")
    if text_lang is not None and text_lang not in KNOWN_LANGS:
        raise ValueError(f"Unknown text_lang {text_lang!r}; expected one of {KNOWN_LANGS} or None")

    was_overridden = text_lang is not None and text_lang != pred_lang
    return RoutingResult(
        pred_lang=pred_lang,
        text_lang=text_lang,
        final_lang=text_lang if was_overridden else pred_lang,
        was_overridden=was_overridden,
    )


@dataclass
class ConversationRouting:
    """Locks a conversation's language from the first turn's RoutingResult.

    MT in, RAG, MT out and TTS read `final_lang` from here for every turn.
    ASR does not: every turn uses the checkpoint audio LID picked on the
    first turn. LID decides the conversation's language exactly once."""

    result: RoutingResult

    @property
    def final_lang(self) -> str:
        return self.result.final_lang

    @property
    def was_overridden(self) -> bool:
        return self.result.was_overridden

    @classmethod
    def from_first_turn(cls, pred_lang: str, text_lang: str | None) -> "ConversationRouting":
        return cls(result=decide(pred_lang, text_lang))
