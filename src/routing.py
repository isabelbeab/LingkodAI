"""Stage 3 -- Routing decision and conversation language lock.

Audio LID (stage 1) picks the ASR checkpoint. This ordering is forced, not
chosen: ASR cannot run until a checkpoint is picked, and a checkpoint cannot
be picked from a transcript that does not exist yet -- so the ASR checkpoint
for turn 1 is always audio LID's choice, even under a later override.

Text LID (stage 3's own model, src/lid_text.py) wins on disagreement for
`final_lang`: when it disagrees with audio LID, the disagreement is recorded
as `was_overridden`, and `final_lang` locks to text LID's label, not audio
LID's. Text LID runs on the actual transcript, which audio LID cannot see.
ASR is not re-run under the new language.

LID decides the conversation's language exactly once, on turn 1. The decision
is locked for every later turn; a wrong first-turn call misroutes the whole
conversation by design -- there is no per-turn correction.

Decided in CHANGELOG.md, 2026-09-22: the original placeholder (audio LID
always wins) was written in Bea's absence and explicitly flagged for revisit
once she weighed in; this is that revisit.
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
    """Locks a conversation's language from turn 1's RoutingResult.

    Every stage past turn 1 -- MT, RAG, MT out, TTS, and even ASR checkpoint
    selection for turn 2 onward -- reads `final_lang` from here rather than
    re-running LID. LID decides the conversation's language exactly once."""

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
