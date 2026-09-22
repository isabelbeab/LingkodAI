"""PRC/demo-scoped ENG verbalizer, sibling to fil_prc_digit_verbalizer.py
and ceb_prc_digit_verbalizer.py, same two-tier separation pattern (kept
separate from the canonical verbalize_eng used by NB1/NB3/PLD eval
synthesis).

ADDED 2026-08-20. Deliberately thin: English digit-words are already
English regardless of target language, so this module has no independent
evidence to gather, it reuses FIL's already-built helpers directly rather
than reimplementing them (never-reimplement discipline, same principle as
the vendored STT converters).

No Spanish branch: FIL's 2-digit Spanish reading (removed 2026-08-20, no
evidence citation) would make even less sense in ENG target text -- a
Spanish-loan number reading in an English sentence has no linguistic
motivation the way it does in FIL, so this was never even considered.
"""
import re

from shared_domain_frontend import verbalize_email, spell_out_initialisms, paren_conditional_strip, handle_slash

from fil_prc_digit_verbalizer import (
    en_number_to_words,
    _english_cardinal_extended,
    _four_digit_english,
    _digit_by_digit_english,
    _MD_LINK_RE,
    _BARE_URL_RE,
    strip_urls_placeholder,
)

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _eng_time_to_words(match):
    """On-hour and half-hour only, same discipline as CEB/FIL: raise on
    anything else rather than guess. AM/PM stays literal (already plain
    English, not digit-bearing), only the H:MM portion is converted."""
    hour, minute = int(match.group(1)), match.group(2)
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    hour_word = _english_cardinal_extended(h12)
    if minute == "00":
        return hour_word
    if minute == "30":
        return f"{hour_word} thirty"
    raise ValueError(
        f"[eng_prc] non-zero/non-half-hour minute '{match.group(0)}' "
        "has no confirmed reading yet, not safe to guess."
    )


_MONEY_RE = re.compile(r"\b(?:PHP|\u20b1|P)\s?([\d,]+(?:\.\d+)?)\b", re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r"\b\d+\b")


def _money_to_words(match: "re.Match") -> str:
    raw = match.group(1).replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        if int(frac) != 0:
            raise ValueError(
                f"[eng_prc] money amount with nonzero cents "
                f"'{match.group(0)}' not covered by this verbalizer."
            )
        raw = whole
    return f"{_english_cardinal_extended(int(raw))} pesos"


def _digit_run_to_words(match: "re.Match") -> str:
    raw = match.group(0)
    n = int(raw)
    if len(raw) == 4:
        return _four_digit_english(raw)
    if len(raw) >= 5:
        return _digit_by_digit_english(raw)
    return en_number_to_words(n)  # 1, 2, or 3 digits, plain English cardinal


def verbalize_eng_prc(text: str, initialism_set: set) -> str:
    """PRC/demo-scoped ENG verbalizer. Order mirrors verbalize_fil_prc
    exactly (load-bearing, same reasoning): emails first (2026-08-21,
    reused from shared_domain_frontend.verbalize_email), URLs next, then
    money before the generic digit pass, generic digit run last as
    catch-all. No list-marker ordinal pass and no counting-construct
    pass -- neither has an ENG-specific form (English list markers and
    counts already read fine via the plain cardinal path), so both are
    omitted rather than stubbed."""
    text = verbalize_email(text)
    text, _ = paren_conditional_strip(text)
    text = spell_out_initialisms(text, "eng", initialism_set)
    text = handle_slash(text, "eng")
    text = _TIME_RE.sub(_eng_time_to_words, text)
    text = strip_urls_placeholder(text)
    text = _MONEY_RE.sub(_money_to_words, text)
    text = _DIGIT_RUN_RE.sub(_digit_run_to_words, text)
    return text
