"""
fil_prc_digit_verbalizer.py
PRC/demo-scoped FIL digit and time verbalizer. Deliberately separate from
the shared verbalize_fil in lingkod_tts_utils.py, same two-tier discipline
already established for CEB (verbalize_ceb vs verbalize_ceb_prc).

Decisions locked 2026-08-19, not corpus-confirmed, review before manuscript use:
- Time: Spanish-derived hour names, on-the-hour and half-hour only.
- Native counting constructs ("<digit> na <noun>"): route to native
  Tagalog reading, not English. Analogous to CEB's identified-but-not-yet-
  implemented "[number] ka <noun>" fix, same underlying pattern. Only
  matches consonant-final digits per Tagalog linker grammar (4, 6, 9, etc,
  "na"), NOT 1 ("isang", not "isa na"), since that form doesn't occur in
  correctly-formed Tagalog text.
- List markers ("1. ", "2. "): read as Filipino ordinals, 1-10 only.
  11+ falls through to the generic digit pass and reads as a cardinal.
- URLs: stripped entirely, placeholder behavior, not real URL reading.
- 4-digit runs: split-year English. 5+ digits: digit-by-digit English.
- Money (PHP/PhP/peso sign): English cardinal + "pesos", uses its own
  extended cardinal function (supports beyond 999), NOT the vendored
  en_number_to_words, which is capped at 0-999 by design and will raise
  on real fee amounts like "PHP 3000".
- General digit-run tiering: 2 digits -> Spanish-derived; everything else
  (1 digit, 3 digits) -> English via the vendored function.
- Loanwords: explicitly skipped. Confirmed adequate by listening.
"""
import re

from shared_domain_frontend import verbalize_email, spell_out_initialisms, paren_conditional_strip, handle_slash

import lingkod_tts_utils as tts
stt = tts._require_stt()
en_number_to_words = stt.en_number_to_words 

_SPANISH_FIL_ONES = {
    0: "sero", 1: "uno", 2: "dos", 3: "tres", 4: "kuwatro",
    5: "singko", 6: "sais", 7: "siyete", 8: "otso", 9: "nuebe",
}
_SPANISH_FIL_TEENS = {
    10: "diyes", 11: "onse", 12: "dose", 13: "trese", 14: "katorse",
    15: "kinse", 16: "disisais", 17: "disisiyete", 18: "disiotso",
    19: "disinuebe",
}
_SPANISH_FIL_TENS = {
    2: "baynte", 3: "trenta", 4: "kuwarenta", 5: "singkuwenta",
    6: "sisenta", 7: "sitenta", 8: "otsenta", 9: "nobenta",
}

_FIL_SPANISH_HOUR_NAMES = {
    1: "ala una", 2: "alas dos", 3: "alas tres", 4: "alas kuwatro",
    5: "alas singko", 6: "alas sais", 7: "alas siyete", 8: "alas otso",
    9: "alas nuebe", 10: "alas diyes", 11: "alas onse", 12: "alas dose",
}

_NATIVE_FIL_ONES = {
    1: "isa", 2: "dalawa", 3: "tatlo", 4: "apat", 5: "lima",
    6: "anim", 7: "pito", 8: "walo", 9: "siyam",
}

_FIL_ORDINALS = {
    1: "una", 2: "pangalawa", 3: "pangatlo",
    4: "pang-apat", 5: "panlima", 6: "pang-anim",
    7: "pampito", 8: "pangwalo", 9: "pansiyam", 10: "pansampu",
}

_EN_ONES_EXT = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS_EXT = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
                6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def _english_cardinal_extended(n: int) -> str:
    """English cardinal supporting beyond the vendored 0-999 cap. Needed
    specifically for money, since fee amounts like PHP 3000 are common in
    real PRC text and the vendored en_number_to_words raises above 999
    by design. Mirrors the same pattern already used in
    demo_verbalize_ceb_shortcut's _english_number_words."""
    if n < 20:
        return _EN_ONES_EXT[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS_EXT[tens] + (f"-{_EN_ONES_EXT[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = f"{_EN_ONES_EXT[hundreds]} hundred"
        return head if rest == 0 else f"{head} {_english_cardinal_extended(rest)}"
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        head = f"{_english_cardinal_extended(thousands)} thousand"
        return head if rest == 0 else f"{head} {_english_cardinal_extended(rest)}"
    raise ValueError(f"[fil_prc] number too large for this verbalizer: {n}")


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
_BARE_URL_RE = re.compile(r"https?://\S+")
# KNOWN RISK: this also matches a sentence-medial "Section 5. The ..." shape,
# not just true list markers. Acceptable for now given list markers are far
# more common in this corpus; revisit if a real misfire shows up.
_LIST_MARKER_RE = re.compile(r"\b(10|[1-9])\.(?=\s+[A-Z])")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_MONEY_RE = re.compile(r"\b(?:PHP|\u20b1|P)\s?([\d,]+(?:\.\d+)?)\b", re.IGNORECASE)
_COUNTING_NA_RE = re.compile(r"\b([1-9])(\s+na\b)")
_DIGIT_RUN_RE = re.compile(r"\b\d+\b")


def strip_urls_placeholder(text: str) -> str:
    """Placeholder, not rigorous. Drops markdown-link URLs (keeps the label
    unless the label is itself a URL) and bare URLs entirely. Runs first in
    the pipeline so URL-embedded digits never reach the digit passes."""
    def _md_replace(m):
        label = m.group(1)
        return "" if label.strip().startswith("http") else label
    text = _MD_LINK_RE.sub(_md_replace, text)
    text = _BARE_URL_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# SUPERSEDED, 2026-08-20, per HK's direct instruction: removed from the
# active dispatch chain in _digit_run_to_words below. Reason: unlike every
# other reading in this file, this function carries no evidence citation --
# no corpus count, no listening round, nothing -- the sole exception to
# this project's own standing evidentiary discipline. Kept here for
# reference/provenance only, not deleted, per the project's rule that
# superseded artifacts are archived with a labeled reason. 2-digit numbers
# now fall through to the same plain-English-cardinal path as 1- and
# 3-digit numbers (see _digit_run_to_words).
def _spanish_two_digit_fil(n: int) -> str:
    if n < 10:
        return _SPANISH_FIL_ONES[n]
    if n < 20:
        return _SPANISH_FIL_TEENS[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _SPANISH_FIL_TENS[tens]
    return f"{_SPANISH_FIL_TENS[tens]}'t {_SPANISH_FIL_ONES[ones]}"


def _four_digit_english(raw: str) -> str:
    # split-year style: "2018" -> "20" + "18" -> "twenty eighteen"
    first, second = int(raw[:2]), int(raw[2:])
    parts = [en_number_to_words(first) if first > 0 else "zero",
             en_number_to_words(second) if second > 0 else "hundred"]
    return " ".join(parts)


def _digit_by_digit_english(raw: str) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return " ".join(words[int(d)] for d in raw)


def _list_marker_to_ordinal(match: "re.Match") -> str:
    return _FIL_ORDINALS[int(match.group(1))] + "."


def _time_to_words(match: "re.Match") -> str:
    hour, minute = int(match.group(1)), match.group(2)
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    base = _FIL_SPANISH_HOUR_NAMES[h12]
    if minute == "00":
        return base
    if minute == "30":
        return f"{base} y medya"
    raise ValueError(
        f"[fil_prc] non-zero/non-half-hour minute '{match.group(0)}' "
        "has no confirmed FIL reading yet, not safe to guess."
    )


def _money_to_words(match: "re.Match") -> str:
    raw = match.group(1).replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        if int(frac) != 0:
            raise ValueError(
                f"[fil_prc] money amount with nonzero centavos "
                f"'{match.group(0)}' not covered by this verbalizer."
            )
        raw = whole
    return f"{_english_cardinal_extended(int(raw))} pesos"


def _counting_construct_to_native(match: "re.Match") -> str:
    digit = int(match.group(1))
    linker = match.group(2)  # preserves the original " na" spacing as-is
    return f"{_NATIVE_FIL_ONES[digit]}{linker}"


def _digit_run_to_words(match: "re.Match") -> str:
    raw = match.group(0)
    n = int(raw)
    # 2-digit no longer special-cased (see _spanish_two_digit_fil's
    # superseded note above) -- falls through with 1- and 3-digit to plain
    # English cardinal, consistent with CEB's own rule that short bare
    # digit runs get a real magnitude reading, digit-by-digit is reserved
    # for long/ambiguous ID-shaped runs only.
    if len(raw) == 4:
        return _four_digit_english(raw)
    if len(raw) >= 5:
        return _digit_by_digit_english(raw)
    return en_number_to_words(n)  # 1, 2, or 3 digits, plain English cardinal


def verbalize_fil_prc(text: str, initialism_set: set) -> str:
    """PRC/demo-scoped FIL verbalizer.

    Order is load-bearing and was chosen deliberately:
    0. Email addresses spoken as literal chunks (2026-08-21, reused from
       shared_domain_frontend.verbalize_email, not reimplemented), before
       everything else -- local-part could contain digits, and this must
       run before any digit pass or a mailbox number would get read as a
       bare cardinal.
    1. URLs stripped next, so URL-embedded digits never reach any digit pass.
    2. List markers before time/money, so "1." resolves as an ordinal rather
       than being consumed by a later pass.
    3. Time and money before the counting-construct pass, since both contain
       digits that must not be read as bare counts.
    4. Counting constructs before the generic digit pass, or "4 na araw"
       reads as English "four" instead of native "apat".
    5. Generic digit run last, as the catch-all fallback.
    """
    text = verbalize_email(text)
    text, _ = paren_conditional_strip(text)
    text = spell_out_initialisms(text, "fil", initialism_set)
    text = handle_slash(text, "fil")
    text = strip_urls_placeholder(text)
    text = _LIST_MARKER_RE.sub(_list_marker_to_ordinal, text)
    text = _TIME_RE.sub(_time_to_words, text)
    text = _MONEY_RE.sub(_money_to_words, text)
    text = _COUNTING_NA_RE.sub(_counting_construct_to_native, text)
    text = _DIGIT_RUN_RE.sub(_digit_run_to_words, text)
    return text