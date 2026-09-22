"""
ceb_prc_digit_verbalizer.py

Ported 2026-08-13 from NB_DEMO_ceb_prc_frontend.ipynb (2026-08-11 build) into
a standalone module, so it can be imported by NB8 and any future notebook
instead of living only inside one demo notebook.

STATUS UNCHANGED BY THIS PORT: still "subject to change" per HK, still
deliberately NOT promoted into lingkod_tts_utils.py, still NOT wired into
NB1 or NB3. This is a location change (notebook cell -> importable file),
not a promotion decision. The reasoning for keeping it separate from the
shared verbalize_ceb is unchanged and preserved verbatim in the docstring
below: verbalize_ceb is a real NB1/NB3 dependency (trainable-subset
filtering, round-trip eval row exclusion) against PLD text, which has a
materially different vocabulary (ages, barangay numbers, native time-of-day
system) that this function does not and should not attempt to cover.

PIPELINE ORDERING REQUIREMENT (found 2026-08-13, confirmed by direct test,
not guessed): this function MUST run BEFORE shared_domain_frontend.py's
apply_shared_frontend, never after. Reason: apply_shared_frontend's
initialism spell-out treats "PHP" as a known initialism (it is high-
frequency in the real corpus and gets pulled into build_initialism_set
automatically). If spell_out_initialisms runs first, "PHP 250" becomes
"pieytspi 250" before verbalize_ceb_money ever sees the literal string
"PHP" it needs to match. The money construction silently never fires,
and the bare digit falls through to the small-cardinal pass, which raises
for the wrong reason (or silently misreads it as a count, for values 1-7).
This is the same collision this file's own money-verbalizer docstring
already anticipated from a different, never-implemented source (a sketched
php->pesos letters-pass mapping in the old frontend reference doc); this
is a second, independent instance of the same failure mode, arising from
shared_domain_frontend.py existing at all, which postdates this file.

Confirmed by direct test 2026-08-13: verbalize_ceb_prc is a no-op (returns
input unchanged, never raises) on any text containing no digits, so it is
always safe to call unconditionally as the first pipeline stage, regardless
of whether a given row needs digit handling at all.
"""

# DRAFT, deliberately kept SEPARATE from lingkod_tts_utils.py's shared
# verbalize_ceb (2026-08-11 decision). The shared verbalize_ceb is a real
# dependency of NB1 (trainable-subset filtering) and NB3 (round-trip eval
# row exclusion), both already-reported pipelines; expanding it here would
# silently change which PLD rows count as trainable and which eval rows
# get included, re-litigating the already-reported 0.224 CEB headroom
# number without saying so. This file's top-level function is named
# verbalize_ceb_prc, NOT verbalize_ceb, specifically so it can never be
# imported or monkeypatched over the shared one by accident.
#
# Scope of verbalize_ceb_prc: PRC/RAG-domain digit patterns only (clock
# times with explicit AM/PM, PHP amounts, list-item ordinals, small native
# cardinals). Does NOT touch CEB_DIGIT_READINGS and does NOT attempt to
# cover PLD's crime-news register (ages, barangay numbers, dates, colon-
# less times, native buntag/udto/hapon/gabii/kadlawon time-of-day system)
# surfaced by the NB1/NB3 grep on 2026-08-11 -- that's a materially
# different vocabulary problem and stays out of scope here on purpose.
# This stays "still subject to change" per HK (2026-08-11): not promoted
# to TTS_M2/lingkod_tts_utils.py, not wired into NB1 or NB3, PRC/demo
# pipeline only until a deliberate integration decision gets made.
#
# Evidence source: corpus scan of ceb_audio_clean.csv (56,426 rows,
# transcript_clean column), 2026-08-11. See lessons file entry for the
# raw counts. Coverage below is deliberately narrow: only readings with
# real corpus support are filled in. Everything else still raises, which
# is the point, not a gap to be embarrassed about.
#
# NOT covered, and this is intentional:
#   - Money / PHP amounts: 24 total hits across gatos/libo/milyon/piso in
#     56K rows is absence of evidence, not evidence of a reading. Do not
#     guess a hundreds/thousands construction. Waiting on Bea's six-sentence
#     numbers-as-words pass, which will give real evidence for the specific
#     amounts that actually occur (PHP 75, PHP 1240, etc.), not an inferred
#     general rule.
#   - Cardinals 8 and above: 'walo' had zero hits, 'siyam' one, 'napulo'
#     two. Too thin to confirm. Raises.
#   - Ordinal 4th and beyond: 'ikaupat' had one hit. Raises.
#   - Any Spanish-derived reading as a default: the native/Spanish split in
#     the scan was closer than the raw totals suggest once 'usa' (which
#     doubles as the general determiner 'one/a', not just the cardinal) is
#     discounted. Native forms are used below because they were still ahead
#     net of that discount, but this is the weaker of the two evidence-backed
#     claims in this file and worth a listening check before the demo, not
#     just a corpus-count check.

from __future__ import annotations
import re
from fil_prc_digit_verbalizer import _english_cardinal_extended, _digit_by_digit_english
from typing import Optional

# Cardinals 1-7. Corpus-attested (native forms), 'usa' discounted for its
# determiner double-duty but still net-supported. 8-10 deliberately absent.
CEB_CARDINALS: dict[int, str] = {
    1: "usa",
    2: "duha",
    3: "tulo",
    4: "upat",
    5: "lima",
    6: "unom",
    7: "pito",
}

# Ordinals for list-item markers (1., 2., 3. -> spoken ordinal). Stops at 3
# because 'ikaupat' had a single corpus hit, below the two-or-three-clips
# confirmation bar applied to the READING itself (separate from the
# two-or-three-CLIPS listening rule, but same spirit: don't finalize on one
# data point).
CEB_ORDINALS: dict[int, str] = {
    1: "una",
    2: "ikaduha",
    3: "ikatulo",
}

# NOTE ON THE ORIGINAL BULLET ABOVE: 'alas otso' is now directly confirmed
# by Bea's translation (2026-08-11), not just corpus-inferred. See the
# _ALAS_HOUR / _AMPM_WORD block below for the current, AM/PM-aware version.

# Clock-hour word, no AM/PM baked in (see below for buntag/hapon handling).
_ALAS_HOUR = {
    1: "alas una", 2: "alas dos", 3: "alas tres", 4: "alas kwatro",
    5: "alas singko", 6: "alas sais", 7: "alas siyete", 8: "alas otso",
    9: "alas nuwebe", 10: "alas diyes", 11: "alas onse", 12: "alas dose",
}
# NOTE: the hour NAMES here (dos, tres, kwatro...) are Spanish-derived,
# riding on the fixed 'alas' construction, which is standard Filipino/
# Cebuano clock-time convention and NOT the same claim as "Spanish-derived
# cardinals are the default reading" above. 'alas otso' itself is
# corpus-supported (otso: 14 hits, consistent with 8:00 appearing 84 times
# in the eval set) AND now directly confirmed by Bea's translation of
# sentence 1 (2026-08-11): "alas otso sa buntag ... alas singko sa hapon"
# for 8:00 AM / 5:00 PM. The others (uno, tres, kwatro...) are still
# extrapolated from the same fixed construction, not independently
# confirmed per hour.

# AM -> buntag, PM -> hapon. CONFIRMED for 8 AM / 5 PM by the same
# translation. NOT yet confirmed for evening hours: Cebuano commonly uses
# "gabii" for night, and "hapon" may not hold much past 5-6 PM. This table
# covers the two times actually present in the eval corpus; extending past
# roughly 6 PM without a listening check would be a guess.
_AMPM_WORD = {"AM": "sa buntag", "PM": "sa hapon"}

_TIME_RE = re.compile(
    r"\b(?:alas\s+)?(\d{1,2}):(\d{2})\s*(?:(AM|PM|am|pm)|(sa buntag|sa hapon))?\b"
)


def verbalize_ceb_time(text: str) -> str:
    """Convert clock times to spoken Cebuano, including the AM/PM
    disambiguator (sa buntag / sa hapon), confirmed by Bea's translation
    of sentence 1. Only handles on-the-hour and half-hour (:00 and :30);
    anything else raises. A disambiguator is REQUIRED, since a bare hour
    with no period is ambiguous and there's no corpus basis for a default.

    Accepts the disambiguator in EITHER of two forms: a literal English
    AM/PM marker (Tim's original demo-sentence format), or the Cebuano
    words "sa buntag"/"sa hapon" already sitting in the source text
    (2026-08-13: found to be how Google Translate renders Bea's CEB
    column for the "PRC office hours" boilerplate answer, e.g. "alas 8:00
    sa buntag" with no literal AM anywhere -- 82 real corpus rows hit
    this exact shape). Both forms resolve to the identical output; this
    is not two different readings, it's the same confirmed buntag=AM /
    hapon=PM equivalence recognized from either direction. No new
    linguistic claim is introduced by accepting the Cebuano form: the
    equivalence itself was already established by Tim's translation, and
    the same hour-range caveat that already applied to English AM/PM
    (confirmed only for 8/5, "sa buntag"/"sa hapon" past roughly 6 PM
    likely gives way to "gabii" and is unconfirmed either way) applies
    unchanged here.

    After conversion, strips a redundant parenthetical digit echo that
    immediately follows (e.g. "alas otso sa buntag (8:00 AM)" -> "alas
    otso sa buntag"). This is a synthesis-side authoring choice, not a
    transcription-accuracy call: we control what the assistant says, so a
    digit gloss that only repeats what was just spoken is dropped rather
    than spoken twice. This is NOT the same as the STT project's
    parenthetical-stripping caution (where the parenthetical was
    sometimes the only thing actually recorded and stripping it lost real
    signal) -- that concern applies to transcribing existing audio, not to
    authoring new text for synthesis, so it doesn't transfer here.

    The strip runs BEFORE conversion, not after: once "8:00 AM" has been
    converted to "alas otso sa buntag" the parenthetical no longer
    contains a digit pattern to match against, so stripping first (while
    the digits are still digits) is the only order that works. Confirmed
    by running this against Bea's actual sentence 1 translation, which is
    where this bug was caught.
    """
    # Pre-strip: a parenthetical digit-time immediately after an
    # ALREADY-PRESENT "sa buntag"/"sa hapon" in the SOURCE text (not
    # something we produced) is redundant and dropped before conversion.
    text = re.sub(
        r"\b(sa (?:buntag|hapon))\s*\(\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*\)",
        r"\1",
        text,
    )

    def repl(m):
        hour, minute = int(m.group(1)), m.group(2)
        ampm_literal, ceb_phrase = m.group(3), m.group(4)
        if hour not in _ALAS_HOUR:
            raise ValueError(f"verbalize_ceb_time: hour {hour} out of 1-12 range: {m.group(0)!r}")
        if ampm_literal is not None:
            ampm_word = _AMPM_WORD.get(ampm_literal.upper())
        elif ceb_phrase is not None:
            # Already exactly "sa buntag" or "sa hapon" -- the match SPAN
            # includes this text, so using it directly (rather than
            # reconstructing via _AMPM_WORD) both produces the right
            # output and consumes the source phrase, avoiding duplication.
            ampm_word = ceb_phrase.lower()
        else:
            raise ValueError(
                f"verbalize_ceb_time: no AM/PM or sa buntag/sa hapon given, "
                f"reading is ambiguous between buntag and hapon, no corpus "
                f"default exists: {m.group(0)!r}"
            )
        base = _ALAS_HOUR[hour]
        if minute == "00":
            out = f"{base} {ampm_word}"
        elif minute == "30":
            # UPDATED 2026-08-11 per native-speaker informant Tim: "medya"
            # (from Spanish "media", half), not "kag tunga" as originally
            # guessed. Spelling "medya" is Tim's own, tentative -- he
            # wasn't certain of the orthography either. Still ONE
            # informant, text only, not yet heard. The corpus scan found
            # "medya" once and "media" once (2 total), too thin to have
            # settled this alone; Tim's confirmation is what makes this
            # usable, not the corpus count.
            out = f"{base} medya {ampm_word}"
        else:
            raise ValueError(
                f"verbalize_ceb_time: minute value {minute!r} has no corpus-backed "
                f"reading, only :00 and :30 are covered: {m.group(0)!r}"
            )
        return f" {out} "

    text = _TIME_RE.sub(repl, text)
    return re.sub(r"[ \t]+", " ", text).strip()


# --- Money: Spanish-derived reading, per native-speaker informant "Tim"  ---
# (2026-08-11, text only, NOT yet heard). Tim's stated priority order for
# spoken PRC-context prices: Spanish numerals first (e.g. "sitenta'y
# singko" for 75), English spoken price second, direct Cebuano translation
# third/least common. This is ONE informant's text confirmation, same
# evidentiary tier as the "pie ar see" PRC respelling: below the
# two-or-three-clips rule, not yet listening-verified, do not cite in the
# manuscript as settled. Only covers 1-99 (tens + ones). Amounts of 100+
# (PHP 900, PHP 1,250, PHP 3,000 all appear in the real RAG answer dump)
# are NOT covered here and still raise: whether those use "libo" combined
# with Spanish tens or a different construction is an open question for
# Tim, not something to infer from the 1-99 pattern.

_SPANISH_TENS = {
    10: "diyes", 20: "beinte", 30: "treinta", 40: "kwarenta",
    50: "singkwenta", 60: "sisenta", 70: "sitenta", 80: "otsenta",
    90: "nobenta",
}
_SPANISH_ONES = {
    1: "uno", 2: "dos", 3: "tres", 4: "kwatro", 5: "singko",
    6: "sais", 7: "siyete", 8: "otso", 9: "nwebe",
}
# NOTE 2026-08-11: 9 corrected from "nuebe" to "nwebe" per Tim's own
# spelling in his hundreds examples ("nwebe syentos" for 900). Tim's
# spelling takes precedence over my earlier guess.


def verbalize_ceb_money_spanish(n: int) -> str:
    """Spanish-style tens+ones reading for 1-99, e.g. 75 -> "sitenta'y
    singko". UNVERIFIED BY EAR. Raises above 99, deliberately: no basis
    yet for the hundreds/thousands construction."""
    if not (1 <= n <= 99):
        raise ValueError(
            f"verbalize_ceb_money_spanish: only 1-99 covered (informant "
            f"confirmation was for a two-digit amount), got {n}. "
            f"Hundreds/thousands construction not yet confirmed with Tim."
        )
    tens, ones = (n // 10) * 10, n % 10
    if tens == 0:
        return _SPANISH_ONES[ones]
    if ones == 0:
        return _SPANISH_TENS[tens]
    return f"{_SPANISH_TENS[tens]}'y {_SPANISH_ONES[ones]}"


# --- Money hundreds/thousands: extended 2026-08-11 with Tim's follow-up,
# which gave FOUR full worked examples (200, 900, 1250, 3000) in BOTH
# root systems. This is strong enough to check a general combination rule
# against, rather than trusting one extrapolated pattern. ---

def verbalize_ceb_money_spanish_full(n: int) -> str:
    """Spanish root, full range 1-9999. Construction: [X mil] [Y syentos]
    [tens'y ones], space-joined. Regression-tested to reproduce Tim's 4
    given examples EXACTLY (200, 900, 1250, 3000) -- see test cell.
    Still: ONE informant, text only, NOT YET HEARD. The exact match on
    all 4 examples is good evidence the general rule is right, but "right
    on paper" and "sounds right and natural" are different claims; this
    still owes a listening pass before it goes near production.
    Bare 100 and bare 1000 are inferred (not literally in an example
    standing alone) from standard Spanish-loan grammar (no "uno" prefix
    on mil), flagged lower confidence than the 4 exact-matched values.
    """
    if not (1 <= n <= 9999):
        raise ValueError(f"verbalize_ceb_money_spanish_full: only 1-9999 covered, got {n}")
    thousands, rem1 = divmod(n, 1000)
    hundreds, remainder = divmod(rem1, 100)
    parts = []
    if thousands:
        if thousands == 1:
            parts.append("mil")  # inferred: no "uno" prefix, standard Spanish-loan pattern
        else:
            parts.append(f"{_SPANISH_ONES[thousands]} mil")
    if hundreds:
        parts.append(f"{_SPANISH_ONES[hundreds]} syentos")  # inferred plural form for bare 100
    if remainder:
        parts.append(verbalize_ceb_money_spanish(remainder))
    if not parts:
        raise ValueError(f"verbalize_ceb_money_spanish_full: n={n} produced no parts")
    return " ".join(parts)


# --- Native root hundreds/thousands. Confirmed multiplier words: usa,
# duha, tulo, upat, lima, unom, pito (corpus + Tim), siyam (Tim, via
# "siyam ka gatos"=900). "walo" (8) is standard Cebuano and included for
# pattern completeness but is NOT directly confirmed by Tim or the
# corpus scan -- flagged distinctly, lower confidence than the rest. ---
_NATIVE_MULT = {
    1: "usa", 2: "duha", 3: "tulo", 4: "upat", 5: "lima",
    6: "unom", 7: "pito", 8: "walo", 9: "siyam",
}
# walo (8): UNCONFIRMED by any informant or corpus hit. Standard Cebuano
# word, included for completeness, needs its own listening check before
# use, same as everything else here but worth calling out twice.

# Native tens for the 1-99 remainder. ONLY "kalim-an" (50) is directly
# Tim-confirmed, via the 1250 example. The rest (napulo, kawhaan,
# katloan, kap-atan, kan-uman, kapitoan, kawaloan, kasiyaman) are the
# standard Cebuano "ka-X-an" pattern from general knowledge of the
# language, NOT informant- or corpus-confirmed. Treat as a weaker claim
# than "kalim-an" specifically.
_NATIVE_TENS = {
    10: "napulo", 20: "kawhaan", 30: "katloan", 40: "kap-atan",
    50: "kalim-an", 60: "kan-uman", 70: "kapitoan", 80: "kawaloan",
    90: "kasiyaman",
}


def verbalize_ceb_money_native(n: int) -> str:
    """Native root, full range 1-9999. Construction and connector rule
    ('g contraction after libo, plain ug elsewhere) copied EXACTLY from
    Tim's 1250 example, not invented: "usa ka libo'g duha ka gatos ug
    kalim-an". Regression-tested to reproduce all 4 given examples.
    Tens+ones combination within the 1-99 remainder (e.g. 45 = "kap-atan
    ug lima") uses uncontracted "ug" deliberately: Tim's example only
    showed a bare tens word (kalim-an, no ones attached), so any
    contraction of tens+ones together (Cebuano sometimes elides these,
    e.g. a form like "kawha'g lima" for 25) is NOT something I have
    confirmation for and I'm not going to guess at the elision spelling.
    Uncontracted "ug" is the safe, if possibly less idiomatic, choice.
    """
    if not (1 <= n <= 9999):
        raise ValueError(f"verbalize_ceb_money_native: only 1-9999 covered, got {n}")
    thousands, rem1 = divmod(n, 1000)
    hundreds, remainder = divmod(rem1, 100)
    tens, ones = (remainder // 10) * 10, remainder % 10

    def ones_word(d):
        if d not in _NATIVE_MULT:
            raise ValueError(f"verbalize_ceb_money_native: no native ones word for {d}")
        return _NATIVE_MULT[d]

    def remainder_word():
        if tens and ones:
            return f"{_NATIVE_TENS[tens]} ug {ones_word(ones)}"
        if tens:
            return _NATIVE_TENS[tens]
        return ones_word(ones)

    parts = []
    if thousands:
        parts.append(f"{_NATIVE_MULT[thousands]} ka libo")
    if hundreds:
        if parts:
            parts[-1] = parts[-1] + "'g"  # contraction, exact form from Tim's example
        parts.append(f"{_NATIVE_MULT[hundreds]} ka gatos")
    if remainder:
        rw = remainder_word()
        parts.append(f"ug {rw}" if parts else rw)
    if not parts:
        raise ValueError(f"verbalize_ceb_money_native: n={n} produced no parts")
    return " ".join(parts)


# --- The actual PHP-amount detector and text-level converter. Regex
# validated 2026-08-11 against every distinct "PHP ..." substring in
# Francis's real RAG answer dump (65 distinct strings, values up to
# PHP 6,500, well inside the 1-9999 range covered above). Confirmed by
# that scan: comma and non-comma thousands formatting BOTH occur for the
# same magnitude (e.g. "PHP 3,000" and "PHP 3000" both appear), so the
# regex accepts either. Every decimal in the real data is ".00"; no
# other cents value occurs, so raising on non-.00 cents costs nothing in
# practice against this corpus and stays honest about what's covered.
_MONEY_RE = re.compile(r"(?:\u20b1|\b(?:PHP|PhP|Php|P))\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?\b")


def verbalize_ceb_money(text: str, *, root: str = "english") -> str:
    """Find "PHP <amount>" and replace with a spoken Cebuano reading plus
    "pesos" (Tim's own convention: "...singkwenta pesos"). Consumes the
    "PHP" marker itself, not just the digits -- see the ordering note in
    verbalize_ceb's docstring for why this matters and what it deliberately
    conflicts with in the frontend reference file's sketched design.

    root: "spanish" (default, Tim's stated first-priority system, the one
    recommended for the Thursday demo) or "native". Deliberately a
    required, explicit choice rather than a silent default buried in
    behavior, since which root to use is a real open product decision
    (Tim: Spanish/English skews urban, native skews far-province), not
    something this function should decide on its own.

    Raises on non-.00 cents (no confirmed reading exists) and on any
    amount outside 1-9999 (out of range for both money verbalizers).
    """
    # STANDING DECISION, 2026-08-20: default is now "english", per HK's
    # direct instruction to make CEB money consistent with FIL's own
    # PRC-tier convention, overriding Tim's Spanish-first preference for
    # PRC prices specifically. "spanish"/"native" kept, not removed, for
    # comparison. English also raises the range cap to 999,999 (was 9999
    # under spanish/native), closing the two PHP-over-9999 rows.
    fn = {"spanish": verbalize_ceb_money_spanish_full,
          "native": verbalize_ceb_money_native,
          "english": _english_cardinal_extended}.get(root)
    if fn is None:
        raise ValueError(f"verbalize_ceb_money: root must be 'spanish', 'native', or 'english', got {root!r}")

    def repl(m):
        raw_amount, cents = m.group(1), m.group(2)
        if cents is not None and cents != "00":
            raise ValueError(
                f"verbalize_ceb_money: non-zero centavos have no confirmed "
                f"reading: {m.group(0)!r}"
            )
        n = int(raw_amount.replace(",", ""))
        words = fn(n)
        return f" {words} pesos "

    text = _MONEY_RE.sub(repl, text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", text)  # drop space before punctuation


# ---------------------------------------------------------------------------
# US Dollar amounts. ADDED 2026-08-20, per HK's direct instruction, after
# finding "US$50.00" (APEC Architect registration fee rows) was falling
# through to the bare-cardinal path with no currency context at all --
# same silent-mismatch shape as the original PHP-ordering bug, different
# currency. "US" read as hyphenated letters ("U-S"), not the English
# pronoun "us" -- matches shared_domain_frontend's existing hyphen-
# separated initialism convention (BIR/USB/CPA/DST listening evidence),
# not a new reading style. Scope: only the exact corpus-confirmed form
# "US$<amount>.00" is matched -- no evidence for a bare "$" shape anywhere
# else in the corpus, so none is guessed at.
_USD_RE = re.compile(r"\bUS\s*\$\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?\b")


def verbalize_ceb_usd(text: str) -> str:
    """Find "US$<amount>" and replace with "U-S <english cardinal> Dollars".
    Raises on non-.00 cents, same as verbalize_ceb_money -- no confirmed
    reading exists for centavos/cents in this corpus."""
    def repl(m):
        raw_amount, cents = m.group(1), m.group(2)
        if cents is not None and cents != "00":
            raise ValueError(
                f"verbalize_ceb_usd: non-zero cents have no confirmed "
                f"reading: {m.group(0)!r}"
            )
        n = int(raw_amount.replace(",", ""))
        words = _english_cardinal_extended(n)
        return f" U-S {words} Dollars "

    text = _USD_RE.sub(repl, text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


# ---------------------------------------------------------------------------
# Native counting construction: "[number] ka <noun>". ADDED 2026-08-13,
# STANDING DECISION per HK (2026-08-13 chat): native root, not Spanish.
#
# EVIDENCE: 2026-08-13 listening round, 8 real corpus values (40, 20, 10,
# 25, 15, 35, 13, 9) x 3 carriers (two "ka minuto"/"ka adlaw" sentence
# carriers plus an isolated-word control), 23 rated variant clips: 16
# perfect, 4 almost perfect, 2 decent-good, 1 bad (value 20, carrier 2,
# the isolated-word carrier -- consistent with that carrier's already-
# documented weakness, RTF anomaly and consonant confusion in two earlier
# rounds, not a new problem). Raw-digit controls: 9/10 bad, confirming the
# model cannot speak bare digits at all, which is the whole reason this
# function exists. Clears the two-or-three-clips bar.
#
# WHY NATIVE, NOT SPANISH, EVEN THOUGH MONEY USES SPANISH: this was a real
# question raised and explicitly decided, not overlooked. Tim's Spanish-
# root preference (verbalize_ceb_money_spanish_full) was stated specifically
# for PRC PRICES; he was never asked about counting durations, and applying
# a money-register answer to a different register would be exactly the kind
# of unearned cross-register generalization this project's two-or-three-
# clips discipline exists to prevent. Separately, Bea's own translated
# corpus already writes this construction with a native root ("40 ka
# minuto", not "kwarenta ka minuto" or "40 minutos") -- native root matches
# what the translator of record actually produced, not just what tested
# well by ear. STANDING DECISION, not permanently closed: if mentors or
# the sponsor prefer the Spanish-root alternative, that comparison has not
# been built or heard and would need its own listening round before being
# trusted equally -- see the 2026-08-13 lessons entry for the full
# rationale trail.
#
# SCOPE: the NUMBER reading (kap-atan, kawhaan, etc) is confirmed by ear
# for the 8 tested values, in "ka minuto" and "ka adlaw" contexts
# specifically. Cebuano numerals do not inflect for the noun they count,
# so extending the confirmed number-reading to any "ka <noun>" (tuig,
# panid, units, examinees, ...) is a reasonable grammatical generalization,
# not a new claim about the number itself -- but the NOUN contexts beyond
# minuto/adlaw were not independently heard. Flagged, not hidden.
#
# Reuses verbalize_ceb_money_native directly: despite the name, it returns
# pure native number words with no currency semantics (the "pesos" suffix
# is added by verbalize_ceb_money's wrapper, not inside this function), so
# it already is the general native-number-to-words converter this needs.
# ---------------------------------------------------------------------------

_KA_COUNT_RE = re.compile(r"\b(\d{1,4})\s*ka\s+(\w+)")


def verbalize_ceb_counting(text: str) -> str:
    """Convert '[number] ka <noun>' to a native-root spoken reading, e.g.
    '40 ka minuto' -> 'kap-atan ka minuto'. Raises via
    verbalize_ceb_money_native's own range check (1-9999) if the value is
    out of range; leaves anything not matching the ka-construction alone."""
    def repl(m):
        n = int(m.group(1))
        noun = m.group(2)
        native = verbalize_ceb_money_native(n)
        return f"{native} ka {noun}"
    return _KA_COUNT_RE.sub(repl, text)


def _cardinal_or_raise(n: int, context: str) -> str:
    if n in CEB_CARDINALS:
        return CEB_CARDINALS[n]
    if n == 0:
        # ADDED 2026-08-29, per HK's explicit instruction: DS6/DS7 rows
        # with malformed or absent currency markers should still
        # synthesize, so any resulting RAG/MT text-quality issue is
        # audible and diagnostic rather than silently blocked. Reuses the
        # Spanish-derived "sero" already established for CEB's digit-by-
        # digit reference-code reading elsewhere in this file
        # (_CEB_SPANISH_DIGIT), not a newly invented word. NOT
        # independently listening-confirmed for a bare-cardinal "zero"
        # context specifically, below the project's own two-or-three-clips
        # bar, applied anyway by explicit standing decision, not silently
        # promoted to "confirmed."
        return "sero"
    # STANDING DECISION, 2026-08-20: cardinals 8+ route through
    # verbalize_ceb_money_native, the same native-number converter already
    # listening-confirmed for "[number] ka <noun>" (2026-08-13, 16/23
    # perfect, 4/23 almost perfect). Same evidence basis, extended to bare
    # cardinal position.
    if 1 <= n <= 9999:
        return verbalize_ceb_money_native(n)
    raise ValueError(
        f"verbalize_ceb: no corpus-backed CEB cardinal reading for {n} "
        f"(1-7 literal, 8-9999 via native fallback). Context: {context[:80]!r}"
    )


# ---------------------------------------------------------------------------
# Redundant number-word echo stripper. ADDED 2026-08-13, evidence-free by
# construction: this never introduces a new reading, it only removes a
# digit gloss that restates a number ALREADY spoken as a word in the
# source text (same category as verbalize_ceb_time's existing parenthetical
# strip for "sa buntag (8:00 AM)", generalized beyond time).
#
# Found via corpus scan: 7 real rows fail purely because of this pattern,
# e.g. "napulo ka porsyento (10%)" (the word "napulo" is already there;
# "(10%)" adds nothing) and "siyam (9) ka mga opisyal" (same, word first,
# parenthetical digit second). Verified safe by requiring the parenthetical
# digit to EXACTLY match the value of the number word already spoken --
# if they don't match, the text is left untouched rather than guessed at.
# ---------------------------------------------------------------------------

_WORD_TO_VALUE: dict[str, int] = {}
for _src in (CEB_CARDINALS, _NATIVE_MULT, _NATIVE_TENS):
    for _val, _word in _src.items():
        _WORD_TO_VALUE.setdefault(_word, _val)

_REDUNDANT_ECHO_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in _WORD_TO_VALUE), key=len, reverse=True)) + r")\b"
    r"((?:\s+ka\s+\w+)?)"          # optional " ka <noun>" between word and echo, e.g. "napulo ka porsyento (10%)"
    r"\s*\(\s*(\d{1,3})\s*%?\s*\)"
)


def strip_redundant_number_echo(text: str) -> str:
    """Drop a parenthetical digit that only restates a number word already
    spoken immediately before it (optionally across a 'ka <noun>' phrase).
    Leaves the text untouched if the parenthetical digit does not match the
    word's value -- never guesses which reading is right when they disagree."""
    def repl(m):
        word, mid, digit = m.group(1), m.group(2), m.group(3)
        if _WORD_TO_VALUE.get(word) == int(digit):
            return f"{word}{mid}"
        return m.group(0)
    return _REDUNDANT_ECHO_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Reference-code detector. ADDED 2026-08-13. Document/form/annex codes like
# "Annex A sa 2025-35", "IAO-QRD-33", "RSD Form No. 8" are not countable
# quantities -- reading "35" as a cardinal count in "2025-35" is a category
# error, not a missing-evidence gap. This does NOT attempt to fix or speak
# these codes (how a voice assistant should render a form number out loud
# is a real product decision, not something to guess at); it only makes
# the raise say what's actually wrong.
#
# NOT a preventive fix -- an ACTIVE bug it caught on first real corpus run.
# An earlier verification pass claimed zero reference-code digits fall in
# CEB_CARDINALS' 1-7 range, checking only single-digit codes directly after
# a hyphen. That check was wrong: it missed zero-padded two-digit codes.
# Once this detector was wired in and run against the real 553-row corpus,
# it caught two real, already-shipping-quality bugs: row 111 ("Form
# ACD-RES-03") was being silently read as "ACD-RES-tulo" (the cardinal
# three), and row 313 ("Form PRD-06") as "Form PRD-unom" (six) -- wrong
# audio with no error raised, the exact failure mode this module's whole
# raise-by-default design exists to prevent. Both rows move from
# "silently wrong, counted as passing" to "correctly blocked, pending a
# real decision" -- a net DROP of 2 in the raw pass-count, but a real
# correctness gain, not a regression. Confirmed by diffing exact passing-
# row sets with and without this check, not by re-eyeballing the corpus.
# ---------------------------------------------------------------------------

# EXTENDED 2026-08-13: two more reference-code shapes found in the
# categorization pass, neither caught by the original hyphen/No. pattern.
# (1) Bare "RA <number>" (Republic Act references without a "No." prefix,
#     e.g. "RA 5734") -- same category as the already-caught
#     "Republic Act (RA) No. 8981", just missing the "No." token.
# (2) "<number>-<word>" adjectival compounds (e.g. "16-digit", a common
#     English-loan shape) -- the digit describes a property of something
#     else, not a countable quantity, and was previously invisible to the
#     detector because it only checked for a LEADING hyphen, not a
#     trailing one.
# ---------------------------------------------------------------------------
# "No. N" reference numbers: SPOKEN, not blocked. STANDING DECISION per HK
# (2026-08-13): "Form No. 8" -> "Form Number walo", keeping whatever
# precedes "No." untouched (Form, Resolution, Republic Act, Board
# Resolution, ...) and converting the digit via the same native-root
# reading just confirmed for counting. "Number" kept in English per HK's
# own wording of the decision, not translated to "Numero" -- not assumed,
# taken directly from how the decision was stated.
#
# NOTED, NOT BUILT: HK flagged that sponsors may prefer a pure English
# fallback ("Form Number Eight") instead. Not implemented now -- recorded
# here so it isn't lost, and so a future "we need the English version"
# request has a clear pointer to where it plugs in (this function, a
# second reading in an English-cardinal branch, gated the same way
# root="spanish"/"native" gates verbalize_ceb_money).
#
# Deliberately narrower than the general reference-code detector below:
# only the "No. N" shape is spoken. Hyphenated codes (ACD-RES-03,
# IAO-QRD-33) and bare "RA <number>" without "No." are NOT covered by this
# decision and remain blocked -- HK's instruction was specific to the
# "Form No. X" shape, and reading a compact alphanumeric code the same way
# ("ACD-RES Number tulo") was not decided, not assumed safe by analogy.
# ---------------------------------------------------------------------------

_NO_NUMBER_RE = re.compile(r"\bNo\.\s*(\d{1,4})\b")


def verbalize_ceb_reference_number(text: str) -> str:
    """'Form No. 8' -> 'Form Number eight'. Only the 'No. N' token changes;
    whatever precedes it is left exactly as-is. STANDING DECISION,
    2026-08-20: switched native -> English, per HK's direct instruction --
    this is the "pure English fallback" flagged NOTED, NOT BUILT above
    when this function was first written 2026-08-13."""
    def repl(m):
        n = int(m.group(1))
        return f"Number {_english_cardinal_extended(n)}"
    return _NO_NUMBER_RE.sub(repl, text)


# Remaining reference-code shapes: BLOCKED until 2026-08-19, now SPOKEN.
# "No. N" removed from this pattern since verbalize_ceb_reference_number
# handles it first.
_REFERENCE_CODE_RE = re.compile(
    r"-(\d{1,4})\b"                    # leading hyphen, e.g. "ACD-RES-03"
    r"|\bRA\s+(\d{1,5})\b"             # Republic Act number, no "No."
    r"|\b(\d{1,3})-[a-zA-Z]+\b"        # adjectival compound, e.g. "16-digit"
)


def _contains_reference_code(text: str) -> bool:
    return bool(_REFERENCE_CODE_RE.search(text))


# ---------------------------------------------------------------------------
# STANDING DECISION, 2026-08-19: reference-code digits are now SPOKEN,
# digit-by-digit, Spanish-derived, rather than blocked outright. This does
# NOT reopen the original bug (row 313, "PRD-06" silently misread as the
# CARDINAL 'unom'/six): these digits are never routed through the cardinal
# system, only through this dedicated digit-by-digit reading, so the
# semantic-category error the detector was built to prevent still can't
# happen. 2-9 reuse the exact spellings already vetted in _ALAS_HOUR above
# (dos, tres, kwatro, singko, sais, siyete, otso, nuwebe), not re-derived.
# "sero" (0) and "uno" (1) are NEW, not otherwise evidenced in this file.
# Low-risk relative to this project's other digit decisions, since zero and
# a bare "one" (distinct from the time-specific "una") have no documented
# native/Spanish ambiguity anywhere in this project's CEB evidence base,
# but not corpus-confirmed either. Flag for a listening check before any
# reported number depends on it.
# ---------------------------------------------------------------------------
_CEB_SPANISH_DIGIT = {
    "0": "sero", "1": "uno", "2": "dos", "3": "tres", "4": "kwatro",
    "5": "singko", "6": "sais", "7": "siyete", "8": "otso", "9": "nuwebe",
}


def _speak_digits_individually(digit_str: str) -> str:
    return " ".join(_CEB_SPANISH_DIGIT[d] for d in digit_str)


def _reference_code_digits_to_speech(match: "re.Match") -> str:
    """Speaks the digit portion of a reference code digit-by-digit, in
    English. STANDING DECISION, 2026-08-20: switched from Spanish-derived
    digit words to English (fil_prc_digit_verbalizer._digit_by_digit_english),
    per HK's direct instruction. Reference codes are labels, not counted
    quantities, so there was no register argument for Spanish the way
    there was for Tim's money preference."""
    if match.group(1) is not None:          # "-NN" shape
        return " " + _digit_by_digit_english(match.group(1))
    if match.group(2) is not None:          # "RA NNNNN" shape
        return "RA " + _digit_by_digit_english(match.group(2))
    if match.group(3) is not None:          # "NN-word" shape
        digits = match.group(3)
        rest = match.group(0)[len(digits):]         # e.g. "-digit"
        return _digit_by_digit_english(digits) + " " + rest.lstrip("-")
    raise AssertionError("no group matched, regex/replacement out of sync")


_LIST_MARKER_RE = re.compile(r"^\s*(\d+)\.\s+")
_BARE_DIGIT_RUN = re.compile(r"(?<!\d)(\d{1,2})(?!\d)(?!\s*\.)")
# 2026-08-11: added the trailing (?!\s*\.) after the sentence-4 demo bug
# (see lessons file / conversation record). A digit immediately followed
# by a period is ambiguous between "genuine sentence-final count" and "an
# orphaned list marker that split_for_synthesis failed to catch" -- the
# latter is common when input text has no real newlines (informal
# translations, chat-pasted text) and the sentence-splitter treats "1."
# as its own sentence, stranding the numeral. Excluding period-adjacent
# digits here means that case RAISES instead of silently converting to
# the wrong reading (a counting cardinal where an ordinal was meant).
# Trade-off: a genuine bare cardinal at true sentence-end (rare in this
# domain) will also raise and need manual handling. That's the safer
# failure mode given the alternative is shipping wrong audio unnoticed.


# ---------------------------------------------------------------------------
# Redundant money-echo stripper. ADDED 2026-08-27, DS6/DS7 finding: real
# multi-turn RAG answers spell a money amount out fully in CEB words and
# then restate it in a trailing parenthetical decimal, e.g. "... Kap-atan
# ug Lima ka Pesos (245.00)". Same evidence-free-by-construction category
# as strip_redundant_number_echo above (introduces no new reading, only
# deletes a confirmed-redundant digit echo), extended to a multi-word
# compound money phrase rather than a single number word. Deliberately
# does NOT cross-check the parenthetical value against the spoken amount
# (unlike strip_redundant_number_echo's exact-match check): parsing a
# multi-word native-root amount back into an integer is a separate,
# harder problem. Scoped narrowly, fires only directly after the literal
# word "Pesos", to keep the blast radius small.
# ---------------------------------------------------------------------------
_MONEY_ECHO_RE = re.compile(r"(\bPesos)\s*\(\s*\d{1,3}(?:,\d{3})*\.\d{2}\s*\)")


def strip_redundant_money_echo(text: str) -> str:
    """Drop a parenthetical decimal that only restates a money amount
    already spoken in full as CEB words immediately before it."""
    return _MONEY_ECHO_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# 5+ digit reference numbers. ADDED 2026-08-29, per HK's explicit
# instruction: "No. NNNNN" at 5+ digits (e.g. "Republic Act No. 10912")
# was previously silently passed through unconverted, a real gap, not a
# deliberate exclusion, the exact silent-wrong-audio failure mode this
# module's raise-by-default design exists to prevent. verbalize_ceb_
# reference_number's own pattern caps at 4 digits and cannot partially
# match a 5+ digit run (no internal word boundary), so this is a genuinely
# separate case, not a widened version of that function. Digit-by-digit
# English reading, same convention already used for every other reference
# code in this file (Form-NN, RA <number>); reading a citation number as a
# magnitude would be a category error, nobody says a law's number as "ten
# thousand nine hundred twelve."
# ---------------------------------------------------------------------------
_NO_NUMBER_LONG_RE = re.compile(r"\bNo\.\s*(\d{5,})\b")


def verbalize_ceb_reference_number_long(text: str) -> str:
    def repl(m):
        return f"Number {_digit_by_digit_english(m.group(1))}"
    return _NO_NUMBER_LONG_RE.sub(repl, text)


def verbalize_ceb_prc(text: str, *, allow_partial: bool = False) -> str:
    """PRC/RAG-domain CEB digit verbalizer. Deliberately named differently
    from and NOT a replacement for the module's shared verbalize_ceb (see
    file header: that one is a real NB1/NB3 dependency, left untouched by
    decision on 2026-08-11). Order of operations matters: list-marker
    ordinals first (context-sensitive, must run before the generic
    cardinal pass would misread the same digit), then clock times, then
    MONEY (must run before the bare-cardinal fallback -- see below for
    why), then bare small cardinals. Anything left over that still
    contains a digit raises.

    MONEY / PHP ORDERING, real bug this prevents: without a dedicated
    money pass running before the bare-cardinal fallback, "PHP 3" would
    be caught by the generic 1-2-digit cardinal regex and silently
    convert via the COUNTING system ("tulo", i.e. "three [of something]")
    instead of the MONEY system, because 3 falls inside CEB_CARDINALS'
    1-7 range. That's not a raise, it's a silently wrong reading in the
    wrong semantic register, exactly what this module's raise-by-default
    design exists to prevent. Money must claim its digits before the
    generic cardinal pass ever sees them.

    ALSO CONSUMES THE "PHP" MARKER ITSELF, not just the digits, and
    emits "<spoken amount> pesos" (matching Tim's own convention: "...
    singkwenta pesos"). This is a deliberate design choice made 2026-08-11
    to avoid a collision with the frontend reference file's separately
    sketched (not yet implemented) letters-pass mapping of php->"pesos":
    if that mapping were implemented too, and ran on either side of this
    one, a properly-formatted amount would end up read either as bare
    number words with no currency word at all, or with "pesos" said
    twice. Consuming PHP here means that letters-pass mapping should NOT
    also be implemented; "php" can stay in CEB_INITIALISMS as a fallback
    for the rare case of a bare PHP with no adjacent digits, but it
    should not additionally get a semantic php->pesos swap.

    allow_partial: if True, returns text with only the covered patterns
    converted rather than raising on the first uncovered digit. Useful for
    a coverage dry run during frontend development; should stay False in
    any real synthesis path, since a silent partial conversion is exactly
    the kind of guess this module's raise-by-default design exists to
    prevent.
    """
    original = text

    # 0. Redundant number-word echo strip (e.g. "napulo ka porsyento (10%)"
    #    -> "napulo ka porsyento"). Pure cleanup, no reading decision, safe
    #    to run before anything else -- see function docstring above.
    text = strip_redundant_number_echo(text)
    text = strip_redundant_money_echo(text)  # ADDED 2026-08-27, DS6/DS7 finding

    # 1. List-item ordinals: "1. " at line start -> ordinal word, marker consumed.
    m = _LIST_MARKER_RE.match(text)
    if m:
        n = int(m.group(1))
        if n in CEB_ORDINALS:
            text = _LIST_MARKER_RE.sub(f"{CEB_ORDINALS[n]}, ", text, count=1)
        elif not allow_partial:
            raise ValueError(
                f"verbalize_ceb: no corpus-backed CEB ordinal for list item "
                f"{n} (only 1-3 covered): {original[:80]!r}"
            )

    # 2. Clock times.
    try:
        text = verbalize_ceb_time(text)
    except ValueError:
        if not allow_partial:
            raise

    # 2.5. US Dollar amounts. MUST run before step 3 (PHP money) and step 4
    #    (bare cardinals) -- same silent-mismatch risk as the PHP-ordering
    #    bug this file's docstring already warns about, different currency.
    try:
        text = verbalize_ceb_usd(text)
    except ValueError:
        if not allow_partial:
            raise
    # 3. Money. MUST run before step 4 (bare cardinals) -- see docstring.
    try:
        text = verbalize_ceb_money(text)
    except ValueError:
        if not allow_partial:
            raise

    # 3.4. Native counting construction: "[number] ka <noun>" (minutes, days,
    #    years, pages, etc). MUST run before step 4 -- see
    #    verbalize_ceb_counting's docstring for the evidence and scope.
    try:
        text = verbalize_ceb_counting(text)
    except ValueError:
        if not allow_partial:
            raise

    # 3.45. Speak "No. N" reference numbers (standing decision, 2026-08-13
    #    -- see function docstring). MUST run before 3.5's block check, or
    #    every "Form No. 8" would still raise instead of being spoken.
    text = verbalize_ceb_reference_number_long(text)  # ADDED 2026-08-29, must run first
    text = verbalize_ceb_reference_number(text)

    # 3.5. Reference-code digits, BEFORE the bare-cardinal pass sees them,
    #    for the same reason as before (a code number in 1-7 would
    #    otherwise be silently misread as a real cardinal). As of
    #    2026-08-19 these are SPOKEN (digit-by-digit), not blocked, see
    #    _reference_code_digits_to_speech's docstring for why this is
    #    still safe against the original row-313 bug.
    text = _REFERENCE_CODE_RE.sub(_reference_code_digits_to_speech, text)

    # 4. Remaining bare small cardinals (counts, IDs, anything 3+ digits,
    #    or 8-99 falls through to here and raises).
    def repl(m):
        n = int(m.group(1))
        try:
            return _cardinal_or_raise(n, original)
        except ValueError:
            if allow_partial:
                return m.group(0)  # leave untouched, caller inspects for remaining digits
            raise
    text = _BARE_DIGIT_RUN.sub(repl, text)

    # STANDING DECISION, 2026-08-20: any digit run surviving every
    # evidence-backed pass gets a last-resort English digit-by-digit
    # reading instead of blocking the row, per HK's direct instruction to
    # reach full corpus coverage. Lowest-confidence tier in this file on
    # purpose -- no magnitude/register claim, literal digit words only --
    # appropriate for what actually lands here (orphaned reference/ID-
    # shaped numbers, e.g. document years, database IDs). Applied
    # regardless of allow_partial, so allow_partial's diagnostic value for
    # remaining-digit scans is reduced from this point on; flagged so it
    # isn't mistaken for an oversight later.
    if re.search(r"\d", text):
        text = re.sub(r"\d+", lambda m: _digit_by_digit_english(m.group(0)), text)

    if not allow_partial and re.search(r"\d", text):
        raise ValueError(
            f"verbalize_ceb: digits remain after evidence-backed conversion "
            f"passes, no reading available: {original[:80]!r}"
        )
    return text
