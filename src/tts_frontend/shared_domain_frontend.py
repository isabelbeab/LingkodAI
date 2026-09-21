"""
Shared-class synthesis frontend: handles the token classes that turned out to
be near-identical across CEB, FIL and (partially) ENG in the 2026-08-12
corpus scan (rag_corpus_scan.py) and vocabulary overlap EDA
(pld_rag_vocab_overlap.py, NB7).

STATUS: sandbox. NOT promoted to lingkod_tts_utils.py. NOT wired into
verbalize_ceb / verbalize_fil / NB1 / NB3. Same status as verbalize_ceb_prc
and ceb_frontend_apply. Promotion needs explicit confirmation per the
project's module discipline.

SCOPE: this module handles ORTHOGRAPHIC / FORMATTING classes that are
language-agnostic in shape (parentheticals, markdown, dashes, slashes, URLs,
initialisms, the -tion tail). It does NOT handle digit verbalization (money,
time, cardinals) -- that stays in verbalize_ceb / verbalize_fil / the
CEB_DIGIT_READINGS discipline, on purpose, because those need real listening
evidence per value and mixing them in here would blur two different
evidence tiers.

REUSE, NOT REIMPLEMENTATION: the -tion -> -syon rule is imported from
ceb_respell_lab.py (CEB_RESPELL_RULES / apply_rules), not redefined here.
That file is the promotion candidate for word-level rules; this file is the
promotion candidate for corpus-structural transforms. Keeping them separate
files with one importing the other avoids a second copy of the rule logic
drifting out of sync, which is exactly the failure mode the project's
"never copy-paste across notebooks" rule exists to prevent.

EVIDENCE TIERS, stated per component, not left implicit:
- paren_conditional_strip: MECHANICAL, no linguistic judgment, safe by
  construction (see rationale in the function docstring).
- strip_markdown_emphasis: MECHANICAL, safe.
- normalize_dashes: HYPOTHESIS. Mechanism 2 (2026-08-11 listening findings)
  found unexplained instability in pause length for space-separated letter
  names, and this reuses that unresolved space-vs-hyphen question for
  clause-boundary dashes too. NOT validated by ear. Ships with hyphen as
  default but the choice is a placeholder, not a decision -- see
  OPEN_DECISIONS at the bottom of this file.
- spell_out_initialism: SEPARATOR DEFAULT REVERSED AGAIN, 2026-08-13
  (second reversal same day). Round 1 (54 clips, 6 words) found
  no-separator > hyphen > space; default was set to "". Later the same
  day, round 0/batch (CPA, DST, BIR, USB, PQF) found the opposite for
  THESE words: BIR "bi-ay-ar" and USB "yu-es-bi" (both hyphenated) heard
  as PERFECT; CPA's no-separator "sipiey" was unstable/wonky across
  occurrences; DST's no-separator "diesti" read too fast. HK's own
  standing call, stated directly: hyphen gives more consistent pause
  timing across letter boundaries, worth the (apparently small) cost
  case-by-case. Default changed from "" back to "-".
  CONSEQUENCE, FLAGGED NOT HIDDEN: PIC, PHP, PRB, CPD, HCPC have no
  explicit RESPELL_OVERRIDES entry, so they silently pick up the new
  hyphenated default -- e.g. PIC changes from the round-1-WINNING
  "piaysi" to the round-1-LOSING "pi-ay-si". Round 1 explicitly found
  no-separator beat hyphen FOR THESE FIVE WORDS specifically. Nothing
  about tonight's BIR/USB/CPA/DST evidence re-tests these five with the
  new default -- their hyphenated form has never been heard. This is a
  real, live gap: the policy is well-motivated by newer words but
  UNCONFIRMED for the original round-1 set, and there is a documented
  chance they sound worse under it. Worth a quick re-listen before
  trusting it, not assumed fine by extrapolation. PRC is unaffected
  either way -- its RESPELL_OVERRIDES entry ("piearsee") is a literal
  string, not generated from this default, and overrides it regardless.
  Letter NAMES remain HYPOTHESIS beyond PRC and BIR/USB: round-2
  cross-tested "pie ar see"/"piearsee" against "pi ar si"/"piarsi" for
  PRC specifically and the override won 2 of 3 carriers, now reflected
  in RESPELL_OVERRIDES. PQF's hyphenated algorithm output ("pi-kyu-ef")
  was heard bad regardless of separator -- HK reports jitter/robotic
  audio or silence specifically at "ef" -- and needs a letter-level
  override, not a separator fix; "pi-kyu-ep" proposed, not yet heard.
  FIL letter names are NOT assumed identical to CEB -- see
  OPEN_DECISIONS. NEW cross-word finding (round-2, not yet in
  OPEN_DECISIONS below because it needs its own writeup): isolated-word
  carrier (carrier_idx 2) shows a consistent P-to-M/B consonant confusion
  across all six words tested ("pi"->"mi", "HCPC"->"HCBC", etc), on top of
  already being the weakest carrier by listening-quality mean score (1.67
  vs 4.56/4.78 for the sentence carriers). Never test or ship an isolated-
  word acronym utterance; always embed in a carrier phrase.
- handle_slash / handle_url: HYPOTHESIS, placeholder wording, lowest
  measured priority (605 and 85 occurrences respectively, against 1636 for
  vowel-free initialisms), not yet worth a listening pass.
"""

from __future__ import annotations
import re
import json
import collections
from typing import Dict, List, Tuple

from ceb_respell_lab import CEB_RESPELL_RULES, apply_rules as apply_word_rules


# ---------------------------------------------------------------------------
# 1. Parenthetical handling -- CONDITIONAL, not blanket.
#
# Blanket suppression was the plan going into this build. Checked against
# the real corpus first (2026-08-12): 701 parenthetical spans across the
# Google-translated CEB column contain an acronym or word that appears
# NOWHERE else in that row's text. Examples: "(SPA)", "(US CDR)", "(TOR)",
# "(ACEND)", "(REPS)" -- first-and-only mentions. Blanket suppression would
# make those acronyms permanently unrecoverable from the synthesized audio,
# not just less prosodic. So this strips a parenthetical ONLY when every
# alphabetic token inside it is a duplicate of something already present
# outside it in the same text -- which is exactly the "tulo (3)",
# "and/or"-gloss, and repeated-acronym cases the original suppression
# decision was aimed at, without the collateral loss.
# ---------------------------------------------------------------------------

_PAREN = re.compile(r"\(([^)]{1,200})\)")
_WORD = re.compile(r"[A-Za-z]+")


def paren_conditional_strip(text: str) -> Tuple[str, List[str]]:
    """Strip a parenthetical span only if it introduces no new alphabetic
    token relative to text with EVERY parenthetical removed. Returns
    (new_text, kept_spans) so a caller can inspect what survived and why.

    BUGS FOUND AND FIXED DURING TESTING (2026-08-12), both against the real
    553-row corpus, not a synthetic case:
    (1) Comparing each span against "the rest of the text" (i.e. other
        parens still present) let two parens that both introduce the same
        acronym mutually treat each other as the bare occurrence, so BOTH
        got stripped and the acronym vanished entirely -- 11/200 rows hit
        this. Fixed by computing one bare-of-all-parens baseline up front.
    (2) Case-insensitive word comparison let a markdown-style link
        "[online.prc.gov.ph](https://...)" (row 348/368) count as a bare
        occurrence of the acronym "PRC", because the lowercase "prc" inside
        the URL domain matched case-insensitively. That is a coincidental
        substring, not a real restatement of the acronym. Fixed by requiring
        an EXACT case match for all-caps inner tokens (len >= 2) specifically,
        while keeping case-insensitive comparison for ordinary words."""
    bare = _PAREN.sub("", text)
    bare_words_ci = {w.lower() for w in _WORD.findall(bare)}
    bare_words_exact = set(_WORD.findall(bare))
    kept = []

    def repl(m: re.Match) -> str:
        span = m.group(0)
        inner_tokens = _WORD.findall(m.group(1))
        new_found = False
        for w in inner_tokens:
            if w.isupper() and len(w) >= 2:
                if w not in bare_words_exact:
                    new_found = True
                    break
            elif w.lower() not in bare_words_ci:
                new_found = True
                break
        if new_found:
            kept.append(span)
            return span  # leave untouched, this paren carries unique content
        return ""

    result = _PAREN.sub(repl, text)
    # Collapse any double space left by a removed span.
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result, kept


# ---------------------------------------------------------------------------
# 2. Markdown emphasis. MECHANICAL, safe: 631 occurrences of ** in all three
# languages (rag_corpus_scan finding), inherited from Francis's English
# source formatting, never semantically load-bearing.
# ---------------------------------------------------------------------------

_MD_EMPH = re.compile(r"\*{1,2}")


def strip_markdown_emphasis(text: str) -> str:
    return _MD_EMPH.sub("", text)


# ---------------------------------------------------------------------------
# 3. Dash normalization. HYPOTHESIS -- see module docstring and
# OPEN_DECISIONS. 762-810 en-dash occurrences per language, used by Google
# Translate as a clause separator (e.g. "Action Sheet – kuhaa kini gikan sa
# PRC"). Currently these are invisible to the MMS tokenizer (not in its
# vocabulary) so the clause boundary is silently dropped across most of the
# corpus. Some mapping is clearly better than none; WHICH mapping is
# unconfirmed.
# ---------------------------------------------------------------------------

_DASH = re.compile(r"[\u2013\u2014\u2012\u2015]")

# PLACEHOLDER, not a decision. MMS has no punctuation in its vocabulary but
# does tokenize space and hyphen (per the project's confirmed architectural
# ceiling). Hyphen chosen as default only because Mechanism 2 flagged SPACE
# as the likely driver of excessive/unstable pauses in letter-name spellouts,
# so avoiding another space-based separator is the more conservative guess
# -- not because hyphen has been heard and confirmed for this purpose.
DASH_REPLACEMENT = "-"


def normalize_dashes(text: str, replacement: str = DASH_REPLACEMENT) -> str:
    return _DASH.sub(replacement, text)


# ---------------------------------------------------------------------------
# 4. Slash handling. HYPOTHESIS, placeholder wording, lowest priority.
# 605-598 occurrences but "/" is confirmed absent from the CEB tokenizer
# vocabulary (2026-08-11 demo notebook finding), so SOME transform is needed
# even as a stopgap. Two shapes observed in the corpus:
#   (a) semantic: "ug/o" (Bea's own CEB rendering of "and/or"), "and/or"
#   (b) formatting: "iya / iyang", "sheet/billing" -- word-boundary slash
#       with no semantic "or"
# This module does NOT try to distinguish (a) from (b) automatically -- that
# is a real linguistic judgment call, not a regex-safe one, and guessing
# wrong silently changes meaning. Default behaviour: replace "/" with a
# space, which is always SAFE (never wrong prosody-wise, just merges two
# words that were meant to be alternatives) but not necessarily CORRECT
# for the semantic cases. An override table is provided for the one
# confirmed corpus-native form.
# ---------------------------------------------------------------------------

SLASH_OVERRIDES: Dict[str, Dict[str, str]] = {
    # lang: {lowercased raw form: replacement}
    # "ug/o" is Bea's own CEB coinage, not a formatting artifact -- worth
    # keeping literal rather than routing through the generic space-fallback.
    "ceb": {"ug/o": "ug o"},
    "fil": {"at/o": "at o"},
}


def handle_slash(text: str, lang: str) -> str:
    overrides = SLASH_OVERRIDES.get(lang, {})
    for raw, repl in overrides.items():
        text = re.sub(re.escape(raw), repl, text, flags=re.IGNORECASE)
    # Fallback for everything else: space. Safe, not necessarily correct.
    return text.replace("/", " ")


# ---------------------------------------------------------------------------
# 5. URL / email handling. HYPOTHESIS, placeholder wording, lowest measured
# priority (85 URL occurrences, 3 emails). Not worth a listening pass yet
# given the initialism and -tion backlogs are two orders of magnitude
# larger. Replaces with a language-tagged placeholder token rather than
# attempting to read the URL aloud, which would be actively wrong (reading
# "https://acoas.prc.gov.ph" letter by letter serves no one).
# ---------------------------------------------------------------------------

_URL = re.compile(r"(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# PLACEHOLDER WORDING, unconfirmed by ear.
URL_PLACEHOLDER = {"ceb": "sa website", "fil": "sa website", "eng": "on the website"}
EMAIL_PLACEHOLDER = {"ceb": "sa email", "fil": "sa email", "eng": "by email"}

# SUPERSEDED, 2026-08-21: EMAIL_PLACEHOLDER no longer used (see
# verbalize_email below). Kept for reference/provenance, not deleted.

def _email_to_speech(match):
    """Reads an email as literal chunks with 'dot'/'at' connectors, e.g.
    'prcncr.fad@gmail.com' -> 'prcncr dot fad at gmail dot com'. STANDING
    DECISION, 2026-08-21, per HK's direct instruction: real spoken
    reading, not a placeholder. Language-agnostic on purpose, same
    reasoning as this session's English-unification decision for
    digits/money/reference-codes."""
    email = match.group(0)
    local, domain = email.split("@", 1)
    return f"{local.replace('.', ' dot ')} at {domain.replace('.', ' dot ')}"


def verbalize_email(text: str) -> str:
    """Shared across all three languages. Called from handle_url() below
    (reaches CEB via apply_shared_frontend), and directly from
    verbalize_fil_prc / verbalize_eng_prc (2026-08-21)."""
    return _EMAIL.sub(_email_to_speech, text)



def handle_url(text: str, lang: str) -> str:
    text = _URL.sub(URL_PLACEHOLDER.get(lang, "sa website"), text)
    text = verbalize_email(text)
    return text


# ---------------------------------------------------------------------------
# 6. Initialism spell-out. HYPOTHESIS beyond "pie ar see" for PRC.
#
# Letter-name table. CEB candidates per ceb_frontend_reference_2026-08-11.md.
# FIL is a COPY of the CEB table, NOT a confirmed Filipino reading -- see
# OPEN_DECISIONS. Filipino and Cebuano orthographic conventions are related
# but not identical (e.g. historically different treatment of "e"/"i" and
# "o"/"u" mergers), so this is a real open question, not a safe assumption.
# ---------------------------------------------------------------------------

CEB_LETTER_NAMES: Dict[str, str] = {
    "a": "ey",  "b": "bi",  "c": "si",  "d": "di",  "e": "i",
    "f": "ef",  "g": "dyi", "h": "eyts", "i": "ay", "j": "dyey",
    "k": "key", "l": "el",  "m": "em",  "n": "en",  "o": "o",
    "p": "pi",  "q": "kyu", "r": "ar",  "s": "es",  "t": "ti",
    "u": "yu",  "v": "vi",  "w": "dobolyu", "x": "eks", "y": "way", "z": "zi",
}

# COPIED from CEB, NOT independently confirmed for Filipino. Flagged.
FIL_LETTER_NAMES: Dict[str, str] = dict(CEB_LETTER_NAMES)


# ADDED 2026-08-21: standard English alphabet pronunciation. Unlike
# CEB_LETTER_NAMES (Tim's informant evidence) and FIL_LETTER_NAMES
# (unconfirmed copy of CEB), this is near-universal, low-uncertainty
# knowledge, not a dialectal judgment call. Still worth a listening
# spot-check before treating as final, but does not carry the same
# evidentiary weight as the CEB/FIL tables.
ENG_LETTER_NAMES: Dict[str, str] = {
    "a": "ay",   "b": "bee",  "c": "see",  "d": "dee",  "e": "ee",
    "f": "eff",  "g": "jee",  "h": "aitch","i": "eye",  "j": "jay",
    "k": "kay",  "l": "el",   "m": "em",   "n": "en",   "o": "oh",
    "p": "pee",  "q": "kyoo", "r": "are",  "s": "ess",  "t": "tee",
    "u": "yoo",  "v": "vee",  "w": "double u", "x": "ex", "y": "why", "z": "zee",
}

LETTER_NAMES: Dict[str, Dict[str, str]] = {"ceb": CEB_LETTER_NAMES, "fil": FIL_LETTER_NAMES, "eng": ENG_LETTER_NAMES}

# CONFIRMED 2026-08-13 (round-2 listening, single voice, 6 words x 3 carriers,
# crossing separator and letter-naming explicitly for PRC). "piearsee" beat
# its space-separated predecessor "pie ar see" in 2 of 3 carriers and tied
# closely in the third; combined with the corpus-wide separator finding
# below, this clears the two-or-three-clips bar. See
# lingkod_tts_lessons_learned.md, 2026-08-13 entry, for the full listening
# table this rests on.
RESPELL_OVERRIDES: Dict[str, Dict[str, str]] = {
    "ceb": {"prc": "piearsee"},
    "fil": {},
}

_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")

# Function words that happen to be all-caps in headings/lists; excluding
# these from spell-out prevents e.g. "SA" (Cebuano "in/at") from being read
# as an initialism. Kept deliberately small -- see rag_corpus_scan.py for
# the fuller rationale on why an over-broad stop list is itself a risk.
STOP_INITIALISM = {
    "OF", "AND", "THE", "TO", "FOR", "OR", "IN", "ON", "AT", "BY", "IS", "AS",
    "BE", "AN", "IF", "NO", "ALL", "ANY", "ONE", "TWO", "NEW", "PER", "NOT",
    "MAY", "CAN", "SEE", "USE", "WHO", "HIS", "HER", "ITS", "ARE", "WAS",
    "II", "III", "IV", "AM", "PM",
    # "US" REMOVED 2026-08-13: was excluded as a generic English stopword
    # without checking it against this corpus. Real bug, not a judgment
    # call -- 17 corpus occurrences, all the country abbreviation ("US
    # Commission on Dietetic Registration", "US CDR"), never a stray
    # function word. This is why it silently passed through unspelled
    # rather than being read as an initialism. AM/PM correctly stay
    # excluded: they have dedicated handling in verbalize_ceb_time and
    # spelling them as initialisms would conflict with that.
    "SA", "ANG", "UG", "NGA", "MGA", "KAY", "NI", "SI", "KUNG", "NA", "PARA",
    "AY", "ISIP", "UPOD", "AT",
}


def build_initialism_set(json_path: str, fields=("answer_ceb", "answer_fil"),
                         min_count: int = 1) -> collections.Counter:
    """Regenerate the initialism inventory directly from the live corpus
    rather than trusting a hardcoded list to stay current. Returns a
    Counter so caller can see real frequency, not just presence."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    counter = collections.Counter()
    for field in fields:
        for row in data:
            for tok in _TOKEN.findall(row[field]):
                if tok.isupper() and len(tok) >= 2 and tok not in STOP_INITIALISM:
                    counter[tok] += 1
    return collections.Counter({k: v for k, v in counter.items() if v >= min_count})


def _spell_out_token(token: str, letter_names: Dict[str, str],
                     separator: str) -> str:
    letters = [ch for ch in token.lower() if ch.isalpha()]
    return separator.join(letter_names.get(ch, ch) for ch in letters)


def spell_out_initialisms(text: str, lang: str,
                          initialism_set: set,
                          letter_separator: str = "-") -> str:
    """Convert known initialisms to a letter-name spellout. Anything not in
    initialism_set passes through unchanged -- this function never guesses
    at whether an all-caps token is a real initialism, that judgment is the
    caller's, made once via build_initialism_set / manual curation."""
    letter_names = LETTER_NAMES.get(lang, CEB_LETTER_NAMES)
    overrides = RESPELL_OVERRIDES.get(lang, {})

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        low = tok.lower()
        stripped = tok.rstrip("s")  # handle CPAs / PICs style English plurals
        if tok not in initialism_set and stripped not in initialism_set:
            return tok
        if low in overrides:
            return overrides[low]
        base = stripped if stripped in initialism_set else tok
        spelled = _spell_out_token(base, letter_names, letter_separator)
        if tok.endswith("s") and stripped == base:
            spelled += "s"  # crude plural marker, unconfirmed by ear
        return spelled

    return _TOKEN.sub(repl, text)


# ---------------------------------------------------------------------------
# 7. -tion rule. REUSED from ceb_respell_lab, not redefined. Confirmed
# productive on 4 words by ear (2026-08-11); corpus-scale application is
# 1960 CEB / 1705 FIL occurrences across 60/75 distinct words -- scale is
# measured, correctness at scale is NOT re-confirmed by this reuse.
# ---------------------------------------------------------------------------

def apply_tion_rule(text: str) -> Tuple[str, List[str]]:
    """Apply CEB_RESPELL_RULES word-by-word. Returns (new_text, words_changed)
    for auditability -- same reasoning as apply_rules() in ceb_respell_lab:
    knowing WHICH words the rule touched matters when something sounds wrong
    downstream."""
    changed = []

    def repl(m: re.Match) -> str:
        word = m.group(0)
        new, fired = apply_word_rules(word)
        if fired:
            changed.append(word)
        return new

    result = re.sub(r"\b[A-Za-z]+\b", repl, text)
    return result, changed


# ---------------------------------------------------------------------------
# Pipeline. Order matters and is stated explicitly, not left implicit:
# 1. paren strip (may remove digit-in-parens collisions, acronym dupes)
# 2. markdown strip
# 3. URL / email (before slash, since URLs contain slashes that would
#    otherwise get mangled by the generic slash fallback)
# 4. slash handling
# 5. dash normalization
# 6. -tion rule (word-internal, order-independent relative to 1-5)
# 7. initialism spellout (LAST, so it does not spell out letters that are
#    about to be stripped as markdown or consumed inside a URL)
#
# Digit verbalization (verbalize_ceb / verbalize_fil) is NOT called here.
# Run this pipeline first, then the language verbalizer, as two separate
# steps -- mixing them would blur the evidence-tier boundary documented in
# the module docstring.
# ---------------------------------------------------------------------------

def apply_shared_frontend(text: str, lang: str, initialism_set: set,
                          verbose: bool = False) -> Dict[str, object]:
    """Run the full shared-class pipeline. Returns a dict with the final
    text plus per-stage diagnostics, so a caller can see what fired without
    re-deriving it -- same transparency contract as ceb_frontend_apply's
    verbose change log."""
    trace: Dict[str, object] = {"input": text}

    text, kept_parens = paren_conditional_strip(text)
    trace["kept_parens"] = kept_parens

    text = strip_markdown_emphasis(text)
    text = handle_url(text, lang)
    text = handle_slash(text, lang)
    text = normalize_dashes(text)

    text, tion_changed = apply_tion_rule(text)
    trace["tion_words_changed"] = tion_changed

    text = spell_out_initialisms(text, lang, initialism_set)

    text = re.sub(r"\s{2,}", " ", text).strip()
    trace["output"] = text

    if verbose:
        print(f"[{lang}] {trace['input'][:60]!r} -> {text[:60]!r}")
    return trace


# ---------------------------------------------------------------------------
# OPEN DECISIONS -- not guessed, surfaced here so they are not lost.
#
# 1. Dash replacement (space / hyphen / something else): STILL untested by
#    ear. Ships with hyphen. NOTE: the 2026-08-13 separator finding resolved
#    space-vs-hyphen-vs-none for LETTER-NAME spellouts specifically (see
#    spell_out_initialism above); it did NOT test dash-as-clause-separator,
#    which is a different mechanism (word-boundary punctuation, not
#    letter-boundary). Do not assume the initialism result transfers here
#    without its own listening pass.
# 2. FIL_LETTER_NAMES is a literal copy of CEB_LETTER_NAMES. Whether
#    Filipino initialisms should be read with Cebuano letter-sound
#    approximations or need their own table is an open linguistic question,
#    not yet asked of anyone. The 2026-08-13 rounds were CEB-only.
# 3. Slash and URL placeholder wording is invented, not sourced from Tim,
#    Bea, or any listening pass. Lowest priority given corpus frequency
#    (605 and 85 respectively vs 1636 for vowel-free initialisms) but still
#    a real gap if the domain tier ships before it's addressed.
# 4. The plural-acronym handling in spell_out_initialisms (stripping a
#    trailing "s" before matching) is a crude heuristic that will misfire
#    on any real initialism that happens to end in S (e.g. "REPS", "SEC"
#    variants). Not cross-checked against the corpus's real S-ending
#    initialism list.
# 5. NEW (2026-08-13): isolated-word carrier (carrier_idx 2, i.e. the
#    acronym synthesized with no surrounding sentence) shows a consistent
#    P-to-M/B consonant confusion across all six words tested in round-2,
#    and was already the weakest carrier by RTF anomaly (round-1) and by
#    listening-quality mean score (round-2: 1.67 vs 4.56/4.78 for the two
#    sentence carriers). This is corpus-scan-relevant: verbalized text that
#    could end up synthesized as a bare word (e.g. a one-word RAG answer
#    field, or a naive per-token TTS call) should be flagged and avoided,
#    not just initialism spellouts specifically. Not yet generalized beyond
#    initialisms because that's all round-2 tested.
# 6. Residual "flows too fast into the next word" complaint, no-separator
#    condition, sentence carriers (0/1), 8 of ~18 relevant round-2 notes.
#    No-separator fixed the between-LETTER pause problem from the 2026-08-11
#    findings but appears to have introduced or left unaddressed a
#    boundary-level pause problem AFTER the spelled-out acronym, going into
#    the next word of the carrier. Not isolated to a specific word; treat as
#    a general property of the no-separator spellout, not a per-word defect.
# ---------------------------------------------------------------------------

# --- R-C blend fix, 2026-08-23, confirmed by listening. See lessons
# file 2026-08-22/23 entry for full test history. PRC + CPD tested in
# all three languages. PWD + CU tested in CEB only, do not extend to
# FIL/ENG without a real listening pass. ---
RESPELL_OVERRIDES.setdefault("ceb", {})["prc"] = "piear, see"
RESPELL_OVERRIDES.setdefault("fil", {})["prc"] = "pi-ar, si"
RESPELL_OVERRIDES.setdefault("eng", {})["prc"] = "pee-are, see"
RESPELL_OVERRIDES.setdefault("ceb", {})["cpd"] = "si-pi-di"
RESPELL_OVERRIDES.setdefault("fil", {})["cpd"] = "si-pi-di"
RESPELL_OVERRIDES.setdefault("eng", {})["cpd"] = "see, pee, dee"
RESPELL_OVERRIDES.setdefault("ceb", {})["pwd"] = "pi, dobolyu, di"
RESPELL_OVERRIDES.setdefault("ceb", {})["cu"] = "si, yu"

# --- ASEAN fix, 2026-08-23, PROVISIONAL, not listening-confirmed for
# FIL/ENG. CEB confirmed good as plain passthrough tonight; using the
# same literal word as an override for FIL/ENG as the lower-risk guess,
# not an invented respelling. Root cause of FIL/ENG ignoring the passed
# initialism_set argument is still undiagnosed, this only papers over
# ASEAN specifically. Needs a real listening pass before treating as
# settled. ---
RESPELL_OVERRIDES.setdefault("ceb", {})["asean"] = "ASEAN"
RESPELL_OVERRIDES.setdefault("fil", {})["asean"] = "ASEAN"
RESPELL_OVERRIDES.setdefault("eng", {})["asean"] = "ASEAN"

# --- PRC fix CORRECTION, 2026-08-23. Previous entries (piear,see /
# pi-ar,si / pee-are,see) were tail-comma-only, a leftover from BEFORE
# the CPD finding established the real rule: ALL initialism letter
# boundaries get commas, no partial hyphen-comma mixing. This was an
# inconsistency, not a re-test, correcting it now. Standing rule for
# any future initialism fix: full comma separation at every letter
# boundary, always, no exceptions. ---
RESPELL_OVERRIDES.setdefault("ceb", {})["prc"] = "pie, ar, see"
RESPELL_OVERRIDES.setdefault("fil", {})["prc"] = "pi, ar, si"
RESPELL_OVERRIDES.setdefault("eng", {})["prc"] = "pee, are, see"

# --- PRC fix, FINAL, 2026-08-23. Confirmed by direct listening on the
# real DS2 EXT-001 row after two prior attempts (tail-comma only, then
# full-comma-throughout) both left an audible P-R/R-C asymmetry. This
# form keeps full internal hyphenation (matches the letter-name spelling
# already used elsewhere) and adds a single trailing comma after the
# whole spelled-out word, rather than a comma between individual
# letters. This line runs AFTER the two earlier PRC assignments further
# up this file (2026-08-23 original, 2026-08-23c correction); Python
# executes top to bottom, so this is the value that actually takes
# effect, those earlier lines are superseded, not deleted. Clean up the
# stale duplicate assignments in a non-deadline pass, they are harmless
# but confusing to read. ---
RESPELL_OVERRIDES.setdefault("ceb", {})["prc"] = "pie-ar-see,"
RESPELL_OVERRIDES.setdefault("fil", {})["prc"] = "pi-ar-si,"
RESPELL_OVERRIDES.setdefault("eng", {})["prc"] = "pee-are-see,"
