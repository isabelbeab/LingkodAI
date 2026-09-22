"""
lingkod_tts_utils.py

Shared module for the LingkodAI TTS project (Team CPT5, owner: HK).
Seeded July 17, 2026 from decisions locked in the TTS kickoff chat.

STATUS: starter module. Validated pieces are ported from lingkod_stt_utils.py
(frozen STT project) or implemented fresh against the known manifest schema.
Functions marked NotImplementedError are deliberate placeholders: build them
in a notebook sandbox first, validate, then promote here with explicit
confirmation (promotion discipline, inherited from STT).

SYNC DISCIPLINE: this file is never modified without confirming that BOTH the
project-knowledge copy and the JOJIE copy are updated in lockstep.

VENDORED DEPENDENCY: scoring-side normalization and the FIL/ENG number
converters are NOT reimplemented here. Copy the frozen lingkod_stt_utils.py
into the TTS BASE_DIR next to this module (record the copy date in the
lessons file). This module imports from it. The vendored copy is frozen:
later STT-side edits do not propagate here unless deliberately re-vendored.

METRIC CONVENTION WARNING (deliberate flip from STT): in THIS project,
corpus WER is PRIMARY and row-average WER is SECONDARY. The STT project is
the reverse. Every table this module produces labels both. Do not carry
numbers between the two projects without re-checking which convention they
were computed under.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

LANGS = ["ceb", "fil", "eng"]

# CONFIRM ON FIRST RUN: proposed TTS base directory. Print and verify in a
# pre-flight cell before any code writes here (path specs go stale; the STT
# eval-CSV naming drift is the canonical example).
DEFAULT_BASE_DIR = Path.home() / "Capstone" / "github" / "Capstone_CPT5" / "tts"

# WEEK-1 VERIFY: exact HF repo ids. The ceb checkpoint is confirmed by the
# pilot's own use; tgl and eng are expected to exist under the same MMS
# naming scheme but have not been checked on the live Hub yet.
MMS_TTS_CHECKPOINTS = {
    "ceb": "facebook/mms-tts-ceb",
    "fil": "facebook/mms-tts-tgl",   # MMS names the language tgl, not fil
    "eng": "facebook/mms-tts-eng",
}

# WEEK-1 VERIFY: exact ids from the QwenLM/Qwen3-TTS README before download.
QWEN3_TTS_BASE_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_TTS_TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# Historical reference point, NOT a metric baseline: the pre-STT pilot
# fine-tuned facebook/mms-tts-ceb on speaker 0209 (~34 usable minutes).
# Output was intelligible but warbly with uneven phrasing. Two suspects
# (data volume, aggressive punctuation-stripping normalization) were never
# isolated. See the lessons file, pilot layer.
PILOT_REFERENCE = {
    "model": "facebook/mms-tts-ceb",
    "speaker": "0209",
    "usable_minutes": 34,
    "outcome": "intelligible, warbly, uneven phrasing; suspects not isolated",
}

# Round-trip judges (decision locked July 17): the team's Whisper-medium is
# the PRIMARY judge (deployment-relevant, avoids Qwen-judging-Qwen optics);
# HK's Qwen3-ASR fine-tunes are the SECONDARY judge and the provisional
# headline if the Whisper model is not in hand by the dry run.
PRIMARY_JUDGE = "whisper-medium (team pipeline ASR)"
SECONDARY_JUDGE = "qwen3-asr-0.6b fine-tunes (HK, STT project)"

TTS_MIN_DURATION_SEC = 1.0


def project_dirs(base_dir: Path = DEFAULT_BASE_DIR) -> Dict[str, Path]:
    """Canonical output locations. Notebooks resolve THROUGH this function
    and print the results in a pre-flight cell, never hardcode paths."""
    base = Path(base_dir)
    return {
        "base": base,
        "manifests": base / "data" / "manifests_tts",
        "datasets": base / "data" / "tts_datasets",
        "models": base / "models",
        "logs": base / "logs",
        "markers": base / "run_markers",
        "synth_audio": base / "data" / "synth_audio",
        "eval_results": base / "data" / "eval_results_tts",
        "listening_kits": base / "data" / "listening_kits",
    }


# ---------------------------------------------------------------------------
# Vendored STT module (scoring normalization, number converters)
# ---------------------------------------------------------------------------

try:
    import lingkod_stt_utils as _stt
except ImportError:
    _stt = None


def _require_stt():
    if _stt is None:
        raise RuntimeError(
            "lingkod_stt_utils.py not importable. Copy the frozen STT module "
            "into the TTS BASE_DIR next to lingkod_tts_utils.py (vendored, "
            "frozen copy; record the copy date in the lessons file). Scoring "
            "normalization and the FIL/ENG number converters live there and "
            "are deliberately not duplicated here."
        )
    return _stt


def scoring_normalize(text: str, lang: str) -> str:
    """SCORING-side normalization for round-trip WER (strip diacritics,
    lowercase, per-language pipeline, year-guarded). Symmetric use only:
    both reference and judge hypothesis go through this identical function.

    Never feed this into TTS synthesis input. The synthesis-side functions
    are prep_tts_text and the verbalize_* family below, which face the
    opposite direction (keep punctuation, expand digits to words)."""
    return _require_stt().safe_language_normalize(text, lang)


def scoring_normalize_df(df: pd.DataFrame, ref_col: str, hyp_col: str,
                         lang: str) -> pd.DataFrame:
    """Adds reference_norm and hypothesis_norm via the STT project's
    apply_symmetric_normalization_safe (year-guarded, symmetric)."""
    return _require_stt().apply_symmetric_normalization_safe(
        df, ref_col, hyp_col, lang=lang)


def base_normalize(text: str) -> str:
    """Base normalization only (NFKD diacritic strip + lowercase), the
    'before normalization' side of every WER table."""
    return _require_stt().normalize_text(text)


def eval_side_normalize(text: str, lang: str, initialism_set: set) -> str:
    """ADDITIVE eval-side normalization, TTS-project-local, layered ON TOP
    of scoring_normalize_df's output. Never modifies the vendored
    apply_symmetric_normalization_safe.

    lang-aware and override-aware on purpose: the collapse step must match
    what spell_out_initialisms ACTUALLY produces for this language, not a
    guessed pattern. Mirrors that function's own logic exactly (letter
    table lookup, override check, plural handling) rather than
    reimplementing a simplified, silently-wrong version -- the first cut
    of this function built the pattern from literal acronym letters
    ("c p d") instead of the real phonetic spelling ("si pi di"), which
    silently failed to collapse CEB/FIL's spelled forms at all. Fixed
    2026-08-21."""
    import shared_domain_frontend as _sdf

    text = text.replace("(", " ").replace(")", " ")  # unpaired-safe: strip both, always
    text = re.sub(r"[.,]+", " ", text)  # ALL periods/commas, not just trailing (2026-08-21: dotted clause numbers like "3.2.4" and paren-wrapped acronyms both slipped through the trailing-only version)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()

    letter_names = _sdf.LETTER_NAMES.get(lang, _sdf.CEB_LETTER_NAMES)
    overrides = _sdf.RESPELL_OVERRIDES.get(lang, {})

    for acr in initialism_set:
        low = acr.lower()
        if low in overrides:
            spelled = overrides[low].replace("-", " ").replace(",", "")
            spelled = re.sub(r"\s+", " ", spelled).strip()
        else:
            base = acr.rstrip("s") if acr.rstrip("s") in initialism_set else acr
            spelled = " ".join(letter_names.get(ch, ch) for ch in base.lower() if ch.isalpha())
            if acr.endswith("s") and base == acr.rstrip("s"):
                spelled += "s"
        pattern = r"\b" + re.escape(spelled) + r"\b"
        text = re.sub(pattern, low, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Metrics: corpus PRIMARY, row-average SECONDARY (flip from STT, on purpose)
# ---------------------------------------------------------------------------

def _import_jiwer():
    try:
        import jiwer  # noqa: PLC0415
        return jiwer
    except ImportError as e:
        raise ImportError(
            "jiwer is required for round-trip WER. Install it in the active "
            "env (pip install jiwer); process_words is the jiwer 3.x API."
        ) from e


def tts_corpus_wer(references: List[str], hypotheses: List[str]) -> float:
    """PRIMARY metric for this project. Corpus WER weights by clip length,
    which is the fairer read of intelligibility when short PLD prompts
    (median 2-3 tokens) sit alongside long PRC FAQ sentences: on a 2-word
    clip a single error is 50 percent WER, and row-averaging would let a
    swarm of tiny clips dominate.

    SIBLING-PROJECT WARNING: the STT project's primary is row-average.
    Label every reported number with its convention."""
    jiwer = _import_jiwer()
    return jiwer.wer(list(references), list(hypotheses))


def tts_row_average_wer(references: List[str], hypotheses: List[str]) -> float:
    """SECONDARY metric here (PRIMARY in the STT project). Reported alongside
    corpus WER in every table, never silently swapped."""
    jiwer = _import_jiwer()
    vals = [jiwer.wer(r, h) for r, h in zip(references, hypotheses)]
    return float(np.mean(vals)) if vals else float("nan")


def roundtrip_wer_table(df: pd.DataFrame, ref_col: str, hyp_col: str,
                        lang: str, judge: str) -> pd.DataFrame:
    """The standard round-trip result table for one model x language x judge.

    Rows: corpus WER (PRIMARY) and row-average WER (SECONDARY).
    Columns: before normalization (base normalize only) and after
    normalization (per-language safe pipeline), side by side, never
    replacing each other.

    ref_col is the text sent to the synthesizer (post prep, pre synthesis);
    hyp_col is the judge ASR's transcript of the synthesized audio."""
    stt = _require_stt()
    base = stt.apply_symmetric_normalization(df, ref_col, hyp_col,
                                             lang="__base__")
    full = stt.apply_symmetric_normalization_safe(df, ref_col, hyp_col,
                                                  lang=lang)
    b_ref, b_hyp = list(base["reference_norm"]), list(base["hypothesis_norm"])
    a_ref, a_hyp = list(full["reference_norm"]), list(full["hypothesis_norm"])
    return pd.DataFrame({
        "metric": ["corpus WER (PRIMARY)", "row-average WER (SECONDARY)"],
        "before_norm (base only)": [
            tts_corpus_wer(b_ref, b_hyp),
            tts_row_average_wer(b_ref, b_hyp),
        ],
        f"after_norm ({lang} pipeline)": [
            tts_corpus_wer(a_ref, a_hyp),
            tts_row_average_wer(a_ref, a_hyp),
        ],
        "n_clips": [len(df)] * 2,
        "judge": [judge] * 2,
    })


# ---------------------------------------------------------------------------
# Manifest helpers (ported from STT, unchanged semantics)
# ---------------------------------------------------------------------------

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """pd.merge with validate='many_to_one' enforced (project rule)."""
    kwargs["validate"] = "many_to_one"
    return pd.merge(left, right, **kwargs)


def speaker_id_str(speaker_id) -> str:
    """Manifest SpeakerID is an integer; filenames use zero-padded 4 digits."""
    return str(int(speaker_id)).zfill(4)


def speaker_from_filename(filename: str) -> str:
    """Extract the speaker token from a wav filename (split on '.')."""
    return Path(filename).name.split(".")[0]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# TTS-grade trainable subset and speaker selection
# ---------------------------------------------------------------------------

def prompt_type_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Full prompt_type census with counts and total minutes. Run this and
    READ it before deciding drop_prompt_types; the exact strings for digit
    prompts and other non-read-speech categories must come from the live
    manifest, not from memory or docs."""
    inv = (df.groupby("prompt_type", dropna=False)
             .agg(n_rows=("prompt_type", "size"),
                  minutes=("duration_sec", lambda s: s.sum() / 60.0))
             .sort_values("n_rows", ascending=False)
             .reset_index())
    inv["minutes"] = inv["minutes"].round(2)
    return inv


def tts_trainable_subset(df: pd.DataFrame, drop_prompt_types: set,
                         min_duration_sec: float = TTS_MIN_DURATION_SEC,
                         verbose: bool = True) -> pd.DataFrame:
    """The canonical TTS-grade filter (pilot lesson: build this ONCE, reuse
    everywhere, and compute all per-speaker durations on ITS output).

    drop_prompt_types has NO default on purpose: it must be an explicitly
    reviewed set taken from prompt_type_inventory on the live manifest.
    Passing an empty set is allowed but must be a decision, not an accident.

    Drops, in order: rows whose audio file is missing, rows flagged no-speech,
    spontaneous rows (transcript is the prompt question, audio is a free-form
    answer: unusable for TTS), the reviewed drop_prompt_types (digit prompts
    and similar non-read-speech), sub-minimum durations, and rows with no
    clean transcript."""
    n0 = len(df)
    out = df[_as_bool(df["audio_exists"])]
    n1 = len(out)
    out = out[~_as_bool(out["no_speech_flag"])]
    n2 = len(out)
    out = out[~_as_bool(out["is_spontaneous"])]
    n3 = len(out)
    out = out[~out["prompt_type"].isin(drop_prompt_types)]
    n4 = len(out)
    out = out[out["duration_sec"] >= min_duration_sec]
    n5 = len(out)
    out = out[out["transcript_clean"].notna()
              & (out["transcript_clean"].astype(str).str.strip() != "")]
    n6 = len(out)
    if verbose:
        print(f"tts_trainable_subset: {n0} rows in")
        print(f"  -{n0 - n1} missing audio -> {n1}")
        print(f"  -{n1 - n2} no-speech flagged -> {n2}")
        print(f"  -{n2 - n3} spontaneous -> {n3}")
        print(f"  -{n3 - n4} dropped prompt_types ({len(drop_prompt_types)} types) -> {n4}")
        print(f"  -{n4 - n5} under {min_duration_sec}s -> {n5}")
        print(f"  -{n5 - n6} empty transcript_clean -> {n6} rows out")
    return out.copy()


def usable_minutes_by_speaker(trainable_df: pd.DataFrame) -> pd.Series:
    """Usable minutes per speaker, computed on the ALREADY-FILTERED subset.
    Pilot lesson paid for at real cost: duration from an unfiltered frame
    makes a speaker look rich when most of their audio is digits or
    spontaneous answers."""
    return (trainable_df.groupby("SpeakerID")["duration_sec"].sum() / 60.0
            ).sort_values(ascending=False)


def rank_anchor_speakers(trainable_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Anchor-speaker candidates ranked by usable minutes, with the metadata
    needed for the report and for any later pooling escalation.

    dialect_resolved falls back SpeakerDialect -> MotherDialect (pilot
    pattern: many SpeakerDialect values are missing; discarding those
    speakers outright throws away data for no reason)."""
    g = trainable_df.copy()
    g["dialect_resolved"] = g["SpeakerDialect"].fillna(g["MotherDialect"])
    agg = (g.groupby("SpeakerID")
            .agg(usable_minutes=("duration_sec", lambda s: s.sum() / 60.0),
                 n_clips=("duration_sec", "size"),
                 mean_clip_sec=("duration_sec", "mean"),
                 gender=("SpeakerGender", "first"),
                 age=("SpeakerAge", "first"),
                 dialect=("dialect_resolved", "first"),
                 mean_rms=("rms", "mean"))
            .sort_values("usable_minutes", ascending=False)
            .head(top_n)
            .reset_index())
    agg["speaker_str"] = agg["SpeakerID"].map(speaker_id_str)
    for col in ("usable_minutes", "mean_clip_sec", "mean_rms"):
        agg[col] = agg[col].round(3)
    return agg


# ---------------------------------------------------------------------------
# Synthesis-side text frontend: prep and verbalization
# (opposite direction from scoring normalization; never cross the streams)
# ---------------------------------------------------------------------------

def prep_tts_text(text: str) -> str:
    """LIGHT cleanup for text going INTO a synthesizer. Keeps punctuation and
    case: sentence punctuation is a prosody cue, and the pilot's aggressive
    stripping is the prime suspect for its phrasing artifacts. Removes only
    genuinely unspeakable artifacts (BOM, non-breaking space, tabs) and
    collapses whitespace. Unicode NFC, NOT the scoring-side NFKD strip."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\ufeff", "").replace("\u00a0", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


_DIGIT_RUN = re.compile(r"\b\d+\b")
_BIG_DIGIT_RUN = re.compile(r"\b\d{4,}\b")


def contains_digits(text: str) -> bool:
    return bool(_DIGIT_RUN.search(str(text)))


# Curated, grows over time, SAME two-or-three-clips discipline as
# CEB_VARIANTS and YEAR_SPOKEN_TO_DIGITS in the STT module: empty until real
# usage evidence settles each entry. The June 25 mentor minutes record that
# PLD speakers mix native and Spanish-derived digit readings in Cebuano
# ("duha" vs "dos" for 2), so there is no single canonical CEB reading yet.
CEB_DIGIT_READINGS: Dict[str, str] = {}


def verbalize_eng(text: str) -> str:
    """ENG synthesis frontend: expand standalone 0-999 integers to words via
    the vendored en_number_to_words. Raises on 4+ digit runs on purpose:
    how the assistant should SPEAK years and long IDs is an unresolved
    product decision, and raising surfaces every case for review instead of
    guessing (same raise-rather-than-guess convention as the STT converters)."""
    stt = _require_stt()
    if _BIG_DIGIT_RUN.search(text):
        raise ValueError(
            "verbalize_eng: 4+ digit run found. Year/ID verbalization is a "
            "deliberate open decision; handle this text manually or extend "
            "the frontend once the convention is settled: " + text[:120])
    return _DIGIT_RUN.sub(lambda m: stt.en_number_to_words(int(m.group(0))), text)


def verbalize_fil(text: str) -> str:
    """FIL synthesis frontend: standalone 0-999 integers to Filipino words
    via the vendored fil_number_to_words. Raises on 4+ digit runs (see
    verbalize_eng docstring; FIL year readings additionally vary between
    speakers and takes per the STT evidence, so guessing is doubly wrong)."""
    stt = _require_stt()
    if _BIG_DIGIT_RUN.search(text):
        raise ValueError(
            "verbalize_fil: 4+ digit run found. FIL year readings vary "
            "between takes (STT evidence); handle manually or extend once "
            "settled: " + text[:120])
    return _DIGIT_RUN.sub(lambda m: stt.fil_number_to_words(int(m.group(0))), text)


def verbalize_ceb(text: str) -> str:
    """CEB synthesis frontend: raises on ANY digit until CEB_DIGIT_READINGS
    has evidence-backed entries. PLD speakers mix 'duha' and 'dos' style
    readings (June 25 minutes), so no converter exists to borrow and none
    should be invented without two or three real clips per pattern."""
    if contains_digits(text):
        found = _DIGIT_RUN.findall(text)
        raise ValueError(
            f"verbalize_ceb: digits {found} found but CEB digit reading is "
            "unsettled (mixed duha/dos evidence). Populate CEB_DIGIT_READINGS "
            "from real clips before verbalizing this text: " + text[:120])
    return text


_VERBALIZERS: Dict[str, Callable[[str], str]] = {
    "eng": verbalize_eng,
    "fil": verbalize_fil,
    "ceb": verbalize_ceb,
}


def get_verbalizer(lang: str) -> Callable[[str], str]:
    if lang not in _VERBALIZERS:
        raise KeyError(f"No verbalizer for lang={lang!r}; known: {list(_VERBALIZERS)}")
    return _VERBALIZERS[lang]


# ---------------------------------------------------------------------------
# Parenthetical transcript reconstruction (promoted from the tests_1 sandbox,
# July 19, with explicit confirmation; validated against the full CEB/FIL/ENG
# parenthetical population plus direct listening, see the July 18 lessons
# entry). Only the v4 count-based stack is promoted; the v1/v2/v3 set-based
# and fuzzy-match classifiers were dead ends (sawug/salug false positive,
# nmmc indeterminate case) and stay in the notebook history only.
# ---------------------------------------------------------------------------

PAREN_SPAN_RE = re.compile(r"\(([^)]*)\)")


def _normalize_words(text: str) -> list:
    """Lowercase and strip punctuation except hyphens and apostrophes, then
    split on whitespace. Mirrors the stage 2 cleaning already applied to
    transcript_clean, so parenthetical content and transcript_clean are
    compared on equal footing rather than against raw, uncleaned text.

    RECONSTRUCTION-INTERNAL ONLY. This is not a scoring normalizer and not
    a synthesis prep function; it exists so the keep/strip comparison sees
    both sides through the same lens."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return text.split()


def classify_paren_span_v4(paren_content: str, main_text: str,
                           transcript_clean: str) -> tuple:
    """Count-based, not set-based: for each word appearing in the
    parenthetical, compare its count in the main sentence alone against its
    count in transcript_clean. If transcript_clean has MORE of that word
    than the main sentence alone would produce, the extra occurrence came
    from the parenthetical being spoken. If the count did not increase, the
    parenthetical's copy was not spoken, even if that same word appears
    elsewhere in the sentence (fixes the nmmc case: 'nmmc' recurs outside
    the parens, but transcript_clean's count for it never exceeds the
    main-sentence-only count, so it correctly resolves to not-kept instead
    of being thrown out as indeterminate).

    KNOWN LIMITATION (documented, not fixed, per evidence-based convention):
    main_text has ALL parenthetical spans removed, so when two different
    spans in one row contain the same word, a count increase cannot be
    attributed to a specific span; both spans would receive the same
    verdict. The full-corpus pass plus listening never surfaced this case
    (the neda/napc row resolves correctly because its spans share no
    words). If a future manifest refresh trips it, the fix is per-span
    main_text reconstruction, not threshold tuning."""
    para_words = set(_normalize_words(paren_content))
    if not para_words:
        return "indeterminate", None

    main_counts = Counter(_normalize_words(main_text))
    clean_counts = Counter(_normalize_words(transcript_clean))

    votes = [1 if clean_counts.get(w, 0) > main_counts.get(w, 0) else 0
             for w in para_words]
    ratio = sum(votes) / len(votes)

    if ratio >= 0.8:
        return "kept", ratio
    elif ratio <= 0.2:
        return "stripped", ratio
    else:
        return "partial", ratio


def reconstruct_transcript_v4(transcript: str, transcript_clean: str,
                              prompt_type: str) -> dict:
    """Punctuation-preserving reconstruction with each parenthetical span
    resolved independently against transcript_clean (whole-row scoring
    fails on rows with mixed decisions; the neda/napc row is the confirmed
    real example).

    MinPairs rows always strip: rule-based gloss stripping, confirmed zero
    kept and zero partial across the full CEB/FIL scan. Match is
    case-insensitive substring ('minpair' in prompt_type), never an exact
    string compare; the exact-match version silently misclassified every
    real MinPairs row (actual values look like CEB_Iso_MinPairs_v1.txt).

    'partial' and 'indeterminate' spans default to stripped AND flag
    needs_review. The default is conservative but NOT safe in every case
    (the 0143 inverted row proved that), which is why needs_review rows
    hard-fail in apply_parenthetical_overrides unless explicitly reviewed.

    Returns {'reconstructed', 'span_decisions', 'needs_review'}."""
    is_minpairs = "minpair" in str(prompt_type).lower()
    spans = list(PAREN_SPAN_RE.finditer(transcript))
    if not spans:
        return {"reconstructed": transcript, "span_decisions": [],
                "needs_review": False}

    main_text = PAREN_SPAN_RE.sub("", transcript)

    decisions = []
    needs_review = False
    out_parts = []
    last_end = 0

    for m in spans:
        out_parts.append(transcript[last_end:m.start()])
        content = m.group(1)

        if is_minpairs:
            status, ratio = "stripped", None
        else:
            status, ratio = classify_paren_span_v4(content, main_text,
                                                   transcript_clean)
            if status in ("partial", "indeterminate"):
                needs_review = True
                status = "stripped"  # conservative default; overrides guard it

        decisions.append({"content": content, "status": status, "ratio": ratio})
        if status == "kept":
            out_parts.append(content)
        last_end = m.end()

    out_parts.append(transcript[last_end:])
    reconstructed = "".join(out_parts)
    reconstructed = re.sub(r"\s+([,.;:!?])", r"\1", reconstructed)
    reconstructed = re.sub(r"\s{2,}", " ", reconstructed).strip()

    return {"reconstructed": reconstructed, "span_decisions": decisions,
            "needs_review": needs_review}


# Hand-verified overrides for parenthetical rows the automated pass flagged
# needs_review. Every row that has EVER appeared with needs_review=True
# across this corpus must have an entry here: either override_text (the
# correct final text, replacing the algorithmic reconstruction) or
# confirmed=True with override_text=None (default reconstruction checked by
# ear and already correct). A flagged row with neither is a gap, not a
# silent pass; apply_parenthetical_overrides raises on it. Same curated
# grow-over-time discipline as CEB_VARIANTS and YEAR_SPOKEN_TO_DIGITS in
# the STT module: entries are added only after a listen, never guessed.
PARENTHETICAL_OVERRIDES: Dict[str, dict] = {
    "0143.120201.050119.0189.wav": {
        "override_text": "ang saklap naman n\u2019yan.",
        "confirmed": True,
        "note": ("Inverted case: transcript's English lead-in ('that's terrible.') "
                 "was NOT spoken; the parenthetical Filipino content was spoken "
                 "instead. Confirmed by ear. transcript_clean agrees "
                 "('ang saklap naman nyan'). The default kept-main/dropped-parens "
                 "assumption is wrong for this row; use the parenthetical as the "
                 "full final text."),
    },
    "1947.150108.165428.0139.wav": {
        "override_text": ("but the who report further stated that it appears to be "
                          "cost-effective to purchase a more costly but also more "
                          "beneficial option pcv thirteen if the number of doses "
                          "given is the limiting factor, and pcv ten if the net "
                          "healthcare budget is the limiting factor."),
        "confirmed": True,
        "note": ("Confirmed by ear: speaker said 'pcv thirteen' in full. "
                 "transcript_clean is shared identically across all 5 speakers "
                 "of this prompt and records it as unspoken for all of them, "
                 "which is wrong for this clip specifically. Very likely a "
                 "programmatically-filled row, never individually reviewed."),
    },
    "1966.150327.080745.0166.wav": {
        "override_text": ("but the who report further stated that it appears to be "
                          "cost-effective to purchase a more costly but also more "
                          "beneficial option pcv thirteen if the number of doses "
                          "given is the limiting factor, and pcv ten if the net "
                          "healthcare budget is the limiting factor."),
        "confirmed": True,
        "note": "Same as 1947.150108.165428.0139.wav; confirmed by ear, said in full.",
    },
    "0154.120215.064139.0238.wav": {
        "override_text": None,
        "confirmed": True,
        "note": ("Confirmed by ear: 'sixty-four, proper name' was NOT spoken. "
                 "Algorithmic 'stripped' default is correct as-is."),
    },
    "1913.141205.025329.0148.wav": {
        "override_text": None,
        "confirmed": True,
        "note": ("Confirmed by ear: 'pcv thirteen' NOT spoken by this speaker. "
                 "Algorithmic 'stripped' default is correct as-is."),
    },
    "1920.141212.073224.0153.wav": {
        "override_text": None,
        "confirmed": True,
        "note": ("Confirmed by ear: 'pcv thirteen' NOT spoken by this speaker. "
                 "Algorithmic 'stripped' default is correct as-is."),
    },
    "1913.141205.025329.0131.wav": {
        "override_text": None,
        "confirmed": True,
        "exclude_from_training": True,
        "note": ("Audio itself is truncated mid-sentence, cuts off after 'and.'. "
                 "Not a text-decision problem: no reconstructed text will match "
                 "this clip's actual audio. Excluded from training via the "
                 "exclude_from_training column rather than a text fix."),
    },
}


def apply_parenthetical_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Applies PARENTHETICAL_OVERRIDES on top of the v4 reconstruction.
    Raises loudly if any row flagged needs_review has no matching entry,
    rather than letting an unreviewed flagged row pass through silently
    (the 0143 inverted row is why the conservative default alone is not
    trusted). Adds an exclude_from_training column; callers must drop
    those rows before building any dataset, tts_trainable_subset does not
    know about them.

    Expects the run_reconstruction_pass_v4 output shape: wav,
    needs_review, reconstructed columns present."""
    out = df.copy()
    out["exclude_from_training"] = False

    unaccounted = out.loc[
        out["needs_review"] & ~out["wav"].isin(PARENTHETICAL_OVERRIDES),
        "wav"
    ].tolist()
    if unaccounted:
        raise ValueError(
            f"{len(unaccounted)} row(s) flagged needs_review have no entry in "
            f"PARENTHETICAL_OVERRIDES, listen to these before trusting the "
            f"reconstruction: {unaccounted}"
        )

    for wav_name, entry in PARENTHETICAL_OVERRIDES.items():
        mask = out["wav"] == wav_name
        if not mask.any():
            continue  # entry exists but this wav is not in this df slice
        if entry["override_text"] is not None:
            out.loc[mask, "reconstructed"] = entry["override_text"]
        if entry.get("exclude_from_training"):
            out.loc[mask, "exclude_from_training"] = True

    return out


def run_reconstruction_pass_v4(lang: str,
                               base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Reconstruction over every row of lang's FULL manifest whose transcript
    contains a parenthesis. Loads via the vendored STT module (full manifest,
    not the split manifest: reconstruction is a property of the corpus, not
    of a training split). base_dir=None uses the STT module's own
    DEFAULT_BASE_DIR, which is where the manifests actually live; the TTS
    DEFAULT_BASE_DIR points at the tts/ subtree and has no manifests."""
    stt = _require_stt()
    if base_dir is None:
        base_dir = stt.DEFAULT_BASE_DIR
    df = stt.load_full_manifest(lang, base_dir=base_dir)
    has_paren = df["transcript"].astype(str).str.contains(r"\(", regex=True,
                                                          na=False)
    subset = df.loc[has_paren].copy()

    results = subset.apply(
        lambda r: reconstruct_transcript_v4(str(r["transcript"]),
                                            str(r["transcript_clean"]),
                                            str(r["prompt_type"])),
        axis=1,
    )
    subset["reconstructed"] = results.apply(lambda d: d["reconstructed"])
    subset["needs_review"] = results.apply(lambda d: d["needs_review"])
    subset["span_decisions"] = results.apply(lambda d: d["span_decisions"])
    subset["lang"] = lang
    return subset


# ---------------------------------------------------------------------------
# Dataset build (pilot lessons: explicit Features schema, save as parquet)
# ---------------------------------------------------------------------------

def build_tts_parquet(df: pd.DataFrame, out_path: Path,
                      audio_col: str = "audio_path_cleaned",
                      text_col: str = "text_tts",
                      sampling_rate: int = 16000,
                      speaker_col: Optional[str] = None,
                      dataset_text_column: str = "text",
                      dataset_audio_column: str = "audio") -> Path:
    """Build an HF dataset and save as parquet for the fine-tuning scripts.

    Pilot lessons baked in: construct via Dataset.from_dict with an EXPLICIT
    Features schema including Audio(sampling_rate=...), never pandas type
    inference (pandas 3.0 large_string breaks cast_column with
    ArrowNotImplementedError); save as train.parquet / test.parquet files,
    not an Arrow directory, because the training scripts load via
    load_dataset.

    WEEK-1 VERIFY: dataset_text_column / dataset_audio_column defaults must
    be checked against what each fine-tuning script actually expects
    (run_vits_finetuning.py for the MMS track, the Qwen3-TTS toolkit for the
    Qwen track). Column names are parameters precisely so this function does
    not hardcode an unverified assumption."""
    from datasets import Audio, Dataset, Features, Value  # noqa: PLC0415

    data = {
        dataset_audio_column: [str(p) for p in df[audio_col]],
        dataset_text_column: [str(t) for t in df[text_col]],
    }
    features = {
        dataset_audio_column: Audio(sampling_rate=sampling_rate),
        dataset_text_column: Value("string"),
    }
    if speaker_col is not None:
        data["speaker_id"] = [int(s) for s in df[speaker_col]]
        features["speaker_id"] = Value("int64")

    ds = Dataset.from_dict(data, features=Features(features))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Audio probe (speaker-selection EDA and synth output sanity checks)
# ---------------------------------------------------------------------------

def probe_audio(path: Path) -> Dict[str, float]:
    """Header + content probe for one audio file: sample rate, duration,
    n_samples, RMS. Uses soundfile (wav/flac); anything not wav must route
    through a conversion helper first, the wav-only assumption bit three
    STT helpers in one session."""
    import soundfile as sf  # noqa: PLC0415
    info = sf.info(str(path))
    audio, sr = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    return {
        "sample_rate": float(info.samplerate),
        "n_samples": float(len(audio)),
        "duration_sec": float(info.frames) / float(info.samplerate),
        "rms": rms,
    }


# ---------------------------------------------------------------------------
# Round-trip harness placeholders (build in NB3 sandbox, promote after)
# ---------------------------------------------------------------------------

def synthesize_manifest(df: pd.DataFrame, model_tag: str, lang: str,
                        out_dir: Path) -> pd.DataFrame:
    """PLACEHOLDER (promotion discipline). Will: take a manifest with a
    prepped text column, synthesize one wav per row into
    synth_audio/{model_tag}/{lang}/, return the manifest with a synth_path
    column. Model-specific loading stays inside, one model on the GPU at a
    time (shared 11GB)."""
    raise NotImplementedError(
        "Build per-model in the Notebook 3 sandbox first, validate against "
        "real audio, then promote here with explicit confirmation.")


def transcribe_with_judge(df: pd.DataFrame, judge_tag: str, lang: str) -> pd.DataFrame:
    """PLACEHOLDER (promotion discipline). Will: transcribe synth_path rows
    with one judge ASR (whisper-medium primary, qwen3-asr-ft secondary) and
    return the manifest with a hypothesis column named for the judge."""
    raise NotImplementedError(
        "Build per-judge in the Notebook 3 sandbox first, validate, then "
        "promote here with explicit confirmation.")


# ---------------------------------------------------------------------------
# Training operations (ported from STT: validated for HF-Trainer layouts,
# which covers the MMS track; the Qwen3-TTS toolkit's checkpoint layout is a
# WEEK-1 VERIFY item before trusting these on that track)
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    best_step, best_path = None, None
    for child in output_dir.iterdir():
        m = _CKPT_RE.match(child.name)
        if m and child.is_dir():
            step = int(m.group(1))
            if best_step is None or step > best_step:
                best_step, best_path = step, child
    return best_path


def read_trainer_state(output_dir: Path) -> Tuple[Optional[dict], Optional[Path]]:
    """trainer_state.json from the latest checkpoint: the DURABLE record of
    training. Log files get overwritten by relaunches; log_history here
    does not."""
    ckpt = find_latest_checkpoint(output_dir)
    if ckpt is None:
        return None, None
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return None, ckpt
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f), ckpt


def marker_path(lang: str, stage: str, base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    """Marker file convention: run_markers/{lang}_{stage}_done.marker,
    touched by the shell chain after each stage (never by a notebook)."""
    return project_dirs(base_dir)["markers"] / f"{lang}_{stage}_done.marker"


def gpu_status() -> str:
    """Raw nvidia-smi plus compute-process query. Read it yourself before
    every launch: the GPU is shared and orphaned allocations happen."""
    try:
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                              timeout=20)
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv"],
            capture_output=True, text=True, timeout=20)
        return full.stdout + "\ncompute processes:\n" + procs.stdout
    except FileNotFoundError:
        return "nvidia-smi not found: are you on a GPU node?"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out"


def find_running_procs(patterns: Tuple[str, ...] = ("run_vits_finetuning",
                                                    "qwen3_tts",
                                                    "qwen-tts")) -> List[str]:
    """ps scan for existing training/eval processes. A duplicate-process
    incident corrupted an STT eval CSV once; check before relaunching
    anything that writes to a shared file."""
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True,
                            timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ["(ps unavailable, check manually in a terminal)"]
    hits = []
    for line in ps.stdout.splitlines():
        if any(p in line for p in patterns) and "grep" not in line:
            hits.append(line)
    return hits
