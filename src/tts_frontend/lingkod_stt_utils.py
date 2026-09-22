"""
lingkod_stt_utils.py

Shared utilities for the LingkodAI STT pipeline (Team CPT5, MSDS2026).
Imported by all three pipeline notebooks:
    1_finetune_qwen3_all_langs.ipynb
    2_results_summary.ipynb
    3_testing_further_analysis.ipynb
plus the presentation/EDA notebook:
    0_eda_and_figures.ipynb

Conventions baked in here (do not change casually, they match every
previously reported number):
- Row-average WER (df["wer"].mean()) is the PRIMARY metric.
  Corpus-level WER is SECONDARY and must always be labeled as such.
- Normalization is applied symmetrically to BOTH reference and hypothesis.
- Per-language normalizers run AFTER the shared base normalize_text
  (NFKD diacritic strip, lowercase, right-single-quote fix, strip).
- Error counts per clip come from jiwer's real alignment
  (substitutions + deletions + insertions), never from wer * word_count.

Environment notes (JOJIE):
- conda env "lingkod-tts" (historical typo in the name, the env is for STT).
- jiwer/matplotlib are imported lazily inside functions so this module can
  be imported even in a stripped-down kernel.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

LANGS = ["ceb", "fil", "eng"]

DEFAULT_BASE_DIR = Path.home() / "Capstone" / "github" / "Capstone_CPT5"

BASE_MODEL_HF_ID = "Qwen/Qwen3-ASR-0.6B"
BASE_MODEL_CACHE_NAME = "models--Qwen--Qwen3-ASR-0.6B"

# Stored zero-shot baselines (row-average WER, %). Do NOT recompute.
# The ENG number predates confirmation that TGL_* rows are legitimate: flag as
# provisional wherever shown.
ZERO_SHOT_BASELINES_PCT = {"ceb": 47.98, "fil": 29.37, "eng": 6.18}
ENG_BASELINE_PROVISIONAL_NOTE = (
    "ENG zero-shot baseline (6.18%) was computed before the TGL_* proper noun "
    "rows were confirmed as legitimate ENG data (July 2026 category check and "
    "listening spot-check); it is not confirmed whether that baseline run's "
    "manifest included these rows. Treat comparisons as provisional until "
    "this is confirmed one way or another."
)

# Confirmed July 2026 (category breakdown + listening spot-check): every
# TGL_* prompt_type category present in the ENG manifest is a proper-noun
# prompt list (airlines, cities, companies, hotels, names, landmarks,
# countries), reused verbatim from the Filipino prompt set rather than
# renamed for the English session. These are genuine English-session
# recordings of proper nouns, not Tagalog sentence content, so they are kept
# in ENG training. Mentors separately asked that proper noun prompts be
# retained since they matter for the PRC use case. If a manifest refresh
# ever introduces a TGL_* category outside this confirmed set, treat it as
# unverified and check it the same way before trusting it.
KNOWN_ENG_TGL_PROPER_NOUN_CATEGORIES = {
    "TGL_Airlines.txt", "TGL_Cities.txt", "TGL_Companies.txt", "TGL_Hotels.txt",
    "TGL_NameFem.txt", "TGL_NameLast.txt", "TGL_NameMale.txt",
    "TGL_Landmarks.txt", "TGL_Countries.txt",
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Shared base normalizer, applied to BOTH reference and hypothesis.

    NFKD-decompose and strip combining marks (abuhon == abuhon with accent),
    map the right single quote (u2019) to an ASCII apostrophe, lowercase,
    and collapse whitespace.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("\u2019", "'")
    return " ".join(text.lower().split())


# --- Filipino digit-to-words (0-999, raises beyond rather than guessing) ---

_FIL_ONES = {0: "sero", 1: "isa", 2: "dalawa", 3: "tatlo", 4: "apat",
             5: "lima", 6: "anim", 7: "pito", 8: "walo", 9: "siyam"}
_FIL_TENS = {10: "sampu", 20: "dalawampu", 30: "tatlumpu", 40: "apatnapu",
             50: "limampu", 60: "animnapu", 70: "pitumpu", 80: "walumpu",
             90: "siyamnapu"}


def _fil_teen(n: int) -> str:
    """11-19: 'labing-' before vowels and w, 'labin' before d/t/l/s, 'labim' before p/b."""
    ones = _FIL_ONES[n - 10]
    first = ones[0]
    if first in "dtls":
        return "labin" + ones
    if first in "pb":
        return "labim" + ones
    return "labing-" + ones  # vowels and w


def _fil_under_100(n: int) -> str:
    if n < 10:
        return _FIL_ONES[n]
    if n in _FIL_TENS:
        return _FIL_TENS[n]
    if n < 20:
        return _fil_teen(n)
    tens, ones = divmod(n, 10)
    return f"{_FIL_TENS[tens * 10]}'t {_FIL_ONES[ones]}"


def fil_number_to_words(n: int) -> str:
    """0-999 to Filipino words. Raises ValueError beyond 999 (project convention:
    raise rather than guess, so out-of-range digits surface for manual review).

    Hundreds linking: '-ng daan' after vowel-final numerals (dalawang daan),
    ' na raan' after consonant-final ones (apat na raan). Remainder joined
    with ' at '. If the reference corpus uses a different convention for the
    'at' join, adjust here, this is the same grow-over-time pattern as the
    CEB variant list.
    """
    if not (0 <= n <= 999):
        raise ValueError(f"fil_number_to_words supports 0-999, got {n}")
    if n < 100:
        return _fil_under_100(n)
    q, r = divmod(n, 100)
    w = _FIL_ONES[q]
    hundreds = (w + "ng daan") if w[-1] in "aeiou" else (w + " na raan")
    return hundreds if r == 0 else f"{hundreds} at {_fil_under_100(r)}"


def normalize_fil(text: str) -> str:
    """FIL pipeline: base normalize, then standalone digit tokens to words."""
    text = normalize_text(text)

    def _repl(m: re.Match) -> str:
        return fil_number_to_words(int(m.group(0)))

    return re.sub(r"\b\d+\b", _repl, text)


# --- Cebuano Spanish-loanword variant map (curated, grows over time) --------

# Transliteration inconsistencies, NOT model errors. Applied token-wise after
# the base normalize, to both sides, so the canonical direction is arbitrary.
CEB_VARIANTS = {
    "biente": "beinte",
    "trenta": "trienta",
    "lisinwebe": "disinwebe",
}


def normalize_ceb(text: str) -> str:
    """CEB pipeline: base normalize, then collapse curated spelling variants.

    Word-boundary regex rather than whitespace-token matching, so a variant
    with attached punctuation ('trenta,') still collapses."""
    text = normalize_text(text)
    for variant, canonical in CEB_VARIANTS.items():
        text = re.sub(rf"\b{variant}\b", canonical, text)
    return text


# --- English digits/ordinals-to-words plus hyphen-to-space ------------------

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
            60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_EN_IRREGULAR_ORDINAL = {"one": "first", "two": "second", "three": "third",
                         "five": "fifth", "eight": "eighth", "nine": "ninth",
                         "twelve": "twelfth"}


def en_number_to_words(n: int) -> str:
    """0-999 to English words, US style, no 'and' (one hundred five).
    Space-joined ('forty nine'): the hyphen-to-space step normalizes any
    hyphenated forms in the references anyway. Raises beyond 999."""
    if not (0 <= n <= 999):
        raise ValueError(f"en_number_to_words supports 0-999, got {n}")
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        base = _EN_TENS[tens * 10]
        return base if ones == 0 else f"{base} {_EN_ONES[ones]}"
    q, r = divmod(n, 100)
    hundreds = f"{_EN_ONES[q]} hundred"
    return hundreds if r == 0 else f"{hundreds} {en_number_to_words(r)}"


def en_ordinal_to_words(n: int) -> str:
    """49 -> 'forty ninth'. Only the last word is ordinalized."""
    words = en_number_to_words(n).split()
    last = words[-1]
    if last in _EN_IRREGULAR_ORDINAL:
        words[-1] = _EN_IRREGULAR_ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def normalize_eng(text: str) -> str:
    """ENG pipeline. Order matters and is a hard lesson:
    1. base normalize
    2. ordinal regex FIRST (otherwise '49th' becomes 'forty nine' + stray 'th')
    3. cardinal regex
    4. hyphen to space (father-in-law vs 'father in law' scored WER 3.0 once)
    5. whitespace collapse
    """
    text = normalize_text(text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b",
                  lambda m: en_ordinal_to_words(int(m.group(1))), text)
    text = re.sub(r"\b\d+\b", lambda m: en_number_to_words(int(m.group(0))), text)
    text = text.replace("-", " ")
    return " ".join(text.split())


_NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "ceb": normalize_ceb,
    "fil": normalize_fil,
    "eng": normalize_eng,
}


def get_normalizer(lang: str) -> Callable[[str], str]:
    """Per-language normalizer; falls back to the shared base for unknown langs."""
    return _NORMALIZERS.get(lang, normalize_text)


def apply_symmetric_normalization(df: pd.DataFrame, ref_col: str, hyp_col: str,
                                  lang: str,
                                  out_ref: str = "reference_norm",
                                  out_hyp: str = "hypothesis_norm") -> pd.DataFrame:
    """Add normalized reference AND hypothesis columns with the SAME function.

    Normalizing only one side manufactures errors (reference '33' vs converted
    hypothesis 'tatlumpu't tatlo' once scored WER 2.0). This helper exists so
    that mistake is structurally hard to repeat.
    """
    fn = get_normalizer(lang)
    out = df.copy()
    out[out_ref] = out[ref_col].map(fn)
    out[out_hyp] = out[hyp_col].map(fn)
    return out


# ---------------------------------------------------------------------------
# Year guard (promoted from the Notebook 2 sandbox, July 8)
# ---------------------------------------------------------------------------
# fil_number_to_words / en_number_to_words are capped at 0-999 by design and
# raise above that. Real eval data contains four-digit years, and years are
# NOT read the same way across languages: CEB uses a Spanish-derived cardinal
# ("dos mil unsi" for 2011), FIL uses the Filipino cardinal with an optional
# "at" join that varies between takes ("isang libo walong daan at siyamnapu't
# lima" for 1895), and only ENG uses the English year reading ("nineteen
# seventy-six" for 1976). A single "years are always English" converter was
# considered and rejected after checking real transcripts; see the lessons
# file addendum on the Notebook 2 build for the full diagnostic.
#
# This guard exempts any 4+ digit run from the cardinal converters via a
# placeholder-and-restore mechanism, and first canonicalizes any KNOWN spoken
# year form to its digit string so it collapses onto the same token as a
# digit reference. Applied identically to reference and hypothesis, so this
# cannot manufacture or hide an error: two genuinely mismatched years still
# score as a real error, only the SAME year in different notations now
# matches. A spoken year not yet in the map is left as a one-sided mismatch
# rather than silently absorbed, the same safety-net shape used for
# unverified ENG TGL_* categories.

# Curated, grows over time (same pattern as CEB_VARIANTS above). Keys are
# matched against BASE-normalized text (post normalize_text), so hyphens
# from the raw transcript are still present; only normalize_eng's own later
# step converts hyphens to spaces.
YEAR_SPOKEN_TO_DIGITS: Dict[str, str] = {
    "isang libo walong daan at siyamnapu't lima": "1895",
    "isang libo walong daan siyamnapu't lima": "1895",
    "dos mil unsi": "2011",
    "nineteen seventy-six": "1976",
}

_BIG_NUM_PATTERN = re.compile(r"\b\d{4,}\b")  # 4+ digit runs: years, not 0-999 cardinals


def safe_language_normalize(text: str, lang: str) -> str:
    """Per-language normalization with the year guard applied first.

    Do not call get_normalizer(lang) directly on text that may contain a
    4+ digit run; it will raise inside the FIL/ENG cardinal converters.
    This is the safe entry point for anything touching real eval or custom
    audio transcripts.
    """
    base = normalize_text(text)
    for spoken, digits in YEAR_SPOKEN_TO_DIGITS.items():
        base = re.sub(rf"\b{re.escape(spoken)}\b", digits, base)

    placeholders: Dict[str, str] = {}

    def _stash(m: "re.Match[str]") -> str:
        key = f"yeartoken{len(placeholders)}"
        placeholders[key] = m.group(0)
        return key

    protected = _BIG_NUM_PATTERN.sub(_stash, base)
    normed = get_normalizer(lang)(protected)  # safe: no 4+ digit run left
    for key, digits in placeholders.items():
        normed = normed.replace(key, digits)
    return normed


def apply_symmetric_normalization_safe(df: pd.DataFrame, ref_col: str, hyp_col: str,
                                       lang: str,
                                       out_ref: str = "reference_norm",
                                       out_hyp: str = "hypothesis_norm") -> pd.DataFrame:
    """Same contract as apply_symmetric_normalization, routed through
    safe_language_normalize so years cannot raise inside the FIL/ENG
    cardinal converters. Both columns get the identical function, same
    symmetry guarantee as the raw version.

    This is the entry point every notebook should call whenever the data
    may contain a 4+ digit run (years, IDs, phone-like sequences). Use the
    raw apply_symmetric_normalization only when the caller has already
    confirmed the input has no such runs.
    """
    out = df.copy()
    out[out_ref] = out[ref_col].map(lambda t: safe_language_normalize(t, lang))
    out[out_hyp] = out[hyp_col].map(lambda t: safe_language_normalize(t, lang))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _import_jiwer():
    try:
        import jiwer  # noqa: PLC0415
        return jiwer
    except ImportError as e:
        raise ImportError(
            "jiwer is required for WER metrics. Install it inside the "
            "lingkod-tts env (pip install jiwer) and check the version: "
            "process_words is the jiwer 3.x API."
        ) from e


def count_word_errors(reference: str, hypothesis: str) -> int:
    """Exact word-error count via jiwer's real alignment.

    substitutions + deletions + insertions per clip. Do NOT approximate this
    from wer * word_count: rounding misclassifies clips right at the 1-vs-2
    error boundary. Pass ALREADY-NORMALIZED text (both sides), or formatting
    mismatches get double-counted as real errors.

    Handles both jiwer 3.x (process_words) and older pinned versions
    (compute_measures) since the two APIs return similar but not identical
    shapes.
    """
    jiwer = _import_jiwer()
    if hasattr(jiwer, "process_words"):
        out = jiwer.process_words(reference, hypothesis)
        return out.substitutions + out.deletions + out.insertions
    measures = jiwer.compute_measures(reference, hypothesis)
    return measures["substitutions"] + measures["deletions"] + measures["insertions"]


def row_wer(reference: str, hypothesis: str) -> float:
    """Single-pair WER. Row-average of this column is the project's PRIMARY metric."""
    jiwer = _import_jiwer()
    return jiwer.wer(reference, hypothesis)


def corpus_wer(references: List[str], hypotheses: List[str]) -> float:
    """SECONDARY metric only. Weights by clip length and reads lower on this
    data than row-average WER. Always label it as corpus-level when reporting,
    mixing the two silently caused a fake regression scare once."""
    jiwer = _import_jiwer()
    return jiwer.wer(list(references), list(hypotheses))


def add_error_columns(df: pd.DataFrame,
                      ref_col: str = "reference_norm",
                      hyp_col: str = "hypothesis_norm",
                      wer_col: str = "wer_norm",
                      err_col: str = "n_word_errors") -> pd.DataFrame:
    """Add per-row WER and exact word-error-count columns.

    Expects NORMALIZED columns (run apply_symmetric_normalization first)."""
    out = df.copy()
    out[wer_col] = [row_wer(r, h) for r, h in zip(out[ref_col], out[hyp_col])]
    out[err_col] = [count_word_errors(r, h) for r, h in zip(out[ref_col], out[hyp_col])]
    return out


def error_count_breakdown(df: pd.DataFrame, err_col: str = "n_word_errors") -> pd.DataFrame:
    """Mentor-requested (7/3 dry run): clips with 0, exactly 1, exactly 2, 3+ errors."""
    n = len(df)
    buckets = {
        "0 errors (exact match)": int((df[err_col] == 0).sum()),
        "exactly 1 error": int((df[err_col] == 1).sum()),
        "exactly 2 errors": int((df[err_col] == 2).sum()),
        "3+ errors": int((df[err_col] >= 3).sum()),
    }
    out = pd.DataFrame({"n_clips": buckets})
    out["pct"] = (out["n_clips"] / n * 100).round(2) if n else 0.0
    return out


def exact_match_summary(df: pd.DataFrame, wer_col: str = "wer_norm",
                        by: Optional[str] = None) -> pd.DataFrame:
    """Exact match rate = fraction of clips with WER == 0.
    Stricter than 1 - WER; the two should NOT be expected to align.
    Pass by='prompt_type' for the per-category view."""
    def _summ(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_clips": len(g),
            "exact_match_rate": float((g[wer_col] == 0).mean()),
        })
    if by is None:
        return _summ(df).to_frame().T
    return df.groupby(by, dropna=False).apply(_summ, include_groups=False).reset_index()


def wer_before_after(df: pd.DataFrame, ref_col: str, hyp_col: str, lang: str) -> pd.DataFrame:
    """Row-average WER with base normalization only (BEFORE) vs the full
    per-language pipeline (AFTER), side by side. Never silently replace the
    raw number.

    The AFTER call is year-guarded (apply_symmetric_normalization_safe):
    this function used to call the raw apply_symmetric_normalization for
    both sides, which meant it recomputed normalization from raw text and
    hit the FIL/ENG 4+ digit crash independently of any fix applied
    elsewhere (see the lessons file addendum on the Notebook 2 build). The
    BEFORE call is untouched on purpose: lang="__base__" already falls
    through to plain normalize_text, which never reaches the cardinal
    converters, so it was never the crash site, and it must keep matching
    the stored zero-shot baselines exactly.
    """
    base = apply_symmetric_normalization(df, ref_col, hyp_col, lang="__base__")
    full = apply_symmetric_normalization_safe(df, ref_col, hyp_col, lang=lang)
    before = [row_wer(r, h) for r, h in zip(base["reference_norm"], base["hypothesis_norm"])]
    after = [row_wer(r, h) for r, h in zip(full["reference_norm"], full["hypothesis_norm"])]
    return pd.DataFrame({
        "metric": ["row-average WER (PRIMARY)"],
        "before_norm (base only)": [sum(before) / len(before) if before else float("nan")],
        f"after_norm ({lang} pipeline)": [sum(after) / len(after) if after else float("nan")],
        "n_clips": [len(df)],
    })


def worst_best_examples(df: pd.DataFrame, n: int = 10, wer_col: str = "wer_norm",
                        cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Both worst-N AND best-N as dataframes (mentors asked for both, 7/3).
    Best-N tie-breaks toward longer references so the examples are non-trivial."""
    if cols is None:
        cols = [c for c in ("audio_path_cleaned", "prompt_type", "reference_norm",
                            "hypothesis_norm", wer_col, "n_word_errors") if c in df.columns]
    work = df.copy()
    work["_ref_len"] = work.get("reference_norm", work.get("reference", "")).astype(str).str.split().str.len()
    worst = work.sort_values(wer_col, ascending=False).head(n)[cols]
    best = work.sort_values([wer_col, "_ref_len"], ascending=[True, False]).head(n)[cols]
    return worst, best


# ---------------------------------------------------------------------------
# Manifest and jsonl helpers
# ---------------------------------------------------------------------------

def load_split_manifest(lang: str, base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame:
    path = Path(base_dir) / "data" / "manifests" / "qwen3_dryrun_split" / f"{lang}_qwen3_dryrun_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    return pd.read_csv(path)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def filter_training_rows(df: pd.DataFrame, lang: str, verbose: bool = True) -> pd.DataFrame:
    """Defensive row filter before jsonl building.

    Drops (when the column exists): missing audio, spontaneous speech,
    no-speech flagged rows, empty transcripts. Zero drops is the expected,
    printed outcome once a manifest is already clean.

    ENG TGL_* prompt_type rows are NOT dropped. See
    KNOWN_ENG_TGL_PROPER_NOUN_CATEGORIES above: these are confirmed
    proper-noun prompts recorded for the English session, not Tagalog
    sentence content, so they stay in training and are only reported here
    for visibility. If a category shows up under TGL_* that isn't in the
    confirmed set, it prints as unverified rather than being silently kept
    or silently dropped, since that would repeat the same kind of
    unchecked assumption that caused this to be mishandled once already.
    """
    out = df.copy()
    report: Dict[str, int] = {}

    if "audio_exists" in out.columns:
        mask = ~_as_bool(out["audio_exists"])
        report["audio_exists == False"] = int(mask.sum())
        out = out[~mask]
    if "is_spontaneous" in out.columns:
        mask = _as_bool(out["is_spontaneous"])
        report["is_spontaneous == True"] = int(mask.sum())
        out = out[~mask]
    if "no_speech_flag" in out.columns:
        mask = _as_bool(out["no_speech_flag"])
        report["no_speech_flag == True"] = int(mask.sum())
        out = out[~mask]
    if "transcript_clean" in out.columns:
        mask = out["transcript_clean"].isna() | (out["transcript_clean"].astype(str).str.strip() == "")
        report["empty transcript_clean"] = int(mask.sum())
        out = out[~mask]

    retained_note = None
    if lang == "eng" and "prompt_type" in out.columns:
        tgl_types = out.loc[out["prompt_type"].astype(str).str.startswith("TGL_"), "prompt_type"]
        n_tgl = int(len(tgl_types))
        if n_tgl:
            unknown = sorted(set(tgl_types.astype(str)) - KNOWN_ENG_TGL_PROPER_NOUN_CATEGORIES)
            retained_note = (f"ENG TGL_* prompt_type: {n_tgl} rows retained "
                            f"(confirmed proper-noun prompts, not Tagalog content)")
            if unknown:
                retained_note += (f"; UNVERIFIED categories present, check these "
                                  f"the same way before trusting them: {unknown}")

    if verbose:
        total_dropped = sum(report.values())
        print(f"[{lang}] filter_training_rows: {len(df)} -> {len(out)} rows "
              f"({total_dropped} dropped)")
        for reason, cnt in report.items():
            print(f"    {reason}: {cnt}")
        if total_dropped == 0:
            print("    zero drops: expected outcome, manifest was already clean")
        if retained_note:
            print(f"    {retained_note}")
    return out


def build_jsonl(df: pd.DataFrame, out_path: Path,
                audio_col: str = "audio_path_cleaned",
                text_col: str = "transcript_clean",
                prompt: str = "") -> int:
    """Write {"audio", "text", "prompt"} lines for qwen3_asr_sft.py.

    prompt defaults to '' because qwen3_eval_batch.py transcribes with NO
    context text; training with a system prompt would create a
    train/inference mismatch.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "audio": str(row[audio_col]),
                "text": str(row[text_col]),
                "prompt": prompt,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def ensure_jsonl(df: pd.DataFrame, out_path: Path, verbose: bool = True, **kwargs) -> int:
    """Idempotent build_jsonl: skip if the file exists with a matching line
    count, rebuild (and say so) otherwise. Keeps the notebook re-runnable
    from a fresh kernel without clobbering anything mid-training."""
    out_path = Path(out_path)
    expected = len(df)
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing == expected:
            if verbose:
                print(f"    {out_path.name}: exists with {existing} rows, skipping")
            return existing
        if verbose:
            print(f"    {out_path.name}: exists with {existing} rows but expected "
                  f"{expected}, REBUILDING")
    n = build_jsonl(df, out_path, **kwargs)
    if verbose:
        print(f"    {out_path.name}: wrote {n} rows")
    return n


def sample_audio_exists(df: pd.DataFrame, audio_col: str = "audio_path_cleaned",
                        n: int = 25, seed: int = 42) -> List[str]:
    """Spot-check that a random sample of audio paths exist on disk.
    Returns the missing paths (empty list = all sampled files present)."""
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    return [p for p in sample[audio_col] if not Path(str(p)).exists()]


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


# ---------------------------------------------------------------------------
# Training operations: checkpoints, status, launch commands
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
    """Load trainer_state.json from the latest checkpoint.

    This file is the DURABLE record of training: log files get overwritten by
    relaunches, log_history in here does not."""
    ckpt = find_latest_checkpoint(output_dir)
    if ckpt is None:
        return None, None
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return None, ckpt
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f), ckpt


def expected_max_steps(n_train: int, batch_size: int = 1, grad_acc: int = 32,
                       epochs: int = 3) -> int:
    """Pre-launch estimate matching HF Trainer's step math for this config.
    Once training has started, trust trainer_state.json's own 'max_steps'
    field instead of this."""
    len_dataloader = math.ceil(n_train / batch_size)
    steps_per_epoch = max(math.ceil(len_dataloader / grad_acc), 1)
    return math.ceil(epochs * steps_per_epoch)


def training_status(lang: str, output_dir: Path, marker_path: Path) -> dict:
    """Async status for one language, readable with no live process.

    Also detects the zero-step-resume false crash: global_step == max_steps
    means training is actually DONE; a relaunch exits almost instantly and
    that instant exit is normal, not a failure."""
    state, ckpt = read_trainer_state(output_dir)
    marker = Path(marker_path).exists()
    if state is None:
        verdict = "no checkpoint yet (not started, or first save_steps not reached)"
        return {"lang": lang, "latest_ckpt": str(ckpt) if ckpt else None,
                "global_step": None, "max_steps": None, "epoch": None,
                "marker": marker, "verdict": verdict}
    gs, ms = state.get("global_step"), state.get("max_steps")
    if ms and gs == ms and marker:
        verdict = "FINISHED (steps complete, marker present)"
    elif ms and gs == ms:
        verdict = ("steps complete but marker missing: post-train eval segment may "
                   "still be running, or the chain died between train end and touch; "
                   "check the log tail")
    else:
        verdict = f"in progress or stopped at step {gs}/{ms}"
    return {"lang": lang, "latest_ckpt": str(ckpt), "global_step": gs,
            "max_steps": ms, "epoch": round(state.get("epoch", 0), 3),
            "marker": marker, "verdict": verdict}


def resolve_base_snapshot(cache_name: str = BASE_MODEL_CACHE_NAME) -> Optional[Path]:
    """Locate the cached base-model snapshot dir (the <hash> folder) that holds
    preprocessor_config.json. Needed both for HF_HUB_OFFLINE sanity and for
    the processor-file copy into finished checkpoints."""
    snap_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if not snap_root.is_dir():
        return None
    candidates = [d for d in snap_root.iterdir()
                  if d.is_dir() and (d / "preprocessor_config.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def ensure_checkpoint_inferable(ckpt_dir: Path, snapshot_dir: Optional[Path] = None,
                                files: Tuple[str, ...] = ("preprocessor_config.json",
                                                          "chat_template.json")) -> Dict[str, str]:
    """Verify (and if needed, repair) the processor bundle in a checkpoint.

    Trainer auto-save does not include the full processor bundle. The
    MakeEveryCheckpointInferableCallback in the training script should have
    copied these already, but verify presence rather than trusting it."""
    ckpt_dir = Path(ckpt_dir)
    if snapshot_dir is None:
        snapshot_dir = resolve_base_snapshot()
    report: Dict[str, str] = {}
    for fn in files:
        dst = ckpt_dir / fn
        if dst.exists():
            report[fn] = "present"
            continue
        if snapshot_dir and (Path(snapshot_dir) / fn).exists():
            shutil.copy2(Path(snapshot_dir) / fn, dst)
            report[fn] = "was missing, copied from base snapshot"
        else:
            report[fn] = "MISSING and no base snapshot found (fix before inference)"
    return report


# --- pre-flight checks -------------------------------------------------------

SCRIPT_PATCHES: List[Tuple[str, str]] = [
    ("fp32 dtype fallback (Turing has no bf16; fp16 crashes GradScaler)", "else torch.float32"),
    ("Adafactor optimizer (AdamW fp32 moments do not fit in 11GB)", 'optim="adafactor"'),
    ("gradient_checkpointing=True (required for the backward pass)", "gradient_checkpointing=True"),
    ("eval batch size tied to CLI batch_size (script default 8 OOMs)", "per_device_eval_batch_size=args_cli.batch_size"),
    ("explicit post-train trainer.evaluate() (zero-step resume skips eval otherwise)", "trainer.evaluate()"),
    ("MakeEveryCheckpointInferableCallback (stock, must remain)", "MakeEveryCheckpointInferableCallback"),
]


def check_script_patches(script_path: Path) -> Dict[str, bool]:
    """Byte-for-byte substring checks for every required patch in the training
    script. Any False result means DO NOT LAUNCH."""
    text = Path(script_path).read_text(encoding="utf-8")
    return {name: (needle in text) for name, needle in SCRIPT_PATCHES}


def gpu_status() -> str:
    """Raw nvidia-smi output plus a compute-process query. Read it yourself
    before every launch: this GPU is shared, and orphaned allocations with no
    visible process do happen here (they usually clear on their own)."""
    try:
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv"],
            capture_output=True, text=True, timeout=20)
        return full.stdout + "\ncompute processes:\n" + procs.stdout
    except FileNotFoundError:
        return "nvidia-smi not found: are you on a GPU node?"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out"


def find_running_procs(patterns: Tuple[str, ...] = ("qwen3_asr_sft.py",
                                                    "qwen3_eval_batch.py")) -> List[str]:
    """ps scan for existing training/eval processes. A duplicate-process
    incident (two eval PIDs writing one CSV) corrupted results once; check
    this before relaunching anything that writes to a shared file."""
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ["(ps unavailable, check manually in a terminal)"]
    hits = []
    for line in ps.stdout.splitlines():
        if any(p in line for p in patterns) and "grep" not in line:
            hits.append(line)
    return hits


# --- launch command builders --------------------------------------------------

def build_train_command(script_path: Path, model_path: str, train_file: Path,
                        eval_file: Path, output_dir: Path, hp: dict) -> str:
    """One single-line python command for one language. No placeholders."""
    return (f"python {script_path} "
            f"--model_path {model_path} "
            f"--train_file {train_file} "
            f"--eval_file {eval_file} "
            f"--output_dir {output_dir} "
            f"--batch_size {hp['batch_size']} "
            f"--grad_acc {hp['grad_acc']} "
            f"--lr {hp['lr']} "
            f"--epochs {hp['epochs']} "
            f"--save_steps {hp['save_steps']} "
            f"--save_total_limit {hp['save_total_limit']} "
            f"--resume {hp['resume']}")


def build_chain_launch(base_dir: Path, per_lang_commands: Dict[str, str],
                       markers: Dict[str, Path], env_exports: str,
                       chain_log: Path) -> str:
    """The full copy-paste terminal launch: nohup + disown, && chaining, a
    touch marker after each language, >> append so relaunches stop erasing
    the log. ONE line, zero placeholders, zero angle brackets.

    Assumes the lingkod-tts conda env is ALREADY ACTIVE in the terminal that
    pastes this (the child bash inherits the env's python via PATH)."""
    inner_parts = [
        'echo "=== chain start $(date) python=$(which python) ==="',
        f"export {env_exports}",
    ]
    for lang, cmd in per_lang_commands.items():
        inner_parts.append(cmd)
        inner_parts.append(f"touch {markers[lang]}")
    inner = " && ".join(inner_parts)
    line = (f"cd {base_dir} && nohup bash -c '{inner}' >> {chain_log} 2>&1 & "
            f'TRAIN_PID=$!; echo "train chain PID: ${{TRAIN_PID}} '
            f'(wrapper bash PID; the python PIDs show up in nvidia-smi and ps)"; disown')
    if "<" in line:
        # A <placeholder> pasted literally caused bash syntax errors twice.
        # Every angle-bracket placeholder contains '<', so this single check
        # catches them all while leaving the intentional >> and 2>&1 alone.
        raise ValueError("'<' found in launch command, a placeholder leaked in; refusing to emit it")
    return line


def build_eval_command(eval_script: Path, checkpoint_dir: Path, manifest_csv: Path,
                       results_csv: Path, split: str = "test") -> str:
    return (f"python {eval_script} "
            f"--checkpoint_dir {checkpoint_dir} "
            f"--manifest_csv {manifest_csv} "
            f"--results_csv {results_csv} "
            f"--split {split}")


MONITOR_FUNCTION_TEMPLATE = """\
monitor_training () {{
    local pid="$1"
    if [ -z "$pid" ]; then echo "usage: monitor_training PID"; return 1; fi
    while kill -0 "$pid" 2>/dev/null; do
        echo "==================== $(date) ===================="
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
        echo "-- markers --"
        ls -1 {markers_dir} 2>/dev/null || echo "(none yet)"
        echo "-- last log lines --"
        tail -n 8 {chain_log}
        sleep 60
    done
    echo "PID $pid has exited. Markers now:"
    ls -1 {markers_dir} 2>/dev/null || echo "(none)"
}}"""


def monitor_function_text(markers_dir: Path, chain_log: Path) -> str:
    """Polling monitor to paste into a terminal. A while loop with kill -0,
    NOT watch: watch segfaults on the JOJIE terminal."""
    return MONITOR_FUNCTION_TEMPLATE.format(markers_dir=markers_dir, chain_log=chain_log)


# --- loss curves from trainer_state, not logs ---------------------------------

def read_log_history(output_dir: Path) -> List[dict]:
    state, _ = read_trainer_state(output_dir)
    return state.get("log_history", []) if state else []


def plot_loss_curves(output_dirs: Dict[str, Path]):
    """Train/eval loss per language from checkpoint trainer_state.json.

    Deliberately does NOT read log files: single > redirection overwrote
    them on every relaunch once, log_history in the checkpoint is the only
    durable record."""
    import matplotlib.pyplot as plt  # lazy: keep module importable anywhere
    n = len(output_dirs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), squeeze=False)
    for ax, (lang, od) in zip(axes[0], output_dirs.items()):
        state, _ = read_trainer_state(od)
        if state is None:
            ax.set_title(f"{lang}: no checkpoint yet")
            ax.set_axis_off()
            continue
        hist = state.get("log_history", [])
        train = [(e["step"], e["loss"]) for e in hist if "loss" in e]
        evals = [(e["step"], e["eval_loss"]) for e in hist if "eval_loss" in e]
        if train:
            ax.plot(*zip(*train), label="train loss")
        if evals:
            ax.plot(*zip(*evals), marker="o", label="eval loss")
        ax.set_title(f"{lang}  (step {state.get('global_step')}/{state.get('max_steps')})")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


# Special-case registration so wer_before_after can request base-only normalization.
_NORMALIZERS["__base__"] = normalize_text


# ---------------------------------------------------------------------------
# EDA and presentation figures (0_eda_and_figures.ipynb)
# ---------------------------------------------------------------------------
# Everything below supports the standalone EDA/slide-figure notebook. It is
# additive: nothing above this line is changed by it, and none of the three
# pipeline notebooks need to import from this section.

# --- brand colors (deck palette) --------------------------------------------

NAVY = "#0A2647"
ORANGE = "#F4A835"
BG = "#F5F8FF"
# Third series color for charts with all three languages side by side.
# A muted slate teal: distinct from both NAVY and ORANGE at a glance (including
# for common red/green colorblindness), rather than a tint of either one.
# Swap this single constant if a different third color is preferred; nothing
# else needs to change.
TEAL = "#3E7C8C"

LANG_COLORS: Dict[str, str] = {"ceb": NAVY, "fil": ORANGE, "eng": TEAL}
LANG_LABELS: Dict[str, str] = {"ceb": "Cebuano (CEB)", "fil": "Filipino (FIL)", "eng": "English (ENG)"}


def set_plot_style() -> None:
    """Apply the deck palette to matplotlib rcParams. Call once per notebook
    session, before the first figure. Every figure in every notebook should
    call this so charts look consistent without repeating rcParams by hand."""
    import matplotlib.pyplot as plt  # lazy: keep module importable anywhere
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.grid": True,
        "axes.axisbelow": True,  # grid draws behind bars/lines, not on top
        "grid.color": "#D8E1F0",
        "grid.alpha": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
    })


def label_bars(ax, fmt: str = "%.0f", padding: int = 2, fontsize: int = 9,
               color: str = NAVY) -> None:
    """Add a value label above (or beside, for barh) every bar currently on
    ax. Call this right after plotting, once per axes, after all ax.bar/
    ax.barh calls for that axes are done. Works for both orientations since
    matplotlib's bar_label reads it off the container.

    Skip this for histograms with many bins; a label per bin clutters rather
    than clarifies. It's meant for the low-bar-count charts (grouped bars,
    category breakdowns), which is most of what this notebook plots."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding, fontsize=fontsize, color=color)


def save_figure(fig, name: str, out_dir: Path, dpi: int = 200) -> Path:
    """Save a figure as a slide-ready PNG. out_dir is created if missing.
    Filename is sanitized (spaces to underscores) so it is safe to drop
    straight into a slide deck folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_")
    if not safe_name.lower().endswith(".png"):
        safe_name += ".png"
    path = out_dir / safe_name
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    return path


# --- full (pre-split) manifest loader ---------------------------------------

def load_full_manifest(lang: str, base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """The full per-language manifest BEFORE train/val/test splitting
    (data/manifests/{lang}_audio_clean.csv). This is a different file from
    load_split_manifest's target: it has every row (including any later
    excluded for spontaneous speech, missing audio, etc.) and is the right
    source for corpus-wide EDA like duration or sample-rate distributions,
    where you want the whole dataset, not just the training split.
    """
    path = Path(base_dir) / "data" / "manifests" / f"{lang}_audio_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Full manifest not found: {path}")
    return pd.read_csv(path)


# --- sample rate probing (header-only, cached) ------------------------------

def probe_sample_rates(paths: List[str]) -> List[Optional[int]]:
    """Read the sample rate from each file's header only (soundfile.info,
    no audio decode), so this stays fast even across tens of thousands of
    files. Returns None for any path that fails to open, rather than raising,
    so one bad path does not stop the whole scan."""
    import soundfile as sf  # lazy: only needed for this EDA path
    rates: List[Optional[int]] = []
    for p in paths:
        try:
            rates.append(int(sf.info(str(p)).samplerate))
        except Exception:
            rates.append(None)
    return rates


def estimate_sample_rate_scan_time(df: pd.DataFrame, audio_col: str = "audio_path",
                                   sample_size: int = 50, seed: int = 0) -> Dict[str, float]:
    """Time a header-only scan on a small sample and extrapolate, same habit
    as the timing-estimate cell in 1_audio_prep_for_qwen3.ipynb before
    committing to a full pass. Run this before load_or_compute_sample_rates
    on a manifest you have not scanned before."""
    import time
    sample = df[audio_col].sample(n=min(sample_size, len(df)), random_state=seed)
    t0 = time.time()
    for p in sample:
        try:
            import soundfile as sf
            sf.info(str(p))
        except Exception:
            pass
    elapsed = time.time() - t0
    per_file = elapsed / len(sample) if len(sample) else 0.0
    n_files = len(df)
    return {"per_file_ms": per_file * 1000, "est_total_sec": per_file * n_files,
            "n_files": n_files}


def sample_rate_cache_path(lang: str, base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    return Path(base_dir) / "data" / "eda_cache" / f"{lang}_sample_rates.csv"


def load_or_compute_sample_rates(df: pd.DataFrame, lang: str, audio_col: str = "audio_path",
                                 base_dir: Path = DEFAULT_BASE_DIR, force: bool = False,
                                 verbose: bool = True) -> pd.DataFrame:
    """Merge a 'sample_rate' column onto df, computing it only for paths not
    already in the on-disk cache. Idempotent: a re-run only probes new or
    previously-unreadable paths, following the same skip-if-already-there
    shape as ensure_jsonl. force=True ignores the cache and reprobes everything.
    """
    cache_path = sample_rate_cache_path(lang, base_dir=base_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=[audio_col, "sample_rate"])

    known = set(cache[audio_col]) if len(cache) else set()
    to_probe = [p for p in df[audio_col].unique() if p not in known]

    if to_probe:
        if verbose:
            print(f"[{lang}] probing sample rate for {len(to_probe)} new file(s) "
                  f"(header-only, {len(known)} already cached)")
        rates = probe_sample_rates(to_probe)
        cache = pd.concat(
            [cache, pd.DataFrame({audio_col: to_probe, "sample_rate": rates})],
            ignore_index=True,
        )
        cache.to_csv(cache_path, index=False)
    elif verbose:
        print(f"[{lang}] sample rate cache already covers all {df[audio_col].nunique()} "
              f"unique file(s), skipping probe")

    return df.merge(cache, on=audio_col, how="left", validate="many_to_one")


# --- category / speaker / split summaries -----------------------------------

def collapse_prompt_types(df: pd.DataFrame, group_patterns: Dict[str, str],
                          prompt_col: str = "prompt_type",
                          out_col: str = "prompt_type_grouped") -> pd.DataFrame:
    """Relabel prompt_type values matching a regex pattern into a single
    group label, leaving every other prompt_type value unchanged.

    group_patterns maps regex -> group label, e.g.:
        {r"^CEBUtt_|^CEBU_": "uncategorized (CEBUtt_/CEBU_)",
         r"_Shib$": "Shib prompts (grouped)"}

    Patterns are checked in the order given; the first match wins for any
    value that matches more than one. Does not touch prompt_col itself,
    writes the collapsed labels to out_col so the original is still there
    for reference.

    This is for genuinely sparse, often auto-generated category codes that
    read poorly as individual bars/boxes in a chart, not a general-purpose
    "other" bucket; unmatched categories are left exactly as they are,
    however sparse. Print df[out_col].value_counts() after calling this the
    first time against real data, to confirm the patterns actually caught
    what you expected and nothing else.
    """
    out = df.copy()
    grouped = out[prompt_col].astype(str).copy()
    already_grouped = pd.Series(False, index=out.index)
    for pattern, group_label in group_patterns.items():
        mask = (out[prompt_col].astype(str).str.contains(pattern, regex=True, na=False)
               & ~already_grouped)
        grouped.loc[mask] = group_label
        already_grouped |= mask
    out[out_col] = grouped
    return out


def prompt_type_distribution(df: pd.DataFrame, prompt_col: str = "prompt_type",
                             top_n: int = 10) -> pd.DataFrame:
    """Row counts per prompt_type, collapsing everything past the top_n most
    frequent categories into a single 'other' bucket so a long tail of rare
    categories does not crowd out the chart."""
    counts = df[prompt_col].astype(str).value_counts()
    if len(counts) > top_n:
        top = counts.iloc[:top_n].copy()
        top.loc["other"] = int(counts.iloc[top_n:].sum())
    else:
        top = counts.copy()
    out = top.reset_index()
    out.columns = [prompt_col, "n_clips"]
    out["pct"] = (out["n_clips"] / len(df) * 100).round(2)
    return out


def split_size_breakdown(langs: List[str] = LANGS,
                         base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """Row counts per split (train/val/test) per language, from the split
    manifests (load_split_manifest), not the full pre-split manifest, since
    the full manifest has no 'split' column."""
    rows = []
    for lang in langs:
        df = load_split_manifest(lang, base_dir=base_dir)
        for split_name, cnt in df["split"].value_counts().items():
            rows.append({"lang": lang, "split": split_name, "n_clips": int(cnt)})
    long = pd.DataFrame(rows)
    return long.pivot(index="lang", columns="split", values="n_clips").fillna(0).astype(int)


def speaker_counts(df: pd.DataFrame, speaker_col: str = "SpeakerID") -> pd.DataFrame:
    """Per-speaker clip counts for one language's manifest, sorted descending.
    Useful for spotting imbalance (e.g. the ENG speaker-1917 concern already
    on the project's radar)."""
    counts = df[speaker_col].value_counts().rename("n_clips").reset_index()
    counts.columns = [speaker_col, "n_clips"]
    counts["pct"] = (counts["n_clips"] / len(df) * 100).round(2)
    return counts.sort_values("n_clips", ascending=False).reset_index(drop=True)


def n_speakers_per_lang(langs: List[str] = LANGS, base_dir: Path = DEFAULT_BASE_DIR,
                        speaker_col: str = "SpeakerID") -> pd.DataFrame:
    """Unique speaker count and total clip count per language, from the full
    (pre-split) manifest."""
    rows = []
    for lang in langs:
        df = load_full_manifest(lang, base_dir=base_dir)
        rows.append({"lang": lang, "n_speakers": int(df[speaker_col].nunique()),
                    "n_clips": len(df)})
    return pd.DataFrame(rows)


# --- word / token count verification -----------------------------------------

def recount_word_count(df: pd.DataFrame, text_col: str = "transcript_clean",
                       out_col: str = "word_count_recount") -> pd.DataFrame:
    """Fresh whitespace-split word count from text_col, independent of
    whatever the manifest's own token_count column was actually computed
    from. Use this instead of trusting token_count at face value; see
    compare_token_count_to_recount for why that matters here."""
    out = df.copy()
    out[out_col] = out[text_col].fillna("").astype(str).str.split().str.len()
    return out


def compare_token_count_to_recount(df: pd.DataFrame, stored_col: str = "token_count",
                                   text_col: str = "transcript_clean",
                                   by_col: str = "prompt_type") -> pd.DataFrame:
    """Per-prompt_type comparison of the manifest's stored token_count against
    a fresh recount of text_col. A single dataset-wide average would hide a
    divergence that only hits one prompt_type, e.g. MinPairs, whose
    transcript_clean has already had parenthetical English translations
    stripped (June 25 meeting notes); token_count may or may not have been
    computed on that same stripped text. Sorted by |mean_diff| descending so
    the most-divergent category surfaces first."""
    tmp = recount_word_count(df, text_col=text_col, out_col="_recount")
    tmp["_diff"] = tmp["_recount"] - tmp[stored_col]
    grouped = tmp.groupby(by_col).agg(
        n_clips=(stored_col, "size"),
        mean_stored=(stored_col, "mean"),
        mean_recount=("_recount", "mean"),
        mean_diff=("_diff", "mean"),
        n_mismatched=("_diff", lambda s: int((s != 0).sum())),
    ).reset_index()
    return grouped.sort_values("mean_diff", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# --- eval results loaders (Notebook 2 outputs) ------------------------------

EVAL_CSV_TEMPLATE = "{lang}_full_eval_results_3epoch.csv"
# Settled convention (see PATH CONVENTIONS notes): earlier docs assumed the
# shorter "{lang}.csv"; that is not what qwen3_eval_batch.py actually writes.
# If a future refresh changes this again, update this one constant.


def eval_results_path(lang: str, base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    return Path(base_dir) / "data" / "eval_results" / EVAL_CSV_TEMPLATE.format(lang=lang)


def load_eval_results(lang: str, base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """Raw eval CSV: exactly four columns (audio path, reference, hypothesis,
    wer), where wer is base-normalized only (matches the stored zero-shot
    baselines). For anything needing prompt_type, token_count, or speaker,
    use load_eval_results_merged instead."""
    path = eval_results_path(lang, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Eval results not found for {lang}: {path}")
    return pd.read_csv(path)


def load_eval_results_merged(lang: str, base_dir: Path = DEFAULT_BASE_DIR,
                             audio_col: str = "audio_path_cleaned",
                             split: Optional[str] = None) -> pd.DataFrame:
    """Eval CSV merged onto the split manifest so prompt_type, token_count,
    and speaker are all available in one frame (e.g. for a per-speaker or
    per-prompt_type WER breakdown, or a speaker-share chart).

    Merge key is audio_col, shared by both frames (qwen3_eval_batch.py's
    --audio_col defaults to "audio_path_cleaned", the same column name the
    split manifest already carries). validate='many_to_one' per project rule.
    Speaker is extracted from the filename (speaker_from_filename), not read
    from a SpeakerID column, since the eval CSV itself has no such column
    before the merge.

    split=None merges against the full split manifest (train+val+test); the
    eval CSV itself only ever contains whatever split it was run against
    (normally 'test'), so filtering is rarely necessary, but is available if
    you want to be explicit about it.
    """
    eval_df = load_eval_results(lang, base_dir=base_dir)
    manifest = load_split_manifest(lang, base_dir=base_dir)
    if split is not None:
        manifest = manifest[manifest["split"] == split]
    merged = safe_merge(eval_df, manifest, on=audio_col, how="left")
    merged["speaker"] = merged[audio_col].map(speaker_from_filename)
    return merged


# --- no-speech ("holes in the clean dataset") reporting ---------------------

# Two column presets for the two audiences this usually gets built for:
# a short sponsor-facing list, and a fuller one for manually spot-checking
# (listening to) why each file got flagged.
PRESENTATION_COLS_NO_SPEECH = ["lang", "wav", "prompt_type", "duration_sec"]
DIAGNOSTIC_COLS_NO_SPEECH = [
    "lang", "wav", "SpeakerID", "prompt_type", "duration_sec",
    "rms", "band_ratio", "no_speech_reason", "transcript_clean",
    "audio_path", "audio_path_16k",
]


def no_speech_rows(lang: str, base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """Rows from lang's full manifest where no_speech_flag == True, with a
    'lang' column prepended so multiple languages concatenate cleanly. Uses
    the same _as_bool coercion filter_training_rows already relies on, so
    this matches exactly what gets dropped before training, nothing more
    or less."""
    df = load_full_manifest(lang, base_dir=base_dir)
    if "no_speech_flag" not in df.columns:
        raise KeyError(f"[{lang}] manifest has no 'no_speech_flag' column")
    flagged = df.loc[_as_bool(df["no_speech_flag"])].copy()
    flagged.insert(0, "lang", lang)
    return flagged


def no_speech_summary_counts(langs: List[str] = LANGS,
                             base_dir: Path = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """One row per language: n_flagged, n_total, pct_flagged. The headline
    number for a 'here's how many holes were in the supposedly clean
    dataset' report."""
    rows = []
    for lang in langs:
        n_total = len(load_full_manifest(lang, base_dir=base_dir))
        n_flagged = len(no_speech_rows(lang, base_dir=base_dir))
        rows.append({
            "lang": lang, "n_flagged": n_flagged, "n_total": n_total,
            "pct_flagged": round(100 * n_flagged / n_total, 3) if n_total else 0.0,
        })
    return pd.DataFrame(rows)


def no_speech_report(langs: List[str] = LANGS, base_dir: Path = DEFAULT_BASE_DIR,
                     cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Combined no-speech-flagged rows across languages. Pass cols to select
    a subset, e.g. PRESENTATION_COLS_NO_SPEECH or DIAGNOSTIC_COLS_NO_SPEECH;
    leave as None for every column in the manifest."""
    frames = [no_speech_rows(lang, base_dir=base_dir) for lang in langs]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cols is not None and len(out):
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise KeyError(f"no_speech_report: requested columns not in manifest: {missing}")
        out = out[cols]
    return out
