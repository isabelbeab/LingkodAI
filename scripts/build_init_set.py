#!/usr/bin/env python3
"""Freeze the TTS initialism set to JSON, once, from the DS6 and DS7 sources.

WHY THIS EXISTS
---------------
The DS6/DS7 synthesis notebook built `INIT_SET` at runtime from the whole
answer column of one .xlsx file, then passed it to every verbalizer. A live
pipeline has one answer per turn, not a corpus, so the obvious move is to
derive the set per answer. That is NOT equivalent, and the difference is
silent.

The notebook's regex is:

    \\b[A-Z]{2,}[0-9]*\\b

Run it on "CPAs are registered" and it returns nothing at all. `[A-Z]{2,}`
matches "CPA", `[0-9]*` matches empty, and then `\\b` demands a word boundary,
but the next character is "s", which is a word character. Backtracking to "CP"
fails the same test against "A". So a plural acronym contributes nothing to the
set.

Meanwhile `spell_out_initialisms` in shared_domain_frontend.py does this:

    stripped = tok.rstrip("s")
    if tok not in initialism_set and stripped not in initialism_set:
        return tok

So "CPAs" IS spelled out, but only when the singular "CPA" made it into the set
from somewhere else in the corpus. Over 2,235 rows it usually did. Over one
answer it often does not, and the acronym then passes through unspelled with no
warning. Same text, different audio.

Freezing the set removes the question. The synthesis behaviour is then exactly
what DS6 and DS7 were evaluated with, and it does not depend on which answer is
being spoken.

WHAT THIS WRITES
----------------
  data/init_set_ds6.json    Scenario 2 (with LID). This is what the pipeline uses.
  data/init_set_ds7.json    Scenario 1 (no LID). Kept for reproducing DS7.
  data/init_set_live.json   Union of both. See "Which set to use" below.

Each file carries its own provenance: source filename, SHA-256, row count, the
regex, and the STOP list, so a future reader can tell where the tokens came from
without rerunning anything.

WHICH SET TO USE
----------------
The golden check against DS6 uses init_set_ds6.json, because reproducing the
evaluated run means reproducing its exact set. Live turns on genuinely new text
may use init_set_live.json: a superset can only spell out MORE acronyms, never
fewer, so it cannot silently un-spell something DS6 spelled. Whichever one a run
used gets recorded in that run's output JSON. Do not mix them inside one run.

A known quirk, preserved on purpose: the notebook's STOP list contains "US",
while shared_domain_frontend.STOP_INITIALISM deliberately REMOVED "US" after
finding 17 corpus occurrences that were all the country abbreviation. These are
two different lists doing two different jobs, and the notebook's list is the one
that produced the evaluated audio. Reproducing DS6 means keeping "US" excluded
here. Do not "fix" it without regenerating and re-listening.

USAGE
-----
    python scripts/build_init_set.py \\
        --ds6 reference/ds6_multiturn_w_lid_backtranslated.xlsx \\
        --ds7 reference/ds7_multiturn_no_lid_backtranslated.xlsx \\
        --out data

Needs pandas and openpyxl. Run it once, commit the JSON, and do not run it again
unless a source file changes. If you do regenerate, tests/test_init_set.py
asserts the counts, so an unintended change fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Copied VERBATIM from the DS6/DS7 synthesis notebook, run-config cell.
# Do not tidy, reorder or extend these two. They define what the evaluated
# audio was produced with, and any edit silently changes synthesis.
# ---------------------------------------------------------------------------

TEXT_COL = "rag_answer_native"

STOP = {
    "OF", "AND", "THE", "TO", "FOR", "OR", "IN", "ON", "AT", "BY", "IS", "AS", "BE",
    "AN", "IF", "NO", "ALL", "ANY", "ONE", "TWO", "NEW", "PER", "NOT", "MAY", "CAN",
    "SEE", "USE", "WHO", "HIS", "HER", "ITS", "ARE", "WAS", "TABLE", "APPLICATION",
    "PROFESSIONAL", "ORIGINAL", "COPY", "LETTER", "REQUEST", "VALID", "CARD",
    "PHOTOCOPY", "GOVERNMENT", "IDENTIFICATION", "ISSUANCE", "CERTIFIED",
    "TRUE", "LEGAL", "DOCUMENTS", "OTHER", "CASES", "APPEALED", "PLEADINGS",
    "FEE", "FEES", "TOTAL", "NA", "II", "III", "IV", "US", "AA", "AM", "PM",
}

_INIT_RE = re.compile(r"\b[A-Z]{2,}[0-9]*\b")


def build_init_set(full_df: pd.DataFrame) -> set[str]:
    """The notebook's own three lines, unchanged.

    Note it reads the FULL file, not just one language's rows. That is
    deliberate in the original: an acronym seen in a Cebuano answer is still a
    real acronym when it turns up in a Filipino one.
    """
    all_text = " ".join(full_df[TEXT_COL].dropna().astype(str))
    return set(t for t in _INIT_RE.findall(all_text) if t not in STOP)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """Hash the source file so a later reader can prove which copy was used.

    Read in 1 MB blocks rather than all at once: these .xlsx files are a few
    megabytes now, but a hash helper that loads whole files into memory is the
    kind of thing that quietly breaks on a bigger input later.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_set(tokens: set[str], out_path: Path, *, label: str,
              source_path: Path | None, source_sha: str | None,
              n_rows: int | None, note: str) -> None:
    """Write one frozen set as JSON, sorted, with its provenance beside it.

    sorted() matters: Python's set iteration order is not stable across runs,
    so writing an unsorted list would produce a different file every time and
    make git diffs meaningless even when nothing actually changed.
    """
    payload = {
        "label": label,
        "generated": date.today().isoformat(),
        "generated_by": "scripts/build_init_set.py",
        "source_file": source_path.name if source_path else None,
        "source_sha256": source_sha,
        "source_rows": n_rows,
        "text_column": TEXT_COL,
        "regex": _INIT_RE.pattern,
        "stop_list": sorted(STOP),
        "n_tokens": len(tokens),
        "note": note,
        "tokens": sorted(tokens),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  wrote {out_path}  ({len(tokens)} tokens)")


def load_set(path: Path) -> set[str]:
    """Read a frozen set back. This is what src/tts.py should call at runtime."""
    with open(path, encoding="utf-8") as fh:
        return set(json.load(fh)["tokens"])


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds6", required=True, type=Path,
                    help="DS6 xlsx (multi-turn RAG answers, WITH LID, Scenario 2)")
    ap.add_argument("--ds7", type=Path, default=None,
                    help="DS7 xlsx (multi-turn RAG answers, no LID, Scenario 1). "
                         "Optional: without it, no DS7 or live set is written.")
    ap.add_argument("--out", type=Path, default=Path("data"),
                    help="output directory (default: data)")
    args = ap.parse_args()

    for p in [args.ds6] + ([args.ds7] if args.ds7 else []):
        if not p.exists():
            print(f"ERROR: source not found: {p}", file=sys.stderr)
            return 1

    print("Building frozen initialism sets")
    print(f"  regex     : {_INIT_RE.pattern}")
    print(f"  text col  : {TEXT_COL}")
    print(f"  stop list : {len(STOP)} tokens\n")

    ds6_df = pd.read_excel(args.ds6)
    ds6_set = build_init_set(ds6_df)
    print(f"DS6 {args.ds6.name}: {len(ds6_df)} rows -> {len(ds6_set)} tokens")
    write_set(
        ds6_set, args.out / "init_set_ds6.json",
        label="ds6", source_path=args.ds6, source_sha=sha256_of(args.ds6),
        n_rows=len(ds6_df),
        note=("Scenario 2 (with LID). Reproduces the evaluated DS6 synthesis "
              "exactly. Use this for the golden check."),
    )

    if args.ds7 is None:
        print("\nNo --ds7 given: DS7 and live sets not written.")
        return 0

    ds7_df = pd.read_excel(args.ds7)
    ds7_set = build_init_set(ds7_df)
    print(f"DS7 {args.ds7.name}: {len(ds7_df)} rows -> {len(ds7_set)} tokens")
    write_set(
        ds7_set, args.out / "init_set_ds7.json",
        label="ds7", source_path=args.ds7, source_sha=sha256_of(args.ds7),
        n_rows=len(ds7_df),
        note="Scenario 1 (no LID). Kept for reproducing DS7 only.",
    )

    live_set = ds6_set | ds7_set
    write_set(
        live_set, args.out / "init_set_live.json",
        label="live", source_path=None, source_sha=None, n_rows=None,
        note=("Union of the DS6 and DS7 sets, for live turns on text that is in "
              "neither file. A superset can only spell out more acronyms, never "
              "fewer, so it cannot un-spell anything DS6 spelled. Never use this "
              "for the golden check."),
    )

    # Diagnostics. These are printed, not asserted: they are here so a human
    # reading the output can see what the two files actually disagree about
    # rather than trusting that they agree.
    only6 = sorted(ds6_set - ds7_set)
    only7 = sorted(ds7_set - ds6_set)
    print("\nDifferences between the two sets")
    print(f"  in DS6 only ({len(only6)}): {only6[:20]}{' ...' if len(only6) > 20 else ''}")
    print(f"  in DS7 only ({len(only7)}): {only7[:20]}{' ...' if len(only7) > 20 else ''}")
    print(f"  union       : {len(live_set)} tokens")
    if only6 or only7:
        print("  The two files hold the same RAG answers but back-translated under")
        print("  different language routing, so a small difference here is expected.")

    # The plural-acronym behaviour this whole script exists because of, shown
    # against the real set rather than described.
    #
    # Two different regexes are at work and that is the whole point:
    #   _INIT_RE  (this script)              decides what ENTERS the set
    #   _SPELL_TOKEN (shared_domain_frontend) decides what gets LOOKED UP
    # The lookup mirrors spell_out_initialisms exactly: try the token, then the
    # token with a trailing "s" stripped. A row where "collected by regex" is
    # empty but "spelled out" is not is a row that a per-answer set would have
    # left unspelled.
    _SPELL_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")
    print("\nPlural-acronym check (why per-answer derivation is unsafe)")
    for probe in ["CPAs are registered professionals.",
                  "Bring your PICs and CORs.",
                  "The PRC issues the COR."]:
        collected = _INIT_RE.findall(probe)
        spelled = [tok for tok in _SPELL_TOKEN.findall(probe)
                   if tok in live_set or tok.rstrip("s") in live_set]
        print(f"  {probe!r}")
        print(f"    collected by regex : {collected}")
        print(f"    spelled out        : {spelled}")
        missed = [t for t in spelled if t not in collected]
        if missed:
            print(f"    -> {missed} would be UNSPELLED if the set came from this answer alone")

    print("\nDone. Commit the JSON files. Do not regenerate without re-listening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
