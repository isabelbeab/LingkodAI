#!/usr/bin/env python3
"""Golden check: reproduce three DS6 conversations, one per language, from
their original recorded audio, and compare every text field against what
DS6 actually produced.

    python scripts/golden_check.py \\
        --ds6 "reference/reference/DS6_augmented.xlsx" \\
        --out outputs/golden_check

Per CLAUDE.md's Testing section: final_lang must match exactly. For
transcript, English query, English answer and native answer, this script
reports an exact-match rate and prints every diff -- it does NOT fail the
run on a text mismatch. Greedy decoding can shift slightly across GPUs and
library versions (RAG originally ran on a roughly 40 GB Colab card vs
JOJIE's cards), so a diff is meant to be read by a human, not auto-failed.
TTS audio is judged by listening, not compared here -- this script only
reports whether synthesis succeeded per turn and where the WAV landed.

DS6's clip_id (e.g. "ceb_01_mt_07") encodes language and turn number but
NOT which of the 96 underlying multi-turn scripts it belongs to -- the same
conversation_id is reused across all three per-language recordings of the
same script, so a real single-language conversation is
(conversation_id, language). By default this script auto-picks the first
conversation_id present in all three languages; pass --conversation-id to
pick another one.

The audio paths recorded in DS6 (source_audio_path / audio_path_16k) are
absolute paths from whoever ran the original DS6 synthesis (JOJIE or Colab)
and will not exist on a laptop. This script raises with the exact list of
missing files rather than silently skipping them -- run it on JOJIE (or
wherever those paths resolve), or pass --audio-root-map OLD=NEW to remap a
path prefix.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_INIT_SET = Path(__file__).resolve().parents[1] / "data" / "init_set_ds6.json"

CLIP_ID_RE = re.compile(r"^(?P<lang>ceb|fil|eng)_(?P<num>\d+)_mt_(?P<turn>\d+)$")

# result field on TurnRecord -> DS6 column it is compared against
COMPARE_FIELDS = {
    "transcript": "transcript_hypothesis",
    "english_query": "translated_to_english",
    "english_answer": "rag_answer",
    "native_answer": "rag_answer_native",
}


def print_environment() -> None:
    print("Environment")
    print(f"  python       {sys.version.split()[0]} ({platform.platform()})")

    import torch

    print(f"  torch        {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU          {name}, {vram_gb:.1f} GB")
    else:
        print("  GPU          none (CPU only)")

    try:
        import transformers

        print(f"  transformers {transformers.__version__}")
    except ImportError:
        print("  transformers not installed")
    print()


@dataclass
class FieldComparison:
    field: str
    n_total: int = 0
    n_match: int = 0
    diffs: list = field(default_factory=list)  # [{clip_id, expected, actual}, ...]

    @property
    def exact_match_rate(self) -> float:
        return self.n_match / self.n_total if self.n_total else float("nan")

    def record(self, clip_id: str, expected: str, actual: str | None) -> None:
        self.n_total += 1
        actual = actual or ""
        if actual == expected:
            self.n_match += 1
        else:
            self.diffs.append({"clip_id": clip_id, "expected": expected, "actual": actual})


def load_ds6(path: Path):
    import pandas as pd

    df = pd.read_excel(path)
    required = {"clip_id", "conversation_id", "turn_id", "final_lang", "source_audio_path",
                "audio_path_16k", *COMPARE_FIELDS.values()}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} is missing expected column(s): {sorted(missing)}")

    parsed = df["clip_id"].str.extract(CLIP_ID_RE)
    unmatched = df.loc[parsed["lang"].isna(), "clip_id"].tolist()
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} clip_id value(s) don't match the expected "
            f"<lang>_<num>_mt_<turn> pattern, e.g. {unmatched[:5]}"
        )
    df = df.assign(clip_lang=parsed["lang"], clip_turn=parsed["turn"].astype(int))
    return df


def pick_conversation_id(df, requested: str | None) -> str:
    if requested is not None:
        return requested
    counts = df.groupby(["conversation_id", "clip_lang"]).size().unstack(fill_value=0)
    for lang in ("ceb", "fil", "eng"):
        if lang not in counts.columns:
            counts[lang] = 0
    eligible = counts[(counts[["ceb", "fil", "eng"]] > 0).all(axis=1)]
    if eligible.empty:
        raise RuntimeError("No conversation_id has turns in all three languages; pass --conversation-id")
    return sorted(eligible.index)[0]


def _readable(path: str) -> bool:
    """True if the file exists and can be opened. An unreadable parent directory
    (another user's home on JOJIE) raises PermissionError from exists(); treat
    that as not found so the missing-files message prints."""
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def resolve_audio_path(raw_path: str, audio_root_map: dict[str, str]) -> str:
    for old, new in audio_root_map.items():
        if raw_path.startswith(old):
            return new + raw_path[len(old):]
    return raw_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds6", required=True, type=Path, help="DS6 xlsx (multi-turn, with LID, Scenario 2)")
    ap.add_argument("--conversation-id", default=None,
                     help="which conversation_id to check (default: first one present in all 3 languages)")
    ap.add_argument("--audio-col", choices=["source_audio_path", "audio_path_16k"],
                     default="source_audio_path", help="which DS6 column to read audio paths from")
    ap.add_argument("--audio-root-map", nargs="*", default=[],
                     help="OLD=NEW path prefix remaps, e.g. /home2/msds2026/ibucayan=/mnt/jojie")
    ap.add_argument("--init-set", type=Path, default=DEFAULT_INIT_SET)
    ap.add_argument("--rag-data-dir", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    audio_root_map = dict(kv.split("=", 1) for kv in args.audio_root_map)

    print_environment()

    if not args.ds6.exists():
        print(f"ERROR: DS6 file not found: {args.ds6}", file=sys.stderr)
        return 1
    if not args.init_set.exists():
        print(f"ERROR: init set not found: {args.init_set}. Run scripts/build_init_set.py first.",
              file=sys.stderr)
        return 1

    df = load_ds6(args.ds6)
    conv_id = pick_conversation_id(df, args.conversation_id)
    print(f"DS6 source    : {args.ds6}")
    print(f"conversation  : {conv_id}")
    print(f"audio column  : {args.audio_col}")
    print()

    from src import pipeline, tts

    init_set = tts.load_init_set(args.init_set)

    field_comparisons = {name: FieldComparison(field=name) for name in COMPARE_FIELDS}
    lang_lock_results = []
    args.out.mkdir(parents=True, exist_ok=True)

    for lang in ("ceb", "fil", "eng"):
        sub = df[(df["conversation_id"] == conv_id) & (df["clip_lang"] == lang)].sort_values("clip_turn")
        if sub.empty:
            print(f"WARNING: no {lang} turns for {conv_id}, skipping")
            continue

        raw_paths = sub[args.audio_col].tolist()
        audio_paths = [resolve_audio_path(p, audio_root_map) for p in raw_paths]
        missing = [p for p in audio_paths if not _readable(p)]
        if missing:
            print(f"ERROR: {lang} conversation {conv_id}: {len(missing)} audio file(s) not found:", file=sys.stderr)
            for p in missing:
                print(f"  {p}", file=sys.stderr)
            print(
                "These are absolute paths recorded when DS6 was produced. Run this on JOJIE "
                "(or wherever they resolve), or pass --audio-root-map OLD=NEW.",
                file=sys.stderr,
            )
            return 1

        print(f"[{lang}] {conv_id}: {len(audio_paths)} turn(s)")
        result = pipeline.run_staged(audio_paths, init_set, rag_data_dir=args.rag_data_dir)

        expected_lang = sub["final_lang"].iloc[0]
        lang_lock_results.append({
            "lang": lang, "conversation_id": conv_id,
            "expected_final_lang": expected_lang,
            "actual_final_lang": result.routing.final_lang,
            "match": expected_lang == result.routing.final_lang,
        })

        clip_ids = sub["clip_id"].tolist()
        for turn, clip_id, (_, row) in zip(result.turns, clip_ids, sub.iterrows()):
            for result_field, ds6_col in COMPARE_FIELDS.items():
                field_comparisons[result_field].record(
                    clip_id, str(row[ds6_col]), getattr(turn, result_field)
                )

            audio_note = "ok" if turn.tts_ok else f"FAILED ({turn.tts_failure_reason})"
            print(f"  {clip_id}: tts {audio_note}")

            turn_out = {
                "clip_id": clip_id,
                **turn.to_json_dict(),
            }
            with open(args.out / f"{clip_id}.json", "w", encoding="utf-8") as fh:
                json.dump(turn_out, fh, indent=2, ensure_ascii=False)
            if turn.audio_out is not None:
                import soundfile as sf

                sf.write(str(args.out / f"{clip_id}.wav"), turn.audio_out, turn.audio_out_sr)

    print("\n=== final_lang ===")
    for r in lang_lock_results:
        status = "MATCH" if r["match"] else "MISMATCH"
        print(f"  {r['lang']}: expected={r['expected_final_lang']} actual={r['actual_final_lang']} [{status}]")
    all_lang_match = all(r["match"] for r in lang_lock_results)

    print("\n=== text field exact-match rates ===")
    for name, comp in field_comparisons.items():
        print(f"  {name}: {comp.n_match}/{comp.n_total} ({comp.exact_match_rate:.1%})")

    print("\n=== diffs ===")
    any_diffs = False
    for name, comp in field_comparisons.items():
        for d in comp.diffs:
            any_diffs = True
            print(f"  [{name}] {d['clip_id']}")
            print(f"    expected: {d['expected']!r}")
            print(f"    actual  : {d['actual']!r}")
    if not any_diffs:
        print("  none")

    report = {
        "ds6_source": str(args.ds6),
        "conversation_id": conv_id,
        "final_lang": lang_lock_results,
        "final_lang_all_match": all_lang_match,
        "fields": {
            name: {
                "n_total": comp.n_total,
                "n_match": comp.n_match,
                "exact_match_rate": comp.exact_match_rate,
                "diffs": comp.diffs,
            }
            for name, comp in field_comparisons.items()
        },
    }
    with open(args.out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote report and per-turn JSON/WAV to {args.out}")
    if not all_lang_match:
        print("\nfinal_lang did NOT match exactly for every conversation -- this is a real", file=sys.stderr)
        print("problem per CLAUDE.md (final_lang must match exactly), not a read-and-judge diff.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
