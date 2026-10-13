#!/usr/bin/env python3
"""Run every recording in a folder: audio in input/, answers in output/.

    python scripts/run_folder.py                       # input/ -> output/
    python scripts/run_folder.py --input DIR --output DIR

What counts as a conversation, inside the input folder:
  - an audio file directly in it is a one-question conversation, named after
    the file (input/renew_id.m4a -> output/renew_id/);
  - a subfolder is a multi-turn conversation, its audio files taken as turns
    in filename order (input/fees/01_ask.m4a, input/fees/02_followup.m4a ->
    output/fees/), so follow-ups are read in the light of earlier answers.
Other files (notes, images, hidden files) are ignored.

Each conversation goes through scripts/run_conversation.py's pipeline and
output layout (manifest.json, turn_N.json, turn_N.wav). Then
output/summary.csv gets one row per question, for every conversation in the
input folder: language, transcript, answers and the answer's WAV file.

Safe to run again: a conversation whose output folder already has a
manifest.json is skipped (pass --force to redo it). If one conversation
fails, the error goes into the summary and the rest still run; the script
then exits with code 1.

The pipeline loads its models once per conversation (the `staged` mode, see
README.md's "Execution modes"), so each conversation pays the model-loading
time. Questions that belong together run faster as one subfolder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_conversation import DEFAULT_INIT_SET, print_environment, write_conversation  # noqa: E402

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
SUMMARY_NAME = "summary.csv"
SUMMARY_FIELDS = [
    "conversation", "turn", "input_file", "status", "final_lang", "transcript",
    "english_query", "english_answer", "native_answer", "chunk_ids", "audio_file",
    "error", "seconds",
]


def _is_audio(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in AUDIO_EXTS


def discover(input_dir: Path) -> list[tuple[str, list[Path]]]:
    """(conversation name, turn files in order) for everything in input_dir,
    sorted by name. Raises on two conversations with the same name."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_dir}")
    found: dict[str, list[Path]] = {}
    sources: dict[str, Path] = {}
    for entry in sorted(input_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            turns = sorted((p for p in entry.iterdir() if _is_audio(p)), key=lambda p: p.name)
            if not turns:
                print(f"  note: {entry.name}/ has no audio files, skipped")
                continue
            name = entry.name
        elif _is_audio(entry):
            name, turns = entry.stem, [entry]
        else:
            continue
        if name in found:
            raise ValueError(
                f"two inputs would both write to output/{name}/: {sources[name].name} and "
                f"{entry.name}. Rename one."
            )
        found[name], sources[name] = turns, entry
    return list(found.items())


def summary_rows(name: str, out_dir: Path, seconds: float | None) -> list[dict]:
    """One row per turn of a finished conversation, read back from its files."""
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for turn_file in manifest["turn_files"]:
        t = json.loads((out_dir / turn_file).read_text(encoding="utf-8"))
        wav = out_dir / f"turn_{t['turn_idx']}.wav"
        if t.get("tts_ok"):
            status = "ok"
        else:
            status = f"text only: {t.get('tts_failure_reason')}"
        rows.append({
            "conversation": name,
            "turn": t["turn_idx"] + 1,
            "input_file": Path(t["audio_path"]).name,
            "status": status,
            "final_lang": t.get("final_lang"),
            "transcript": t.get("transcript"),
            "english_query": t.get("english_query"),
            "english_answer": t.get("english_answer"),
            "native_answer": t.get("native_answer"),
            "chunk_ids": " ".join(t.get("chunk_ids") or []),
            "audio_file": f"{name}/{wav.name}" if wav.exists() else "",
            "error": "",
            "seconds": "" if seconds is None else f"{seconds:.0f}",
        })
    return rows


def run_folder(input_dir: Path, output_dir: Path, init_set_path: Path,
               rag_data_dir: Path | None = None, force: bool = False) -> int:
    conversations = discover(input_dir)
    print(f"Input folder  : {input_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Init set      : {init_set_path}")
    print(f"Conversations : {len(conversations)}")
    for name, turns in conversations:
        print(f"  {name}: {len(turns)} turn(s): {', '.join(p.name for p in turns)}")
    print()
    if not conversations:
        print(f"Nothing to do: put audio files or conversation folders in {input_dir}")
        return 0
    if not init_set_path.exists():
        print(f"ERROR: init set not found: {init_set_path}", file=sys.stderr)
        return 1

    from src import pipeline, tts

    init_set = tts.load_init_set(init_set_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failed = 0
    for i, (name, turns) in enumerate(conversations, 1):
        out_dir = output_dir / name
        header = f"[{i}/{len(conversations)}] {name}"
        if (out_dir / "manifest.json").exists() and not force:
            print(f"{header}: already done, skipped (--force to redo)")
            rows += summary_rows(name, out_dir, None)
            continue
        print(f"{header}: running {len(turns)} turn(s)")
        start = time.monotonic()
        try:
            result = pipeline.run_staged([str(p) for p in turns], init_set, rag_data_dir=rag_data_dir)
            write_conversation(out_dir, result)
        except Exception as exc:  # recorded and reported, not hidden: see the exit code
            failed += 1
            traceback.print_exc()
            print(f"{header}: FAILED ({type(exc).__name__}: {exc})")
            rows.append({**{k: "" for k in SUMMARY_FIELDS}, "conversation": name,
                         "input_file": " ".join(p.name for p in turns), "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}",
                         "seconds": f"{time.monotonic() - start:.0f}"})
            continue
        seconds = time.monotonic() - start
        new_rows = summary_rows(name, out_dir, seconds)
        rows += new_rows
        print(f"{header}: done in {seconds:.0f} s, language {new_rows[0]['final_lang']}")

    summary = output_dir / SUMMARY_NAME
    with open(summary, "w", encoding="utf-8-sig", newline="") as fh:  # -sig: Excel reads it as UTF-8
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {summary} ({len(rows)} row(s))")
    if failed:
        print(f"{failed} conversation(s) failed; see the error column", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("input"), help="folder of recordings (default: input)")
    ap.add_argument("--output", type=Path, default=Path("output"), help="results folder (default: output)")
    ap.add_argument("--init-set", type=Path, default=DEFAULT_INIT_SET,
                    help=f"frozen TTS initialism set JSON (default: {DEFAULT_INIT_SET})")
    ap.add_argument("--rag-data-dir", type=Path, default=None,
                    help="override the PRC chunk directory (default: src/rag.py's DATA_DIR)")
    ap.add_argument("--force", action="store_true", help="redo conversations that already have output")
    args = ap.parse_args()

    print_environment()
    try:
        return run_folder(args.input, args.output, args.init_set, args.rag_data_dir, args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
