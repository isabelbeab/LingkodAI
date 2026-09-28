#!/usr/bin/env python3
"""CLI entry point: audio files in, per-turn JSON and WAV out.

    python scripts/run_conversation.py --audio TURN1.wav TURN2.wav --out outputs/RUN_NAME

TURN1.wav, TURN2.wav and RUN_NAME are placeholders -- pass as many --audio
files as the conversation has turns, in order.

Runs the `staged` pipeline (src/pipeline.run_staged): phase-major over the
whole conversation, the way the evaluated DS6 run was produced. See
README.md's "Execution modes" for why -- the models do not fit together on
an 11 GB GPU.

Prints resolved paths and versions before doing any work, per this project's
"no silent fallbacks" rule -- a missing file, token, or unexpected shape
raises with a clear message rather than falling back to something plausible.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_INIT_SET = Path(__file__).resolve().parents[1] / "data" / "init_set_ds6.json"


def print_environment() -> None:
    """Print resolved paths and versions before doing any work."""
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


def write_turn_outputs(out_dir: Path, turn) -> None:
    """One JSON file (everything except raw audio) and, if synthesis
    succeeded, one WAV file per turn."""
    json_path = out_dir / f"turn_{turn.turn_idx}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(turn.to_json_dict(), fh, indent=2, ensure_ascii=False)

    if turn.audio_out is not None:
        import soundfile as sf

        wav_path = out_dir / f"turn_{turn.turn_idx}.wav"
        sf.write(str(wav_path), turn.audio_out, turn.audio_out_sr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", nargs="+", required=True, help="one audio file per turn, in order")
    ap.add_argument("--out", required=True, type=Path, help="output directory (created if missing)")
    ap.add_argument(
        "--init-set",
        type=Path,
        default=DEFAULT_INIT_SET,
        help=f"frozen TTS initialism set JSON (default: {DEFAULT_INIT_SET})",
    )
    ap.add_argument(
        "--rag-data-dir",
        type=Path,
        default=None,
        help="override the PRC chunk directory (default: src/rag.py's DATA_DIR)",
    )
    args = ap.parse_args()

    print_environment()

    missing_audio = [p for p in args.audio if not Path(p).exists()]
    if missing_audio:
        print(f"ERROR: audio file(s) not found: {missing_audio}", file=sys.stderr)
        return 1
    if not args.init_set.exists():
        print(
            f"ERROR: init set not found: {args.init_set}\n"
            "Run scripts/build_init_set.py first (needs the DS6 golden xlsx), "
            "or pass --init-set to point at one.",
            file=sys.stderr,
        )
        return 1

    print(f"Audio turns   : {args.audio}")
    print(f"Output dir    : {args.out}")
    print(f"Init set      : {args.init_set}")
    print(f"RAG data dir  : {args.rag_data_dir or '(default)'}")
    print()

    from src import pipeline, tts

    init_set = tts.load_init_set(args.init_set)
    print(f"Loaded {len(init_set)} frozen initialism tokens\n")

    result = pipeline.run_staged(
        list(args.audio),
        init_set,
        rag_data_dir=args.rag_data_dir,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for turn in result.turns:
        write_turn_outputs(args.out, turn)

    manifest = {
        "n_turns": len(result.turns),
        "final_lang": result.routing.final_lang,
        "was_overridden": result.routing.was_overridden,
        "turn_files": [f"turn_{t.turn_idx}.json" for t in result.turns],
    }
    with open(args.out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nDone. final_lang={result.routing.final_lang}"
          f" (was_overridden={result.routing.was_overridden})")
    for turn in result.turns:
        audio_note = "audio ok" if turn.tts_ok else f"NO AUDIO ({turn.tts_failure_reason})"
        print(f"  turn {turn.turn_idx}: {audio_note}")
    print(f"\nWrote {len(result.turns)} turn(s) to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
