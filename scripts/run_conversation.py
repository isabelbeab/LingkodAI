#!/usr/bin/env python3
"""CLI entry point: audio files in, per-turn JSON and WAV out.

    python scripts/run_conversation.py --audio TURN1.wav TURN2.wav --out outputs/RUN_NAME

TURN1.wav, TURN2.wav and RUN_NAME are placeholders -- pass as many --audio
files as the conversation has turns, in order.

Stage 7 backend (branch tts-cpp): --tts-backend hf (default, src/tts.py) or
--tts-backend cpp (src/tts_cpp.py on a patched qwentts.cpp tts-server built by
scripts/build_qwentts.sh), with --tts-mode parity (arm B) or segmented
(arm C). The cpp backend also writes per-segment WAVs and timeline.json, the
simulated play-ahead (time to first audio, stalls) for every turn:

    python scripts/run_conversation.py --audio TURN1.wav --out outputs/RUN_NAME \
        --tts-backend cpp --tts-mode segmented --tts-seed 1234

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
    succeeded, one WAV file per turn, plus one per segment for the cpp
    backend (turn_N_segKK.wav, in playback order)."""
    json_path = out_dir / f"turn_{turn.turn_idx}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(turn.to_json_dict(), fh, indent=2, ensure_ascii=False)

    if turn.audio_out is not None:
        import soundfile as sf

        wav_path = out_dir / f"turn_{turn.turn_idx}.wav"
        sf.write(str(wav_path), turn.audio_out, turn.audio_out_sr)
        for k, seg_audio in enumerate(turn.tts_segment_audio or []):
            sf.write(str(out_dir / f"turn_{turn.turn_idx}_seg{k:02d}.wav"), seg_audio, turn.audio_out_sr)


def timeline_summary(turns) -> list[dict]:
    """Per turn: the simulated play-ahead plus a compact view of each
    segment, for timeline.json."""
    keep = ("index", "chars", "pause_after_s", "expected_frames", "cap", "frames", "cap_hit", "retries",
            "audio_s", "wall_s")
    return [
        {
            "turn_idx": t.turn_idx,
            "tts_ok": t.tts_ok,
            "tts_failure_reason": t.tts_failure_reason,
            "tts_wall_sec": t.tts_wall_sec,
            "tts_rtf": t.tts_rtf,
            "timeline": t.tts_timeline,
            "segments": [{k: seg.get(k) for k in keep} for seg in (t.tts_segments or [])],
        }
        for t in turns
    ]


def cpp_tts_options(args) -> dict:
    """tts_cpp.load keyword arguments from the --tts-* flags. Unset flags are
    left out so tts_cpp's environment variables and defaults apply."""
    opts = {
        "mode": args.tts_mode or "parity",
        "seed": args.tts_seed,
        "server_bin": args.tts_server_bin,
        "talker_gguf": args.tts_talker_gguf,
        "codec_gguf": args.tts_codec_gguf,
        "talker_ctx": args.tts_talker_ctx,
        "extra_server_args": tuple(args.tts_server_arg or ()),
        "log_dir": args.out,
    }
    return {k: v for k, v in opts.items() if v not in (None, ())}


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
    tts_group = ap.add_argument_group("Stage 7 backend")
    tts_group.add_argument("--tts-backend", choices=("hf", "cpp"), default="hf",
                           help="hf: qwen-tts in process (evaluated path, default); cpp: qwentts.cpp tts-server")
    tts_group.add_argument("--tts-mode", choices=("parity", "segmented"), default=None,
                           help="cpp only. parity: src/tts.py's text path (default); segmented: split, cap, retry")
    tts_group.add_argument("--tts-seed", type=int, default=None,
                           help="cpp only. Base seed; drawn and recorded when omitted")
    tts_group.add_argument("--tts-server-bin", type=Path, default=None,
                           help="cpp only. tts-server binary (else QWENTTS_SERVER_BIN or the build script's path)")
    tts_group.add_argument("--tts-talker-gguf", type=Path, default=None,
                           help="cpp only. Talker GGUF (else QWENTTS_TALKER_GGUF)")
    tts_group.add_argument("--tts-codec-gguf", type=Path, default=None,
                           help="cpp only. Codec GGUF (else QWENTTS_CODEC_GGUF)")
    tts_group.add_argument("--tts-talker-ctx", type=int, default=None,
                           help="cpp only. Talker KV positions (default: src/tts_cpp.DEFAULT_TALKER_CTX)")
    tts_group.add_argument("--tts-server-arg", action="append", default=None,
                           help="cpp only. Extra tts-server flag, repeatable (e.g. --tts-server-arg=--clamp-fp16)")
    args = ap.parse_args()

    cpp_only = [flag for flag, value in (
        ("--tts-mode", args.tts_mode), ("--tts-seed", args.tts_seed), ("--tts-server-bin", args.tts_server_bin),
        ("--tts-talker-gguf", args.tts_talker_gguf), ("--tts-codec-gguf", args.tts_codec_gguf),
        ("--tts-talker-ctx", args.tts_talker_ctx), ("--tts-server-arg", args.tts_server_arg),
    ) if value is not None]
    if args.tts_backend == "hf" and cpp_only:
        print(f"ERROR: only valid with --tts-backend cpp: {', '.join(cpp_only)}", file=sys.stderr)
        return 1

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
    print(f"TTS backend   : {args.tts_backend}"
          + (f" ({args.tts_mode or 'parity'})" if args.tts_backend == "cpp" else ""))
    print()

    from src import pipeline, tts

    init_set = tts.load_init_set(args.init_set)
    print(f"Loaded {len(init_set)} frozen initialism tokens\n")

    # Created before the run: the cpp backend writes tts-server.log here.
    args.out.mkdir(parents=True, exist_ok=True)
    result = pipeline.run_staged(
        list(args.audio),
        init_set,
        rag_data_dir=args.rag_data_dir,
        tts_backend=args.tts_backend,
        tts_options=cpp_tts_options(args) if args.tts_backend == "cpp" else None,
    )

    for turn in result.turns:
        write_turn_outputs(args.out, turn)

    manifest = {
        "n_turns": len(result.turns),
        "final_lang": result.routing.final_lang,
        "was_overridden": result.routing.was_overridden,
        "turn_files": [f"turn_{t.turn_idx}.json" for t in result.turns],
        "tts_backend": args.tts_backend,
    }
    if args.tts_backend == "cpp":
        manifest["tts"] = result.tts_info
        with open(args.out / "timeline.json", "w", encoding="utf-8") as fh:
            json.dump(timeline_summary(result.turns), fh, indent=2, ensure_ascii=False)
    with open(args.out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nDone. final_lang={result.routing.final_lang}"
          f" (was_overridden={result.routing.was_overridden})")
    for turn in result.turns:
        audio_note = "audio ok" if turn.tts_ok else f"NO AUDIO ({turn.tts_failure_reason})"
        if turn.tts_timeline:
            tl = turn.tts_timeline
            audio_note += (f", {len(turn.tts_segments)} segment(s), time to first audio {tl['ttfa_s']:.2f}s, "
                           f"{tl['n_stalls']} stall(s) ({tl['total_stall_s']:.2f}s)")
        print(f"  turn {turn.turn_idx}: {audio_note}")
    print(f"\nWrote {len(result.turns)} turn(s) to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
