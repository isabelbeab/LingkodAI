#!/usr/bin/env python3
"""Transcribe the golden-check clips with two sets of ASR checkpoints and require identical output.

WHY THIS EXISTS
---------------
The ASR default repo IDs moved from beabucayan/lingkodai-whisper-* to
lingkodai/lingkodai-whisper-* (CHANGELOG.md, 2026-09-28).
scripts/compare_hf_repos.py already showed the Hub files are identical. This
is the GPU-side check: the org copies download with the project token, patch,
load through src/asr.py and give exactly the same transcripts as the old
copies, on the same GPU, audio and code. Only stage 2 changed, so a full
golden check (every stage, hours) is not needed.

WHAT IT DOES
------------
Reads the per-turn JSON files an earlier scripts/golden_check.py run wrote
(clip_id, audio_path, final_lang) and groups the clips by final_lang. For each
language it loads that language's checkpoint from --old-root/<lang>,
transcribes the clips, unloads it, then does the same from --new-root/<lang>.
Every clip is transcribed with its own language's checkpoint, so all three
models are exercised, including the CEB custom-token path. Each root is a
scripts/patch_asr_tokenizers.py --out directory.

src/asr.py is not edited: its CHECKPOINTS dict is pointed at each root in turn.

READING THE RESULT
------------------
  exit 0   every clip's transcript is identical between old and new
  exit 2   at least one transcript differs (every diff is printed)
  exit 1   a missing directory, file or language

USAGE
-----
    python scripts/ab_asr_checkpoints.py \\
        --old-root outputs/asr_patched --new-root outputs/asr_patched_org \\
        --golden-dir outputs/golden_check
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANGS = ("ceb", "fil", "eng")


def print_environment(args: argparse.Namespace) -> None:
    """Print resolved paths and versions before doing any work."""
    print("Environment")
    print(f"  python       {sys.version.split()[0]} ({platform.platform()})")

    import torch
    import transformers

    print(f"  torch        {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU          {name}, {vram_gb:.1f} GB")
    else:
        print("  GPU          none (CPU only)")
    print(f"  transformers {transformers.__version__}")
    print(f"  old root     {args.old_root.resolve()}")
    print(f"  new root     {args.new_root.resolve()}")
    print(f"  golden dir   {args.golden_dir.resolve()}")
    print()


def load_clips(golden_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Return {lang: [(clip_id, audio_path), ...]} from a golden_check output dir."""
    clips: dict[str, list[tuple[str, str]]] = {lang: [] for lang in LANGS}
    for path in sorted(golden_dir.glob("*.json")):
        if path.name == "report.json":
            continue
        turn = json.loads(path.read_text(encoding="utf-8"))
        lang = turn["final_lang"]
        if lang not in clips:
            raise ValueError(f"{path}: unexpected final_lang {lang!r}")
        if not Path(turn["audio_path"]).is_file():
            raise FileNotFoundError(f"{path}: audio_path does not exist: {turn['audio_path']}")
        clips[lang].append((turn["clip_id"], turn["audio_path"]))
    missing = [lang for lang, items in clips.items() if not items]
    if missing:
        raise ValueError(f"{golden_dir}: no turn JSON files with final_lang {missing}")
    return clips


def transcribe_all(checkpoint: Path, lang: str, audio_by_clip: dict[str, "object"]) -> dict[str, str]:
    """Load one checkpoint directory through src/asr.py and transcribe every clip."""
    import torch

    from src import asr

    asr.CHECKPOINTS[lang] = str(checkpoint)
    bundle = asr.load(lang)
    try:
        return {clip_id: bundle.transcribe(audio) for clip_id, audio in audio_by_clip.items()}
    finally:
        bundle.unload()
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-root", required=True, type=Path, help="patched checkpoints from the old repos")
    ap.add_argument("--new-root", required=True, type=Path, help="patched checkpoints from the lingkodai org")
    ap.add_argument("--golden-dir", required=True, type=Path, help="an earlier golden_check.py --out directory")
    args = ap.parse_args()

    print_environment(args)

    for root in (args.old_root, args.new_root):
        for lang in LANGS:
            if not (root / lang / "config.json").is_file():
                print(f"ERROR: no checkpoint at {root / lang} (run scripts/patch_asr_tokenizers.py)", file=sys.stderr)
                return 1
    if not args.golden_dir.is_dir():
        print(f"ERROR: golden dir not found: {args.golden_dir}", file=sys.stderr)
        return 1

    from src.audio import load_audio

    try:
        clips = load_clips(args.golden_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    diffs = 0
    total = 0
    for lang in LANGS:
        audio_by_clip = {clip_id: load_audio(path) for clip_id, path in clips[lang]}
        print(f"[{lang}] {len(audio_by_clip)} clips")
        old = transcribe_all(args.old_root / lang, lang, audio_by_clip)
        new = transcribe_all(args.new_root / lang, lang, audio_by_clip)
        for clip_id in audio_by_clip:
            total += 1
            if old[clip_id] == new[clip_id]:
                print(f"  {clip_id}: identical")
            else:
                diffs += 1
                print(f"  {clip_id}: DIFFERS")
                print(f"    old: {old[clip_id]}")
                print(f"    new: {new[clip_id]}")

    print()
    print(f"{total - diffs}/{total} transcripts identical")
    return 0 if diffs == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
