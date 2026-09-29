#!/usr/bin/env python3
"""Download the private eval assets and verify them against their manifest.

WHY THIS EXISTS
---------------
The golden check and TTS need files that cannot live in this public GitHub
repo: the three locked TTS reference clips (Philippine Languages Database
speakers, CC-BY-NC, research use only), DS6_augmented.xlsx (the evaluated
Scenario 2 run) and the 24 source clips of the golden conversation
(CONV-MT-001, 8 turns per language). They live in one private Hugging Face
dataset repo, lingkodai/lingkodai-eval-assets, with a MANIFEST.sha256 listing
every file's sha256. See CHANGELOG.md, 2026-09-29.

WHAT IT DOES
------------
Downloads the repo with snapshot_download, checks every file in
MANIFEST.sha256 exists with the listed sha256, and prints the values the
pipeline needs:

  TTS_REF_DIR       for src/tts.py
  --ds6             for scripts/golden_check.py
  --audio-root-map  for scripts/golden_check.py, remapping the JOJIE paths
                    recorded in DS6 onto the download

Downloads no model weights, never touches CUDA and needs no GPU.

READING THE RESULT
------------------
  exit 0   every manifest entry present and matching
  exit 2   a file is missing or its sha256 differs; do not use the download
  exit 1   the repo could not be read (token, name or access problem)

USAGE
-----
    python scripts/fetch_eval_assets.py --out outputs/eval_assets

The repo is private. The token is read from HF_TOKEN in the environment (a
fine-grained read-only token that includes this dataset repo); it is
deliberately not a command-line flag, so it never lands in shell history.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path

import huggingface_hub

REPO = "lingkodai/lingkodai-eval-assets"
MANIFEST = "MANIFEST.sha256"

# The JOJIE prefix recorded in DS6's source_audio_path. The dataset repo keeps
# the layout below it under golden/audio/.
DS6_AUDIO_ROOT = "/home2/msds2026/ibucayan/Capstone/PRC_synthetic_qa"

# The only conversation whose audio is in the repo.
CONVERSATION = "CONV-MT-001"

TTS_REF_SUBDIR = "tts_ref"
DS6_RELPATH = "golden/DS6_augmented.xlsx"
GOLDEN_AUDIO_SUBDIR = "golden/audio"

# 3 reference clips + DS6 + 24 golden clips, as staged.
EXPECTED_ENTRIES = 28
EXPECTED_TTS_REF = 3
EXPECTED_GOLDEN_AUDIO = 24


def parse_manifest(text: str) -> dict[str, str]:
    """Parse `sha256sum`-style lines into {relative path: sha256}.

    Raises on a malformed line or a duplicate path rather than skipping it."""
    entries: dict[str, str] = {}
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            raise ValueError(f"{MANIFEST} line {n} is malformed: {line!r}")
        if rel in entries:
            raise ValueError(f"{MANIFEST} lists {rel} twice")
        entries[rel] = digest
    return entries


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: Path) -> list[str]:
    """Check root against its manifest. Returns a list of problems, empty if clean.

    Checks the expected counts too, so a manifest that lost a file (and so
    still verifies) does not pass."""
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        return [f"{manifest_path} does not exist"]
    entries = parse_manifest(manifest_path.read_text(encoding="utf-8"))

    problems = []
    for rel, digest in sorted(entries.items()):
        path = root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif (got := sha256_file(path)) != digest:
            problems.append(f"sha256 mismatch: {rel} (manifest {digest[:12]}, file {got[:12]})")

    counts = {
        "entries": (len(entries), EXPECTED_ENTRIES),
        "tts_ref clips": (sum(r.startswith(TTS_REF_SUBDIR + "/") for r in entries), EXPECTED_TTS_REF),
        "golden audio clips": (sum(r.startswith(GOLDEN_AUDIO_SUBDIR + "/") for r in entries), EXPECTED_GOLDEN_AUDIO),
        "DS6": (int(DS6_RELPATH in entries), 1),
    }
    for what, (got, want) in counts.items():
        if got != want:
            problems.append(f"{MANIFEST} has {got} {what}, expected {want}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="download directory")
    args = ap.parse_args()
    out = args.out.resolve()

    print("Environment")
    print(f"  python           {sys.version.split()[0]} ({platform.platform()})")
    print(f"  huggingface_hub  {huggingface_hub.__version__}")
    print("  torch            not used (no model is loaded)")
    print(f"  repo             {REPO} (dataset)")
    print(f"  out              {out}")
    print()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set (the eval-assets repo is private).", file=sys.stderr)
        return 1

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        snapshot_download(REPO, repo_type="dataset", token=token, local_dir=str(out))
    except HfHubHTTPError as exc:
        print(f"ERROR: could not download {REPO}: {exc}", file=sys.stderr)
        print("Check that HF_TOKEN can read this dataset repo.", file=sys.stderr)
        return 1

    problems = verify(out)
    if problems:
        print(f"FAILED: {len(problems)} problem(s) in {out}:")
        for p in problems:
            print(f"  {p}")
        return 2

    print(f"OK: all {EXPECTED_ENTRIES} files in {MANIFEST} present and matching.")
    print()
    print("Point the pipeline at the download:")
    print(f"  export TTS_REF_DIR={out / TTS_REF_SUBDIR}")
    print("  python scripts/golden_check.py \\")
    print(f"      --ds6 {out / DS6_RELPATH} \\")
    print(f"      --conversation-id {CONVERSATION} \\")
    print(f"      --audio-root-map {DS6_AUDIO_ROOT}={out / GOLDEN_AUDIO_SUBDIR} \\")
    print("      --out outputs/golden_check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
