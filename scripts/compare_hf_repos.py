#!/usr/bin/env python3
"""Check that the lingkodai org copies of the models match what the pipeline uses today.

WHY THIS EXISTS
---------------
The three ASR checkpoints were moved from Bea's personal Hugging Face account
(beabucayan/lingkodai-whisper-{ceb,fil,eng}) to the stakeholders' org
(lingkodai/lingkodai-whisper-{ceb,fil,eng}), and the audio LID head was
uploaded there as lingkodai/lingkodai-audio-lid-head. Before the code is
pointed at the org, the org copies must be the same files: a re-upload of a
different local checkpoint would load and run without raising, and only
produce different transcripts. (The tokenizer fix in
scripts/patch_asr_tokenizers.py is applied to local copies, not stored in
either set of repos, so a tokenizer_config.json match means both carry the
same unpatched config and the patch step still applies.)

The audio LID head used to be committed as models/deeper_50chunks.pt. Since
2026-10-12 src/lid_audio.py downloads it from the org repo instead, pinned to
a commit and SHA-256, and the committed copy is gone; this script now looks
for that recorded SHA-256 among the org repo's files.

WHAT IT DOES
------------
Reads each repo's file listing with its content hashes from the Hub API
(sha256 for LFS files, the git blob id for small files). It downloads no
weights, never touches CUDA and needs no GPU, so it can run anywhere with
network access and a token.

  ASR   each beabucayan/<name> repo is compared file by file with lingkodai/<name>
  LID   the recorded sha256 of deeper_50chunks.pt (formerly committed in
        models/) is looked for among the files of
        lingkodai/lingkodai-audio-lid-head

README.md and .gitattributes differences are reported but do not fail the
check: model cards are expected to differ and neither file is read when loading.

READING THE RESULT
------------------
  exit 0   every load-relevant file matches; safe to switch the repo IDs
  exit 2   at least one mismatch or missing file; do not switch, report it
  exit 1   a repo could not be read (token, name or access problem)

USAGE
-----
    python scripts/compare_hf_repos.py

The org repos are private (checked 2026-09-28; the personal repos were public
then). The token must be able to read every repo that is private. It is read from HF_TOKEN in the environment, or
from a `huggingface-cli login` cache; it is deliberately not a command-line
flag, so the token never lands in shell history.

Paste the full output into the run record.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]

ASR_NAMES = ("lingkodai-whisper-ceb", "lingkodai-whisper-fil", "lingkodai-whisper-eng")
LID_HEAD_NAME = "lingkodai-audio-lid-head"
# SHA-256 of the head formerly committed at models/deeper_50chunks.pt; the same
# value is pinned as src/lid_audio.py's HEAD_SHA256 (not imported: that module
# pulls in torch, which this script avoids).
LID_HEAD_FILE = "deeper_50chunks.pt"
LID_HEAD_SHA256 = "101c32f02a254a1596617d44a98c9eccbc0f761862232b787ddf31e2b066dc99"

# Not read by from_pretrained; a difference here is informational only.
NON_LOAD_FILES = {"README.md", ".gitattributes"}


def file_hashes(api: HfApi, repo: str, token: str) -> tuple[str, dict[str, tuple[str, str, int]]]:
    """Return (commit sha, {filename: (hash kind, hash, size)}) for a model repo.

    LFS files are keyed by their content sha256. Small files are keyed by the
    git blob id, which is a hash of the content, so identical content gives an
    identical id in any repo.
    """
    info = api.model_info(repo, files_metadata=True, token=token)
    files: dict[str, tuple[str, str, int]] = {}
    for s in info.siblings or []:
        if s.lfs is not None:
            files[s.rfilename] = ("sha256", s.lfs.sha256, s.lfs.size)
        else:
            if s.blob_id is None:
                raise RuntimeError(f"{repo}: no hash returned for {s.rfilename}")
            files[s.rfilename] = ("blob", s.blob_id, s.size or 0)
    if not files:
        raise RuntimeError(f"{repo}: the repo lists no files")
    return info.sha, files


def compare_pair(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return (report lines, load-relevant problems) for two file listings."""
    lines: list[str] = []
    problems: list[str] = []
    for name in sorted(set(old) | set(new)):
        informational = name in NON_LOAD_FILES
        tag = "  (not loaded, informational)" if informational else ""
        if name not in new:
            status = "MISSING in org copy"
        elif name not in old:
            status = "EXTRA in org copy"
        elif old[name][0] != new[name][0]:
            status = "CANNOT COMPARE (LFS in one repo only; sizes " \
                     f"{old[name][2]:,} vs {new[name][2]:,})"
        elif old[name][1] == new[name][1]:
            status = "match"
        else:
            status = "DIFFERS"
        lines.append(f"    {status:<10} {name}{tag}")
        if status != "match" and not informational:
            problems.append(f"{name}: {status}")
    return lines, problems



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--old-owner", default="beabucayan")
    ap.add_argument("--new-owner", default="lingkodai")
    args = ap.parse_args()

    print("Hugging Face repo comparison")
    print(f"  python           {platform.python_version()}")
    print(f"  huggingface_hub  {huggingface_hub.__version__}")
    print("  torch            not used (no weights are loaded)")
    print(f"  repo root        {ROOT}")
    print(f"  LID head sha256  {LID_HEAD_SHA256} (recorded)")

    token = huggingface_hub.get_token()
    if not token:
        print("\nNo Hugging Face token found. Set HF_TOKEN (see .env.example) or run "
              "`huggingface-cli login`, with read access to the lingkodai org repos.", file=sys.stderr)
        return 1
    source = "HF_TOKEN environment variable" if os.environ.get("HF_TOKEN") else "login cache"
    print(f"  token source     {source}")

    api = HfApi()
    all_problems: list[str] = []

    def read(repo: str):
        try:
            return file_hashes(api, repo, token)
        except HfHubHTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            raise SystemExit(
                f"\nCould not read {repo} (HTTP {code}). Check the repo name and that "
                f"the token has read access to it.\n{exc}"
            ) from None

    for name in ASR_NAMES:
        old_repo, new_repo = f"{args.old_owner}/{name}", f"{args.new_owner}/{name}"
        old_sha, old_files = read(old_repo)
        new_sha, new_files = read(new_repo)
        print(f"\n{name}")
        print(f"  old  {old_repo}  commit {old_sha}")
        print(f"  new  {new_repo}  commit {new_sha}")
        lines, problems = compare_pair(old_files, new_files)
        print("\n".join(lines))
        all_problems += [f"{name}: {p}" for p in problems]

    lid_repo = f"{args.new_owner}/{LID_HEAD_NAME}"
    local_sha = LID_HEAD_SHA256
    lid_sha, lid_files = read(lid_repo)
    print(f"\n{LID_HEAD_NAME}")
    print(f"  recorded {LID_HEAD_FILE}  sha256 {local_sha}")
    print(f"  repo  {lid_repo}  commit {lid_sha}")
    matches = [f for f, (kind, h, _) in lid_files.items() if kind == "sha256" and h == local_sha]
    for f, (kind, h, size) in sorted(lid_files.items()):
        mark = "MATCH " if f in matches else "      "
        print(f"    {mark}{f}  {kind} {h[:16]}...  {size:,} bytes")
    if not matches:
        all_problems.append(f"{LID_HEAD_NAME}: no file matches the recorded {LID_HEAD_FILE}")

    print("\nVerdict")
    if all_problems:
        print("  NOT identical. Do not switch the repo IDs yet. Problems:")
        for p in all_problems:
            print(f"    {p}")
        return 2
    print("  Every load-relevant file in the org copies matches the personal repos,")
    print("  weights and tokenizer files included, and the LID head file matches")
    print(f"  the recorded {LID_HEAD_FILE}. Safe to switch the repo IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
