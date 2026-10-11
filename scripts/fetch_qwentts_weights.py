#!/usr/bin/env python3
"""Download the qwentts.cpp GGUF weights at a pinned revision and verify them.

Serveurperso/Qwen3-TTS-GGUF (public) at REVISION. Each file is checked
against the SHA-256 the Hub records for it at that revision (Git LFS
metadata, read 2026-10-11); a mismatch raises. F32 is the default backend
weight; Q8_0 is the extra benchmark arm.

    python scripts/fetch_qwentts_weights.py --out /content/models/gguf            # F32 and Q8_0
    python scripts/fetch_qwentts_weights.py --out /content/models/gguf --quant F32
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "Serveurperso/Qwen3-TTS-GGUF"
REVISION = "b7ee2e8c7459c3bea99da23e3d178125a7d1713c"
FILES = {
    "F32": {
        "qwen-talker-1.7b-base-F32.gguf": "7f563aa562e6dd2327e35e10ee7c050760495d2eca654d21f1d5b1460bd6528a",
        "qwen-tokenizer-12hz-F32.gguf": "b16b95557c7c7340a121757bd6855b9609e1cf4ad3fad0778b89393293ae5f3d",
    },
    "Q8_0": {
        "qwen-talker-1.7b-base-Q8_0.gguf": "4b9a33a236908dd9435a42f7a396e38038329d053b704342a6413c08544c4fda",
        "qwen-tokenizer-12hz-Q8_0.gguf": "1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94",
    },
}


def sha256_of(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quant", choices=sorted(FILES), action="append", default=None,
                    help="repeatable; default F32 and Q8_0")
    args = ap.parse_args()

    from huggingface_hub import __version__ as hub_version
    from huggingface_hub import hf_hub_download

    quants = args.quant or ["F32", "Q8_0"]
    print(f"repo         {REPO}\nrevision     {REVISION}\nout          {args.out}\nquants       {quants}")
    print(f"huggingface_hub {hub_version}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for q in quants:
        for name, expected in FILES[q].items():
            path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=args.out))
            got = sha256_of(path)
            ok = got == expected
            print(f"{'OK  ' if ok else 'FAIL'} {name}  {path.stat().st_size / 1e9:.2f} GB  sha256 {got}")
            if not ok:
                failed.append((name, expected, got))
    if failed:
        for name, expected, got in failed:
            print(f"ERROR: {name}: expected sha256 {expected}, got {got}", file=sys.stderr)
        return 1
    print("\nAll files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
