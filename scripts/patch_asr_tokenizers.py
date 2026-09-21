#!/usr/bin/env python3
"""Make the three ASR checkpoints loadable under transformers 4.57.3.

The beabucayan/lingkodai-whisper-{ceb,fil,eng} repos store `extra_special_tokens`
in tokenizer_config.json as a LIST. transformers 4.57.3 (pinned by qwen-tts, so
it cannot change) expects a dict there and fails with
    AttributeError: 'list' object has no attribute 'keys'
A list of special tokens belongs under `additional_special_tokens`.

src/asr.py is Bea's file and stays untouched. It already reads ASR_CEB, ASR_FIL
and ASR_ENG, so this script builds a patched copy of each checkpoint folder
(weights symlinked from the HF cache, only tokenizer_config.json rewritten) and
prints the exports that point asr.py at them. See DECISIONS.md #9.

    python scripts/patch_asr_tokenizers.py --out outputs/asr_patched

Each patched tokenizer is checked before it is trusted: the special-token ids
must match base openai/whisper-medium (ceb: the shared ids, plus <|cebuano|> at
51865). Any mismatch raises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOS = {
    "ceb": "beabucayan/lingkodai-whisper-ceb",
    "fil": "beabucayan/lingkodai-whisper-fil",
    "eng": "beabucayan/lingkodai-whisper-eng",
}
BASE_REPO = "openai/whisper-medium"
CEB_TOKEN = "<|cebuano|>"
CEB_TOKEN_ID = 51865


def patch_tokenizer_config(cfg: dict) -> dict:
    """Return a copy with a list-valued extra_special_tokens moved to
    additional_special_tokens. A dict value is left alone; a clash with an
    existing additional_special_tokens raises rather than guessing."""
    out = dict(cfg)
    extra = out.get("extra_special_tokens")
    if not isinstance(extra, list):
        return out
    if out.get("additional_special_tokens"):
        raise ValueError("both a list extra_special_tokens and additional_special_tokens are present")
    del out["extra_special_tokens"]
    out["additional_special_tokens"] = extra
    return out


def build_patched_dir(snapshot: Path, dest: Path) -> None:
    """Mirror snapshot into dest with symlinks, except tokenizer_config.json,
    which is rewritten. The original is kept beside it as .orig."""
    for src in sorted(snapshot.rglob("*")):
        rel = src.relative_to(snapshot)
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        if rel.as_posix() == "tokenizer_config.json":
            cfg = json.loads(src.read_text(encoding="utf-8"))
            (dest / "tokenizer_config.json.orig").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            target.write_text(json.dumps(patch_tokenizer_config(cfg), indent=2), encoding="utf-8")
        else:
            target.symlink_to(os.path.realpath(src))


def verify(lang: str, patched: Path, base_tok) -> None:
    from transformers import WhisperTokenizerFast

    tok = WhisperTokenizerFast.from_pretrained(str(patched))
    base_specials = {t: base_tok.convert_tokens_to_ids(t) for t in base_tok.all_special_tokens}
    bad = {t: (i, tok.convert_tokens_to_ids(t)) for t, i in base_specials.items() if tok.convert_tokens_to_ids(t) != i}
    if bad:
        raise RuntimeError(f"{lang}: special-token ids differ from {BASE_REPO} (base id, patched id): {bad}")
    if lang == "ceb":
        got = tok.convert_tokens_to_ids(CEB_TOKEN)
        if got != CEB_TOKEN_ID:
            raise RuntimeError(f"ceb: {CEB_TOKEN} is at id {got}, expected {CEB_TOKEN_ID}")
    sample = "ngunit ang alimango'y hindi maaaring umakyat sa punongkahoy"
    if lang != "ceb" and tok(sample).input_ids != base_tok(sample).input_ids:
        raise RuntimeError(f"{lang}: tokenizing a sample sentence differs from {BASE_REPO}")
    print(f"  {lang}: verified ({len(tok)} tokens, {len(base_specials)} base special tokens match)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set (the ASR repos are private).", file=sys.stderr)
        return 1

    import transformers
    from huggingface_hub import snapshot_download
    from transformers import WhisperTokenizerFast

    print(f"transformers {transformers.__version__}, out = {args.out.resolve()}")
    base_tok = WhisperTokenizerFast.from_pretrained(BASE_REPO)

    exports = []
    for lang, repo in REPOS.items():
        print(f"{lang}: {repo}")
        snapshot = Path(snapshot_download(repo, token=token))
        dest = args.out / lang
        build_patched_dir(snapshot, dest)
        verify(lang, dest, base_tok)
        exports.append(f"export ASR_{lang.upper()}={dest.resolve()}")

    print("\nPoint src/asr.py at the patched copies:")
    print("\n".join(exports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
