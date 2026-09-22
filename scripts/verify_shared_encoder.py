#!/usr/bin/env python3
"""Check whether the three ASR checkpoints really do share one encoder.

WHY THIS EXISTS
---------------
Bea's guidance is that the audio LID encoder and all three STT encoders are
frozen, so the pipeline could hold one encoder instead of four. That is true
only if "frozen" meant frozen DURING FINE-TUNING, so the encoder weights were
never updated and each checkpoint still carries the original
openai/whisper-medium encoder byte for byte.

If any language was fine-tuned with its encoder unfrozen, sharing one encoder
swaps the wrong weights into that language's model. Nothing raises. The model
still produces fluent-looking text, just worse text, and no log line says so.
That is the reason this check exists rather than a comment saying the encoders
are probably the same.

WHAT IT DOES
------------
Loads each checkpoint on CPU in fp32, hashes its encoder state dict, and reports
which fingerprints match. Also prints vocab and decoder embedding sizes, which
is where the Cebuano <|cebuano|> token at id 51865 shows up: that difference is
expected and lives in the DECODER, so it has no bearing on encoder sharing.

It never touches CUDA, so it is safe to run while someone else is using the GPU.
It does hold roughly 3GB of host RAM at a time, one model at a time, freeing each
before loading the next.

READING THE RESULT
------------------
  ceb == fil == eng            sharing across the three ASR models is safe
  ceb == fil == eng == base    the LID encoder is also the same weights, though
                               DECISIONS.md item 2 still keeps LID separate
                               because of the fp32/fp16 difference, not identity
  any mismatch                 do not share. Report which pair differs and stop.

USAGE
-----
    conda activate lingkod-e2e
    python scripts/verify_shared_encoder.py

If the repos are gated, pass a read token:

    python scripts/verify_shared_encoder.py --token hf_yourTokenHere

Note that hf_yourTokenHere above is a PLACEHOLDER. Replace the whole thing with
a real token, and do not type the word placeholder or any angle brackets.

To check local checkpoint directories instead of the Hub:

    python scripts/verify_shared_encoder.py --ceb /path/to/ceb --fil /path/to/fil --eng /path/to/eng

Write the output into the run record. A later reader needs to see the
fingerprints, not a claim that they matched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys

import torch
from transformers import WhisperForConditionalGeneration

DEFAULT_REPOS = {
    "ceb": "beabucayan/lingkodai-whisper-ceb",
    "fil": "beabucayan/lingkodai-whisper-fil",
    "eng": "beabucayan/lingkodai-whisper-eng",
    "base": "openai/whisper-medium",
}


def fingerprint_encoder(repo: str, token: str | None = None) -> tuple[str, int, int, int]:
    """Return (sha256 of encoder weights, param count, vocab size, decoder embedding rows).

    Everything is loaded in fp32 regardless of how it was saved. That matters:
    if one checkpoint was stored in fp16 and another in fp32, comparing them as
    stored would report a difference that is only a storage format. Upcasting
    both to fp32 is deterministic, so identical weights hash identically.

    The keys are sorted before hashing because a state dict's iteration order is
    not guaranteed stable, and an unsorted hash would differ run to run even for
    a single unchanged file.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=token,
    )

    encoder_sd = model.model.encoder.state_dict()
    h = hashlib.sha256()
    n_params = 0
    for key in sorted(encoder_sd):
        tensor = encoder_sd[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(tensor.numpy().tobytes())
        n_params += tensor.numel()

    vocab_size = model.config.vocab_size
    embed_rows = model.model.decoder.embed_tokens.weight.shape[0]

    del encoder_sd, model
    gc.collect()

    return h.hexdigest(), n_params, vocab_size, embed_rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--token", default=None, help="Hugging Face read token, if the repos are gated")
    ap.add_argument("--ceb", default=DEFAULT_REPOS["ceb"])
    ap.add_argument("--fil", default=DEFAULT_REPOS["fil"])
    ap.add_argument("--eng", default=DEFAULT_REPOS["eng"])
    ap.add_argument("--base", default=DEFAULT_REPOS["base"])
    ap.add_argument("--skip-base", action="store_true",
                    help="skip openai/whisper-medium, for when only the three fine-tunes matter")
    args = ap.parse_args()

    targets = {"ceb": args.ceb, "fil": args.fil, "eng": args.eng}
    if not args.skip_base:
        targets["base"] = args.base

    print("Encoder fingerprint check")
    print("Loading on CPU in fp32, one model at a time. Roughly 3GB of RAM each.\n")

    results: dict[str, tuple[str, int, int, int]] = {}
    for name, repo in targets.items():
        print(f"  {name:<5} loading {repo} ...", flush=True)
        try:
            results[name] = fingerprint_encoder(repo, token=args.token)
        except Exception as exc:
            print(f"\nFAILED to load {name} ({repo}): {exc}", file=sys.stderr)
            print("Stopping. A partial comparison is worse than none.", file=sys.stderr)
            return 1

    print("\nResults")
    print(f"  {'name':<6} {'encoder sha256':<20} {'enc params':>12} {'vocab':>8} {'dec emb':>9}")
    for name, (digest, n_params, vocab, emb) in results.items():
        print(f"  {name:<6} {digest[:16]}...  {n_params:>12,} {vocab:>8,} {emb:>9,}")

    asr_names = [n for n in ("ceb", "fil", "eng") if n in results]
    asr_digests = {results[n][0] for n in asr_names}

    print("\nVerdict")
    if len(asr_digests) == 1:
        print("  The three ASR encoders are byte identical.")
        print("  Sharing one encoder across them is SAFE. See DECISIONS.md item 2")
        print("  for how to implement it: assign the shared module to")
        print("  model.model.encoder on CPU, then move each model to CUDA.")
        shared_ok = True
    else:
        print("  The ASR encoders are NOT all identical:")
        for name in asr_names:
            print(f"    {name}: {results[name][0][:16]}...")
        print("  DO NOT share encoders. At least one checkpoint was fine-tuned")
        print("  with its encoder unfrozen, and sharing would silently degrade")
        print("  that language. Report this rather than working around it.")
        shared_ok = False

    if "base" in results and shared_ok:
        if results["base"][0] in asr_digests:
            print("\n  The base openai/whisper-medium encoder matches as well, which")
            print("  confirms the fine-tunes left the encoder untouched. The audio LID")
            print("  encoder still stays separate for now: DECISIONS.md keeps it in fp32")
            print("  because its head was trained on fp32 outputs and its label locks the")
            print("  whole conversation. That is a numerics decision, not an identity one.")
        else:
            print("\n  The base encoder DIFFERS from the fine-tunes, so the three moved")
            print("  together but away from base. Sharing across the three is still fine.")
            print("  Sharing with audio LID is not, if LID loads base weights.")

    print("\nRecord this output in the run log. A future reader needs the")
    print("fingerprints themselves, not an assurance that they matched.")
    return 0 if shared_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
