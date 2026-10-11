#!/usr/bin/env python3
"""Validate patch 0001 (ICL non_streaming_mode=True layout) against qwen-tts.

Runs qwentts.cpp's own cossim harness (tests/debug-clone-cossim.py: greedy
ICL clone, Python qwen-tts and C++ qwen-tts dumping the same intermediates)
and then checks the dumps itself, because the harness's cosine helper
truncates both arrays to the shorter one and so cannot see a layout that has
the wrong number of rows. A case passes when:

  - prompt ids match exactly (same length, same ids)
  - the talker input embedding has the same shape on both sides, and its
    shape decomposes into the expected layout (role + codec prefix, then
    the ICL block for that layout)
  - for the non_streaming_mode=True layout, every text row (ref text +
    text + tts_eos, over codec_pad) has cosine >= 0.9999; these rows do not
    depend on the reference codes, which the two codec encoders do not
    reproduce exactly (upstream's own F32 CUDA clone log: 82% exact)
  - prefill logits have the same shape and cosine >= 0.999

Cases, each run with a fresh model load:
  control   upstream layout (non_streaming_mode=False), the harness's English
            example: shows the harness and checks behave as upstream reports
  patched   the same example with --non-streaming
  ceb       the patched layout with LingkodAI's padded ceb reference (24 kHz,
            prepared as src/tts_cpp.py does), its locked ref_text, a ceb
            sentence through the parity frontend, --lang auto (needs --ref-dir)

The harness expects ../build/qwen-tts, ../models/*.gguf,
../checkpoints/Qwen3-TTS-12Hz-1.7B-Base and ../../Qwen3-TTS/qwen_tts relative
to its tests/ directory; this script creates those links (the last one to
the installed qwen-tts package, the code arm A runs) and refuses to replace
anything that is not already the right link.

    python scripts/validate_qwentts_patch.py --qwentts-dir build/qwentts.cpp \\
        --gguf-dir /content/models/gguf --checkpoint /path/to/Qwen3-TTS-12Hz-1.7B-Base \\
        --ref-dir "$TTS_REF_DIR" --out outputs/patch_validation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TEXT_ROW_COS_MIN = 0.9999
LOGITS_COS_MIN = 0.999
CEB_SENTENCE = "Dad-a ang imong aplikasyon ug usa ka balido nga ID sa opisina sa PRC."


def load_dump(path: Path) -> np.ndarray:
    """Read the harness's dump format: int32 ndim, int32 shape..., f32 data."""
    raw = np.fromfile(path, dtype=np.uint8)
    ndim = int(np.frombuffer(raw[0:4], dtype=np.int32)[0])
    shape = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * ndim], dtype=np.int32))
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.float32).reshape(shape)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0


def check_dumps(cpp_dir: Path, py_dir: Path, non_streaming: bool) -> dict:
    """The checks listed in the module docstring, on one pair of dump dirs."""
    out: dict = {"checks": {}}
    checks = out["checks"]

    ids_c, ids_p = load_dump(cpp_dir / "prompt-ids.bin").ravel(), load_dump(py_dir / "prompt-ids.bin").ravel()
    checks["prompt_ids_exact"] = ids_c.shape == ids_p.shape and bool(np.array_equal(ids_c, ids_p))

    emb_c, emb_p = load_dump(cpp_dir / "talker-input-embed.bin"), load_dump(py_dir / "talker-input-embed.bin")
    emb_c, emb_p = emb_c.reshape(-1, emb_c.shape[-1]), emb_p.reshape(-1, emb_p.shape[-1])
    out["embed_shape_cpp"], out["embed_shape_py"] = list(emb_c.shape), list(emb_p.shape)
    checks["embed_shape_equal"] = emb_c.shape == emb_p.shape

    n_text = len(ids_p) - 8  # input_id[3:-5]
    n_ref = load_dump(py_dir / "ref-ids.bin").size - 5  # ref_id[3:-2]
    ref_t = load_dump(cpp_dir / "ref-codes.bin").shape[-1]
    text_lens, codec_lens = n_ref + n_text + 1, 1 + ref_t
    icl_rows = text_lens + codec_lens if non_streaming else codec_lens
    prefix = emb_p.shape[0] - icl_rows
    out.update(n_text=n_text, n_ref_text=n_ref, ref_frames=ref_t, text_lens=text_lens, prefix_rows=prefix)
    # role (3) + [think/nothink, think_bos, (language), think_eos, speaker, codec_pad]
    checks["layout_decomposes"] = prefix in (8, 9)

    if non_streaming and checks["embed_shape_equal"] and checks["layout_decomposes"]:
        rows = range(prefix, prefix + text_lens)
        row_cos = [cos(emb_c[r], emb_p[r]) for r in rows]
        out["text_rows_cos_min"] = min(row_cos)
        checks["text_rows_cos"] = out["text_rows_cos_min"] >= TEXT_ROW_COS_MIN
    out["embed_cos"] = cos(emb_c, emb_p) if checks["embed_shape_equal"] else None

    lg_c, lg_p = load_dump(cpp_dir / "talker-logits-prefill.bin"), load_dump(py_dir / "talker-logits-prefill.bin")
    out["logits_cos"] = cos(lg_c, lg_p) if lg_c.shape == lg_p.shape else None
    checks["logits_cos"] = out["logits_cos"] is not None and out["logits_cos"] >= LOGITS_COS_MIN
    out["passed"] = all(checks.values())
    return out


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"{link} points at {link.resolve()}, expected {target}; remove it first")
        return
    if link.exists():
        raise RuntimeError(f"{link} exists and is not a link; move it away first")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_ceb_case(ref_dir: Path, work: Path) -> dict:
    import soundfile as sf

    from src import tts, tts_cpp

    padded = tts._pad_ref_audio_with_silence(tts.find_ref_clip(ref_dir, tts.REF_CLIP_FILENAMES["ceb"]))
    wav = work / "ceb_ref_24k.wav"
    sf.write(wav, tts_cpp._load_ref_24k(padded), tts_cpp.SAMPLE_RATE, subtype="FLOAT")
    (work / "ceb_ref.txt").write_text(tts.REF_TEXT_CONFIG["ceb"]["ref_text"], encoding="utf-8")
    sentence = tts_cpp.text_segments(CEB_SENTENCE, "ceb", tts.load_init_set(REPO_ROOT / "data" / "init_set_ds6.json"),
                                     "parity")[0][0]
    (work / "ceb_prompt.txt").write_text(sentence, encoding="utf-8")
    return {"--ref-wav": str(wav), "--ref-text": str(work / "ceb_ref.txt"), "--prompt": str(work / "ceb_prompt.txt"),
            "--lang": "auto"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwentts-dir", type=Path, default=REPO_ROOT / "build" / "qwentts.cpp")
    ap.add_argument("--gguf-dir", type=Path, required=True, help="holds qwen-talker-1.7b-base-<Q>.gguf and codec")
    ap.add_argument("--checkpoint", type=Path, required=True, help="local Qwen3-TTS-12Hz-1.7B-Base snapshot")
    ap.add_argument("--quant", default="F32")
    ap.add_argument("--ref-dir", type=Path, default=None, help="TTS_REF_DIR, for the ceb case")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    qdir = args.qwentts_dir.resolve()
    tests_dir = qdir / "tests"
    spec = importlib.util.find_spec("qwen_tts")
    if spec is None or not spec.submodule_search_locations:
        print("ERROR: qwen-tts is not installed", file=sys.stderr)
        return 1
    qwen_tts_pkg = Path(list(spec.submodule_search_locations)[0])
    print("validate_qwentts_patch")
    for k, v in (("qwentts dir", qdir), ("gguf dir", args.gguf_dir), ("checkpoint", args.checkpoint),
                 ("quant", args.quant), ("qwen_tts pkg", qwen_tts_pkg), ("ref dir", args.ref_dir), ("out", args.out)):
        print(f"  {k:<13} {v}")
    for p in (qdir / "build" / "qwen-tts", tests_dir / "debug-clone-cossim.py",
              args.gguf_dir / f"qwen-talker-1.7b-base-{args.quant}.gguf",
              args.gguf_dir / f"qwen-tokenizer-12hz-{args.quant}.gguf", args.checkpoint / "config.json"):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    ensure_link(qdir / "models", args.gguf_dir)
    ensure_link(qdir / "checkpoints" / "Qwen3-TTS-12Hz-1.7B-Base", args.checkpoint)
    ensure_link(qdir.parent / "Qwen3-TTS" / "qwen_tts", qwen_tts_pkg)

    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="qwentts_validate_", dir=args.out))
    cases = [("control", False, {}), ("patched", True, {})]
    if args.ref_dir:
        cases.append(("ceb", True, prepare_ceb_case(args.ref_dir, work)))
    else:
        print("  ceb case     skipped: no --ref-dir")

    report = {}
    for name, non_streaming, extra in cases:
        cmd = [sys.executable, "debug-clone-cossim.py", "--quant", args.quant]
        for k, v in extra.items():
            cmd += [k, v]
        if non_streaming:
            cmd.append("--non-streaming")
        print(f"\n== case {name}: {' '.join(cmd)}")
        log_path = args.out / f"cossim_{name}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(log_path, "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=tests_dir, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        harness_lines = [ln for ln in log_path.read_text(errors="replace").splitlines() if ln.startswith("[Cossim]")]
        for ln in harness_lines:
            print("  " + ln)
        if rc != 0:
            report[name] = {"passed": False, "error": f"harness exited {rc}, see {log_path}"}
            print(f"  FAILED: harness exited {rc}; log {log_path}")
            continue
        result = check_dumps(tests_dir / "cpp" / "clone", tests_dir / "python" / "clone", non_streaming)
        result["harness"] = harness_lines
        report[name] = result
        print(f"  checks: {result['checks']}")
        print(f"  embed {result['embed_shape_cpp']} vs {result['embed_shape_py']}, prefix {result['prefix_rows']}, "
              f"text rows {result['text_lens']}, ref frames {result['ref_frames']}, "
              f"logits cos {result['logits_cos']}, text-row cos min {result.get('text_rows_cos_min')}")
        print("  PASS" if result["passed"] else "  FAIL")

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    ok = all(r.get("passed") for r in report.values())
    print(f"\n{'ALL CASES PASS' if ok else 'VALIDATION FAILED'}; report {args.out / 'report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
