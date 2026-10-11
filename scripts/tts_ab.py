#!/usr/bin/env python3
"""Stage 7 A/B on DS6 answers: the evaluated qwen-tts path against qwentts.cpp.

Every arm speaks the same DS6 `rag_answer_native` text in its `final_lang`,
so the comparison isolates Stage 7 (no ASR, MT or RAG drift). Arms:

    A  hf         src/tts.py (qwen-tts, fp32), the evaluated path
    B  parity     src/tts_cpp.py, parity mode: same text path as A
    C  segmented  src/tts_cpp.py, segmented mode: split, length cap, retry

Three steps, each its own process so a model's VRAM is freed in between:

    python scripts/tts_ab.py select --ds6 DS6_augmented.xlsx --out outputs/tts_ab/selection.csv
    python scripts/tts_ab.py synth  --ds6 DS6_augmented.xlsx --selection outputs/tts_ab/selection.csv \\
        --arm C --label C-seg-F32 --seed 1234 --out outputs/tts_ab/C-seg-F32
    python scripts/tts_ab.py score  --ds6 DS6_augmented.xlsx --runs outputs/tts_ab/A-hf ... \\
        --out outputs/tts_ab/scores

select: the golden-check conversation (CONV-MT-001, all three languages), a
length-stratified sample per final_lang (verbalized-length quartiles, seeded),
ceb_01_mt_185 (DS6's one runaway) and every row with tts_rt_wer > 2.

synth: one WAV per row (plus per-segment WAVs for B and C), rows.jsonl with
wall time, RTF, audio length against the length expected from the verbalized
text, and for B and C the segment records and simulated play-ahead. Resumes
where it stopped; a cpp arm reuses the base seed recorded in run.json.

Resuming after a dropped runtime (the Colab notebook keeps every output on
Google Drive): a cut-off last line in rows.jsonl or a score cache is
reported and removed, and a recorded row whose WAV is missing or does not
match its recorded length is reported and synthesized again.

score: the DS6/DS7 judge, ported from the DS2/DS5 judge scripts: the
per-language fine-tuned Whisper-medium (lingkodai/lingkodai-whisper-{lang};
identity with JOJIE's best_checkpoints is not verified), base whisper-medium
processor, fp16, batch 8, max_length 440, no long-form chunking. The judge
stops transcribing long clips early, so WER counts only rows whose verbalized
text has at most 75 words (the DS6/DS7 error analysis cutoff). The
hypothesis is normalized as DS6's hyp_norm (scoring_normalize, plus
eval_side_normalize for eng; reproduces DS6 on 2,235/2,235 rows) and scored
against DS6's stored reference_norm. UTMOS22 via utmos-pytorch 0.1.0
UTMOSScoreTorch on every row (soundfile, first channel, scipy resample to
16 kHz), as DS6 was scored. DS6's own stored values for the same rows are
reported as a fourth column set.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# Vendored frontend modules import each other by top-level name (see src/tts.py).
sys.path.insert(0, str(REPO_ROOT / "src" / "tts_frontend"))

DEFAULT_INIT_SET = REPO_ROOT / "data" / "init_set_ds6.json"
GOLDEN_CONVERSATION = "CONV-MT-001"
RUNAWAY_CLIP = "ceb_01_mt_185"
HIGH_WER = 2.0
WER_MAX_REF_WORDS = 75
ARMS = {"A": ("hf", None), "B": ("cpp", "parity"), "C": ("cpp", "segmented")}

JUDGE_BASE_MODEL = "openai/whisper-medium"
JUDGE_REPOS = {lang: f"lingkodai/lingkodai-whisper-{lang}" for lang in ("ceb", "fil", "eng")}
JUDGE_WHISPER_LANG = {"fil": "tagalog", "eng": "english"}
JUDGE_CEB_TOKEN, JUDGE_CEB_TOKEN_ID = "<|cebuano|>", 51865
JUDGE_BATCH, JUDGE_MAX_LENGTH, JUDGE_SR = 8, 440, 16000
UTMOS_SR = 16000


def print_environment(extra: dict | None = None) -> dict:
    """Print and return paths and versions before any work."""
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = f"{props.name}, {props.total_memory / 1e9:.1f} GB"
        else:
            env["gpu"] = "none (CPU only)"
    except ImportError:
        env["torch"] = "not installed"
    from importlib import metadata

    # Versions from package metadata: importing qwen_tts here would pull in
    # torchaudio's CUDA extension.
    for dist in ("transformers", "accelerate", "qwen-tts", "librosa", "soundfile", "jiwer", "utmos-pytorch"):
        try:
            env[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[dist] = "not installed"
    env.update(extra or {})
    print("Environment")
    for k, v in env.items():
        print(f"  {k:<14} {v}")
    print()
    return env


def load_ds6(path: Path):
    from golden_check import load_ds6 as _load

    df = _load(path)
    missing = {"rag_answer_native", "verbalized_text", "reference_norm", "hyp_norm", "tts_rt_wer", "utmos22",
               "duration_sec"} - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} lacks the DS7 evaluation column(s) {sorted(missing)}; use DS6_augmented.xlsx")
    return df.set_index("clip_id", drop=False)


def ref_words(text) -> int:
    return len(str(text).split())


def read_jsonl(path: Path) -> list[dict]:
    """The records in a JSONL file this script appends to.

    A process that dies mid-append (a Colab runtime dropping while the file
    sits on Google Drive) can leave a cut-off last line. That line is
    reported and removed, so the next append starts on a clean line; a bad
    line anywhere else raises.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    records = []
    for k, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if k < len(lines) - 1:
                raise ValueError(f"{path}: record {k + 1} of {len(lines)} is not valid JSON") from None
            print(f"WARNING: {path}: removing a cut-off last line: {line[:80]!r}")
    if len(records) < len(lines) or (text and not text.endswith("\n")):
        write_jsonl(path, records)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=float) + "\n" for r in records),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def select_rows(df, per_stratum: int, seed: int):
    """Deterministic selection; returns a DataFrame, one row per clip."""
    import pandas as pd
    from golden_check import pick_conversation_id

    tags: dict[str, list[str]] = {}

    def tag(ids, name):
        for cid in ids:
            tags.setdefault(cid, []).append(name)

    conv = pick_conversation_id(df.reset_index(drop=True), GOLDEN_CONVERSATION)
    tag(df.index[df["conversation_id"] == conv], "golden")
    if RUNAWAY_CLIP in df.index:
        tag([RUNAWAY_CLIP], "runaway")
    tag(df.index[df["tts_rt_wer"] > HIGH_WER], "high_wer")

    chars = df["verbalized_text"].astype(str).str.len()
    for lang in ("ceb", "fil", "eng"):
        sub = df[df["final_lang"] == lang]
        quartile = pd.qcut(chars[sub.index], 4, labels=["q1", "q2", "q3", "q4"])
        for q in ("q1", "q2", "q3", "q4"):
            pool = sub[(quartile == q) & ~sub.index.isin(list(tags))]
            n = min(per_stratum, len(pool))
            tag(pool.sample(n=n, random_state=seed).index, f"sample_{q}")

    rows = []
    for cid in sorted(tags):
        r = df.loc[cid]
        words = ref_words(r["verbalized_text"])
        rows.append({
            "clip_id": cid,
            "final_lang": r["final_lang"],
            "tags": "+".join(tags[cid]),
            "chars": len(str(r["verbalized_text"])),
            "ref_words": words,
            "wer_eligible": words <= WER_MAX_REF_WORDS,
            "ds6_duration_sec": float(r["duration_sec"]),
            "ds6_utmos22": float(r["utmos22"]),
            "ds6_tts_rt_wer": float(r["tts_rt_wer"]),
        })
    return pd.DataFrame(rows)


def cmd_select(args) -> int:
    print_environment({"ds6": str(args.ds6), "out": str(args.out)})
    df = load_ds6(args.ds6)
    sel = select_rows(df, args.per_stratum, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(args.out, index=False)
    print(f"Selected {len(sel)} rows -> {args.out}")
    print(sel.groupby("final_lang").agg(rows=("clip_id", "size"), wer_eligible=("wer_eligible", "sum"),
                                        ds6_audio_min=("ds6_duration_sec", lambda s: round(s.sum() / 60, 1))))
    return 0


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


def _row_record(arm: str, label: str, sel_row, result, chars: int) -> dict:
    from src import tts_cpp

    audio_s = (len(result.audio) / result.sr) if result.ok else None
    expected_s = tts_cpp.FRAMES_PER_CHAR * chars / 12.5
    rec = {
        "arm": arm,
        "label": label,
        "clip_id": sel_row["clip_id"],
        "lang": sel_row["final_lang"],
        "tags": sel_row["tags"],
        "ok": bool(result.ok),
        "failure_reason": result.failure_reason,
        "wall_sec": result.wall_sec,
        "rtf": result.rtf,
        "audio_s": audio_s,
        "chars": chars,
        "expected_s": expected_s,
        "length_ratio": (audio_s / expected_s) if audio_s is not None and expected_s else None,
    }
    segments = getattr(result, "segments", None)
    if segments is not None:
        frames = [s["frames"] for s in segments if s["frames"] is not None]
        rec.update({
            "n_segments": len(segments),
            "speech_s": sum(frames) / 12.5 if result.ok else None,
            "cap_hit_attempts": sum(bool(a["cap_hit"]) for s in segments for a in s["attempts"]),
            "retries": sum(max(len(s["attempts"]) - 1, 0) for s in segments),
            "timeline": result.timeline,
            "segments": segments,
        })
    else:
        # A: audio exists only once the whole answer is done.
        rec["timeline"] = {"ttfa_s": result.wall_sec, "n_stalls": 0, "total_stall_s": 0.0} if result.ok else None
    return rec


def split_rows_missing_audio(records: list[dict], wav_dir: Path) -> tuple[list[dict], list[str]]:
    """(rows to keep, clip_ids to redo): a recorded ok row whose WAV is
    missing, unreadable or not as long as its recorded audio_s is redone (the
    row reached Drive, the WAV did not). Text-only rows are results; kept."""
    import soundfile as sf

    kept, redo = [], []
    for rec in records:
        if rec["ok"]:
            wav = wav_dir / f"{rec['clip_id']}.wav"
            try:
                info = sf.info(str(wav)) if wav.is_file() else None
            except RuntimeError:
                info = None
            if info is None or info.frames != round(rec["audio_s"] * info.samplerate):
                redo.append(rec["clip_id"])
                continue
        kept.append(rec)
    return kept, redo


def cmd_synth(args) -> int:
    import pandas as pd
    import soundfile as sf

    backend, mode = ARMS[args.arm]
    args.out.mkdir(parents=True, exist_ok=True)
    run_path, rows_path = args.out / "run.json", args.out / "rows.jsonl"
    prior = json.loads(run_path.read_text()) if run_path.exists() else None
    if prior and (prior["arm"], prior["label"]) != (args.arm, args.label):
        print(f"ERROR: {args.out} holds arm {prior['arm']} / {prior['label']}", file=sys.stderr)
        return 1
    seed = args.seed
    if backend == "cpp" and prior:
        if seed is not None and seed != prior["base_seed"]:
            print(f"ERROR: --seed {seed} differs from the recorded {prior['base_seed']}", file=sys.stderr)
            return 1
        seed = prior["base_seed"]

    env = print_environment({"arm": f"{args.arm} ({backend}{', ' + mode if mode else ''})", "label": args.label,
                             "selection": str(args.selection), "out": str(args.out)})
    from src import tts, tts_cpp

    df = load_ds6(args.ds6)
    sel = pd.read_csv(args.selection)
    records, redo = split_rows_missing_audio(read_jsonl(rows_path), args.out / "wavs")
    if redo:
        print(f"WARNING: redoing {len(redo)} recorded row(s) whose WAV is missing or cut short: {redo}")
        write_jsonl(rows_path, records)
    done = {r["clip_id"] for r in records}
    todo = sel[~sel["clip_id"].isin(done)]
    print(f"{len(sel)} selected, {len(done)} already done, {len(todo)} to synthesize")
    init_set = tts.load_init_set(args.init_set)

    if backend == "hf":
        bundle, synth = tts.load(), tts.synthesize_turn
        info = {"model": tts.QWEN3_TTS_MODEL_ID, "seed": "unseeded (as src/tts.py)"}
    else:
        opts = {"mode": mode, "seed": seed, "log_dir": args.out}
        for key in ("server_bin", "talker_gguf", "codec_gguf", "talker_ctx"):
            if getattr(args, key) is not None:
                opts[key] = getattr(args, key)
        if args.server_arg:
            opts["extra_server_args"] = tuple(args.server_arg)
        bundle, synth = tts_cpp.load(**opts), tts_cpp.synthesize_turn
        info = bundle.info
    run = {"arm": args.arm, "label": args.label, "backend": backend, "mode": mode, "env": env, "tts": info,
           "base_seed": getattr(bundle, "base_seed", None), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    run_path.write_text(json.dumps({**(prior or {}), **run}, indent=2, ensure_ascii=False, default=str))

    (args.out / "wavs").mkdir(exist_ok=True)
    try:
        for i, (_, sel_row) in enumerate(todo.iterrows(), 1):
            cid = sel_row["clip_id"]
            r = df.loc[cid]
            result = synth(bundle, cid, r["rag_answer_native"], r["final_lang"], init_set)
            rec = _row_record(args.arm, args.label, sel_row, result, len(str(r["verbalized_text"])))
            if result.ok:
                sf.write(str(args.out / "wavs" / f"{cid}.wav"), result.audio, result.sr)
                for k, seg_audio in enumerate(getattr(result, "segment_audio", None) or []):
                    seg_dir = args.out / "segments"
                    seg_dir.mkdir(exist_ok=True)
                    sf.write(str(seg_dir / f"{cid}_seg{k:02d}.wav"), seg_audio, result.sr)
            with open(rows_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            note = (f"{rec['audio_s']:.1f}s audio, RTF {rec['rtf']:.2f}, ratio {rec['length_ratio']:.2f}"
                    if result.ok else f"TEXT ONLY: {result.failure_reason}")
            print(f"  [{i}/{len(todo)}] {cid} ({r['final_lang']}): {note}", flush=True)
    finally:
        bundle.unload()
    print(f"Done: {rows_path}")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def hyp_normalize(hyp: str, lang: str, init_set: set) -> str:
    """DS6's hyp_norm: scoring_normalize, plus eval_side_normalize for eng."""
    import lingkod_tts_utils as tu

    norm = tu.scoring_normalize(str(hyp), lang)
    return tu.eval_side_normalize(norm, lang, init_set) if lang == "eng" else norm


def build_judge(lang: str, device: str):
    """Port of build_bundle in run_whisper_judge_ds2_ds5.py."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(JUDGE_REPOS[lang], dtype=dtype).to(device).eval()
    embed = model.get_input_embeddings().weight.shape[0]
    if lang == "ceb":
        processor = WhisperProcessor.from_pretrained(JUDGE_BASE_MODEL, task="transcribe")
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [JUDGE_CEB_TOKEN]})
        ceb_id = processor.tokenizer.convert_tokens_to_ids(JUDGE_CEB_TOKEN)
        if ceb_id != JUDGE_CEB_TOKEN_ID or embed != len(processor.tokenizer):
            raise RuntimeError(f"ceb judge: token id {ceb_id}, embed {embed}, tokenizer {len(processor.tokenizer)}")
        forced = [[1, ceb_id], [2, processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")],
                  [3, processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")]]
    else:
        processor = WhisperProcessor.from_pretrained(JUDGE_BASE_MODEL, language=JUDGE_WHISPER_LANG[lang],
                                                     task="transcribe")
        if embed != len(processor.tokenizer):
            raise RuntimeError(f"{lang} judge: embed {embed} != tokenizer {len(processor.tokenizer)}")
        forced = processor.get_decoder_prompt_ids(language=JUDGE_WHISPER_LANG[lang], task="transcribe")
    model.generation_config.forced_decoder_ids = forced
    return processor, model


def judge_transcribe(paths: list[Path], processor, model, device: str) -> list[str]:
    import librosa
    import torch

    audios = [librosa.load(str(p), sr=JUDGE_SR, mono=True)[0] for p in paths]
    feats = processor.feature_extractor(audios, sampling_rate=JUDGE_SR, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats.to(device, dtype=model.dtype), max_length=JUDGE_MAX_LENGTH)
    return [t.strip() for t in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def utmos_score(path: Path, utmos, device: str) -> float:
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample as scipy_resample

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio[:, 0]
    if sr != UTMOS_SR:
        audio = scipy_resample(audio, int(round(len(audio) * UTMOS_SR / sr))).astype(np.float32)
    with torch.no_grad():
        return float(utmos.score(torch.from_numpy(audio).unsqueeze(0).to(device)).item())


def _load_cache(path: Path) -> dict:
    return {(d["label"], d["clip_id"]): d["value"] for d in read_jsonl(path)}


def _append_cache(path: Path, label: str, clip_id: str, value) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"label": label, "clip_id": clip_id, "value": value}, ensure_ascii=False) + "\n")


def load_runs(run_dirs: list[Path]):
    import pandas as pd

    frames = []
    for d in run_dirs:
        rows_path = d / "rows.jsonl"
        if not rows_path.exists():
            raise FileNotFoundError(f"{rows_path} not found; run synth first")
        rows = read_jsonl(rows_path)
        for r in rows:
            r["wav"] = str(d / "wavs" / f"{r['clip_id']}.wav") if r["ok"] else None
        frames.append(pd.DataFrame(rows))
    runs = pd.concat(frames, ignore_index=True)
    dup = runs.duplicated(["label", "clip_id"])
    if dup.any():
        pairs = runs.loc[dup, ["label", "clip_id"]].values[:5]
        raise RuntimeError(f"duplicate (label, clip_id) rows across runs: {pairs}")
    return runs


def summarize(scores, ds6_sel):
    """Per (label, lang) and per label: completion, WER, UTMOS, length, speed."""
    import numpy as np
    import pandas as pd
    import lingkod_tts_utils as tu

    def agg(g):
        ok = g[g["ok"]]
        wer_rows = ok[ok["wer_eligible"] & ok["hyp_norm"].notna()]
        ratio = ok["length_ratio"].dropna()
        ttfa = ok["ttfa_s"].dropna()
        return pd.Series({
            "rows": len(g),
            "text_only": int((~g["ok"]).sum()),
            "wer_rows": len(wer_rows),
            "corpus_wer": tu.tts_corpus_wer(wer_rows["reference_norm"].tolist(), wer_rows["hyp_norm"].tolist())
            if len(wer_rows) else np.nan,
            "utmos_mean": ok["utmos"].mean(),
            "ratio_median": ratio.median(),
            "ratio_max": ratio.max(),
            "ratio_over_2": int((ratio > 2).sum()),
            "rtf_median": ok["rtf"].median(),
            "ttfa_median_s": ttfa.median(),
            "ttfa_max_s": ttfa.max(),
            "rows_with_stalls": int((ok["n_stalls"].fillna(0) > 0).sum()),
            "cap_hit_attempts": int(g["cap_hit_attempts"].fillna(0).sum()),
            "retries": int(g["retries"].fillna(0).sum()),
        })

    ds6 = ds6_sel.assign(label="DS6 (stored)", ok=True, rtf=np.nan, ttfa_s=np.nan, n_stalls=np.nan,
                         cap_hit_attempts=np.nan, retries=np.nan)
    both = pd.concat([scores, ds6], ignore_index=True)
    # Explicit loops rather than groupby.apply, whose signature changed across pandas 2.x.
    by_lang = pd.DataFrame({key: agg(g) for key, g in both.groupby(["label", "lang"], sort=False)}).T
    by_lang.index.names = ["label", "lang"]
    overall = pd.DataFrame({key: agg(g) for key, g in both.groupby("label", sort=False)}).T
    overall.index.name = "label"
    return by_lang, overall


def cmd_score(args) -> int:
    import numpy as np
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    env = print_environment({"runs": " ".join(str(r) for r in args.runs), "out": str(args.out)})
    try:
        import utmos_pytorch  # noqa: F401
    except ImportError:
        print("ERROR: utmos-pytorch is not installed (pip install utmos-pytorch==0.1.0)", file=sys.stderr)
        return 1
    import torch
    from src import tts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_set = tts.load_init_set(args.init_set)
    df = load_ds6(args.ds6)
    runs = load_runs(args.runs)
    missing = sorted(set(runs["clip_id"]) - set(df.index))
    if missing:
        raise RuntimeError(f"clip_id(s) not in DS6: {missing[:5]}")

    hyp_cache_path, utmos_cache_path = args.out / "judge_hyps.jsonl", args.out / "utmos.jsonl"
    hyps = _load_cache(hyp_cache_path)
    ok_rows = runs[runs["ok"]]
    for lang in ("eng", "fil", "ceb"):
        not_done = np.array([(lb, c) not in hyps for lb, c in zip(ok_rows["label"], ok_rows["clip_id"])], dtype=bool)
        todo = ok_rows[(ok_rows["lang"] == lang).to_numpy() & not_done]
        if todo.empty:
            continue
        print(f"judge {lang}: {JUDGE_REPOS[lang]} on {len(todo)} clip(s)")
        processor, model = build_judge(lang, device)
        items = list(zip(todo["label"], todo["clip_id"], todo["wav"]))
        for i in range(0, len(items), JUDGE_BATCH):
            batch = items[i:i + JUDGE_BATCH]
            for (label, cid, _), h in zip(batch, judge_transcribe([Path(w) for _, _, w in batch], processor, model,
                                                                  device)):
                hyps[(label, cid)] = h
                _append_cache(hyp_cache_path, label, cid, h)
        del processor, model
        if device == "cuda":
            torch.cuda.empty_cache()

    utmos_vals = _load_cache(utmos_cache_path)
    todo = [(lb, c, w) for lb, c, w in zip(ok_rows["label"], ok_rows["clip_id"], ok_rows["wav"])
            if (lb, c) not in utmos_vals]
    if todo:
        from utmos_pytorch import UTMOSScoreTorch

        print(f"UTMOS on {len(todo)} clip(s)")
        utmos = UTMOSScoreTorch(device=device)
        for label, cid, wav in todo:
            utmos_vals[(label, cid)] = utmos_score(Path(wav), utmos, device)
            _append_cache(utmos_cache_path, label, cid, utmos_vals[(label, cid)])
        del utmos

    sel_cols = ["clip_id", "final_lang", "verbalized_text", "reference_norm", "hyp_norm", "tts_rt_wer", "utmos22",
                "duration_sec"]
    ds6_part = df.loc[runs["clip_id"].unique(), sel_cols].reset_index(drop=True)
    ds6_part["wer_eligible"] = ds6_part["verbalized_text"].map(ref_words) <= WER_MAX_REF_WORDS
    keyed = list(zip(runs["label"], runs["clip_id"]))
    scores = runs.merge(ds6_part[["clip_id", "reference_norm", "wer_eligible"]], on="clip_id", how="left")
    scores["judge_hyp"] = [hyps.get(k) for k in keyed]
    scores["hyp_norm"] = [hyp_normalize(h, lang, init_set) if h is not None else None
                          for h, lang in zip(scores["judge_hyp"], scores["lang"])]
    scores["utmos"] = [utmos_vals.get(k, np.nan) for k in keyed]
    scores["ttfa_s"] = [tl.get("ttfa_s") if isinstance(tl, dict) else np.nan for tl in scores["timeline"]]
    scores["n_stalls"] = [tl.get("n_stalls") if isinstance(tl, dict) else np.nan for tl in scores["timeline"]]
    for col in ("cap_hit_attempts", "retries"):
        if col not in scores:
            scores[col] = np.nan
    import jiwer

    scores["row_wer"] = [jiwer.wer(r, h) if isinstance(h, str) and str(r).strip() else np.nan
                         for r, h in zip(scores["reference_norm"], scores["hyp_norm"])]

    ds6_sel = ds6_part.rename(columns={"final_lang": "lang", "utmos22": "utmos", "duration_sec": "audio_s"})
    ds6_sel["length_ratio"] = ds6_sel["audio_s"] / (ds6_sel["verbalized_text"].str.len() * 0.84 / 12.5)
    by_lang, overall = summarize(scores, ds6_sel)

    keep = ["label", "arm", "clip_id", "lang", "tags", "ok", "failure_reason", "wer_eligible", "row_wer", "utmos",
            "audio_s", "length_ratio", "wall_sec", "rtf", "ttfa_s", "n_stalls", "cap_hit_attempts", "retries",
            "judge_hyp"]
    scores[[c for c in keep if c in scores]].to_csv(args.out / "scores.csv", index=False)
    by_lang.to_csv(args.out / "summary_by_lang.csv")
    overall.to_csv(args.out / "summary.csv")
    (args.out / "score_env.json").write_text(json.dumps(env, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\nBy label and language (corpus WER on rows of at most 75 reference words):")
        print(by_lang)
        print("\nOverall:")
        print(overall)
    print(f"\nWrote {args.out}/scores.csv, summary.csv, summary_by_lang.csv")
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="choose the DS6 rows every arm synthesizes")
    s.add_argument("--ds6", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True, help="selection CSV to write")
    s.add_argument("--per-stratum", type=int, default=3, help="rows per length quartile per language (default 3)")
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("synth", help="synthesize the selection with one arm")
    s.add_argument("--ds6", type=Path, required=True)
    s.add_argument("--selection", type=Path, required=True)
    s.add_argument("--arm", choices=sorted(ARMS), required=True, help="A hf, B cpp parity, C cpp segmented")
    s.add_argument("--label", required=True, help="run name, e.g. C-seg-Q8_0")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--init-set", type=Path, default=DEFAULT_INIT_SET)
    s.add_argument("--seed", type=int, default=None, help="cpp base seed (recorded; reused on resume)")
    s.add_argument("--server-bin", type=Path, default=None)
    s.add_argument("--talker-gguf", type=Path, default=None)
    s.add_argument("--codec-gguf", type=Path, default=None)
    s.add_argument("--talker-ctx", type=int, default=None)
    s.add_argument("--server-arg", action="append", default=None, help="extra tts-server flag, repeatable")

    s = sub.add_parser("score", help="judge WER and UTMOS for one or more synth runs")
    s.add_argument("--ds6", type=Path, required=True)
    s.add_argument("--runs", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--init-set", type=Path, default=DEFAULT_INIT_SET)

    args = ap.parse_args()
    if args.cmd == "synth" and ARMS[args.arm][0] == "hf":
        cpp_only = [k for k in ("seed", "server_bin", "talker_gguf", "codec_gguf", "talker_ctx", "server_arg")
                    if getattr(args, k) is not None]
        if cpp_only:
            print(f"ERROR: arm A takes no cpp options: {cpp_only}", file=sys.stderr)
            return 1
    return {"select": cmd_select, "synth": cmd_synth, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
