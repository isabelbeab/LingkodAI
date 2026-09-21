"""Stage 7 -- Text-to-speech synthesis, native answer to cloned-voice audio.

Qwen3-TTS-12Hz-1.7B-Base, ICL voice cloning, fp32 (fp16 produces NaN/inf).
Three locked reference speakers, one per language, each padded with 0.5s of
trailing silence before cloning. Generation: x_vector_only_mode=False,
non_streaming_mode=True, max_new_tokens=4096 with one OOM retry at 2048,
language="english" for eng and None for fil and ceb.

Text path: prep_tts_text, then per language:
  ceb  ceb_full_pipeline.full_ceb_pipeline_segments(text, initialism_set,
       demo_fallback=False) -- the regression-verified path, never the demo
       shortcut -- per-segment generation, ceb_segmentation.concat_segments.
  fil  verbalize_fil_prc
  eng  verbalize_eng_prc

Ported from reference/TTS files/LingkodAI_DS6_DS7_Synthesis.ipynb, cells 5-6
(model load + locked reference speakers, synthesize()). CLAUDE.md's stage
table names this reference/hk_ds6_ds7_synthesis.ipynb; that file arrived
under a different path/name, confirmed to be the same notebook.

The frozen initialism set (data/init_set_ds6.json, generated once by
scripts/build_init_set.py) must be loaded, never derived per answer -- see
that script's docstring for why per-answer derivation silently breaks
acronym spelling.

The TTS frontend raises on some inputs by design (three CEB rows raised in
the DS7 run and were excluded). synthesize() preserves that: it raises.
synthesize_turn() is the non-crashing entry point the pipeline should call --
a raise there is caught and recorded, audio is set to None, and the turn's
text answer is unaffected.

Never use scoring-side normalizers (scoring_normalize*) here -- those are for
WER evaluation, not synthesis.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# tts_frontend modules import each other by top-level name (vendored verbatim
# from JOJIE, never edited -- see src/tts_frontend/VENDORED.md), so this
# directory goes on sys.path rather than rewriting their imports.
_TTS_FRONTEND_DIR = Path(__file__).resolve().parent / "tts_frontend"
if str(_TTS_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_TTS_FRONTEND_DIR))

import ceb_full_pipeline  # noqa: E402
import ceb_segmentation  # noqa: E402
from eng_prc_digit_verbalizer import verbalize_eng_prc  # noqa: E402
from fil_prc_digit_verbalizer import verbalize_fil_prc  # noqa: E402
from lingkod_tts_utils import prep_tts_text  # noqa: E402

QWEN3_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_NEW_TOKENS_PRIMARY = 4096
MAX_NEW_TOKENS_FALLBACK = 2048
REF_PAD_SILENCE_SEC = 0.5

# Locked reference clips, one per language. Filenames only: the audio is PLD
# speaker audio under a research-only CC-BY-NC licence and is never committed.
# It is read from local disk under TTS_REF_DIR (DECISIONS.md #8).
REF_CLIP_FILENAMES = {
    "eng": "1904.141120.022212.0152.wav",
    "fil": "0052.110908.020820.0443.wav",  # 0443, not 0433
    "ceb": "0233.111026.023506.0490.wav",
}

# ref_text and language, copied verbatim from REF_CONFIG in
# reference/TTS files/LingkodAI_DS6_DS7_Synthesis.ipynb (Cell 5). The CEB
# ref_text carries a deliberate manual substitution ("siyete"); keep it.
REF_TEXT_CONFIG = {
    "eng": {
        "ref_text": (
            "a number of stories in the collection were excluded due to loss of the "
            "original manuscript sent to various authors at the time were plotted on "
            "the first quarter storm rust no shed on the moment of separation and "
            "the dry river at the foot of the mountain"
        ),
        "language": "english",
    },
    "fil": {
        "ref_text": (
            "ngunit ang alimango'y hindi maaaring umakyat sa punongkahoy ang "
            "pariralang ito ay magiging lalong tama sa pagong sa aming paniwala "
            "ang alimangong may sipit at mga paa ay maaaring umakyat na katulad "
            "ng matsing madaling naaakyat ng alimango ang alin mang bato moog at "
            "iba pa"
        ),
        "language": None,
    },
    "ceb": {
        "ref_text": (
            "kini ang nalatid sa kamanduan nga gipakanaog kagahapon sa hapon ni "
            "regional trial court executive judge meinrado paredes human "
            "gisumiter sa criminal investigation and detection group siyete ang "
            "nakuha nilang mga ebidensya sa giingong suspek sa kaso ni pique "
            "didto sa barangay inayagan dakbayan sa naga"
        ),
        "language": None,
    },
}


def find_ref_clip(ref_dir: Path, filename: str) -> Path:
    """Locate one locked reference clip under ref_dir, at any depth.

    Raises with the resolved directory if it is missing or ambiguous; no
    silent fallback to another clip.
    """
    ref_dir = Path(ref_dir).expanduser()
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"TTS_REF_DIR does not exist or is not a directory: {ref_dir.resolve()}")
    hits = sorted(p for p in ref_dir.rglob(filename) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"reference clip {filename} not found under {ref_dir.resolve()}")
    if len(hits) > 1:
        raise RuntimeError(f"reference clip {filename} is ambiguous under {ref_dir.resolve()}: {[str(h) for h in hits]}")
    return hits[0]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_init_set(path: str | Path) -> set[str]:
    """Load the frozen initialism set written by scripts/build_init_set.py.

    Never derive this at runtime from the answer being spoken -- see that
    script's docstring for why per-answer derivation silently leaves plural
    acronyms unspelled.
    """
    with open(path, encoding="utf-8") as fh:
        return set(json.load(fh)["tokens"])


def _pad_ref_audio_with_silence(
    audio_path: Path, silence_sec: float = REF_PAD_SILENCE_SEC, out_dir: Path | None = None
) -> Path:
    """Ported verbatim from the notebook's pad_ref_audio_with_silence."""
    audio, sr = sf.read(str(audio_path))
    n_silence = int(round(silence_sec * sr))
    silence = (
        np.zeros(n_silence, dtype=audio.dtype)
        if audio.ndim == 1
        else np.zeros((n_silence, audio.shape[1]), dtype=audio.dtype)
    )
    padded = np.concatenate([audio, silence], axis=0)
    out_dir = out_dir or Path(tempfile.gettempdir()) / "lingkodai_ref_padded"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{audio_path.stem}_pad{silence_sec}s.wav"
    sf.write(str(out_path), padded, sr)
    return out_path


@dataclass
class TTSBundle:
    """A loaded Qwen3-TTS model plus its three padded reference speakers."""

    model: "Qwen3TTSModel"
    device: str
    ref_sr: int
    ref_config: dict[str, dict]

    def unload(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load(
    ref_dir: str | Path | None = None,
    device: str | None = None,
) -> TTSBundle:
    """Find the locked reference clips under TTS_REF_DIR, load the model, and
    confirm the sample rate with a short warm-up clone -- same sequence as
    the notebook's model load cell.

    qwen_tts is imported here, not at module level: its torchaudio
    dependency pulls in a CUDA extension that only loads on a real GPU
    machine, so importing it at module level would break `import src.tts`
    (and the CPU-only smoke tests) on HK's laptop.
    """
    from qwen_tts import Qwen3TTSModel

    ref_dir = ref_dir or os.environ.get("TTS_REF_DIR")
    if not ref_dir:
        raise ValueError(
            "TTS_REF_DIR is not set. It must point at a directory holding the three "
            "locked reference clips (searched recursively); see .env.example."
        )
    device = device or get_device()

    ref_config: dict[str, dict] = {}
    for lang, filename in REF_CLIP_FILENAMES.items():
        clip_path = find_ref_clip(Path(ref_dir), filename)
        entry = REF_TEXT_CONFIG[lang]
        ref_config[lang] = {
            "ref_audio": _pad_ref_audio_with_silence(clip_path),
            "ref_text": entry["ref_text"],
            "language": entry["language"],
        }

    model = Qwen3TTSModel.from_pretrained(QWEN3_TTS_MODEL_ID, dtype=torch.float32, device_map=device)

    # Warm-up clone confirms the sample rate the model actually returns,
    # rather than assuming one -- same as the notebook's REF_SR line.
    _, ref_sr = model.generate_voice_clone(
        text="test",
        language=ref_config["eng"]["language"],
        ref_audio=str(ref_config["eng"]["ref_audio"]),
        ref_text=ref_config["eng"]["ref_text"],
        x_vector_only_mode=False,
        non_streaming_mode=True,
        max_new_tokens=64,
    )

    return TTSBundle(model=model, device=device, ref_sr=ref_sr, ref_config=ref_config)


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _gen_once(bundle: TTSBundle, text: str, lang: str, max_new_tokens: int):
    cfg = bundle.ref_config[lang]
    t0 = time.time()
    wavs, sr = bundle.model.generate_voice_clone(
        text=text,
        language=cfg["language"],
        ref_audio=str(cfg["ref_audio"]),
        ref_text=cfg["ref_text"],
        x_vector_only_mode=False,
        non_streaming_mode=True,
        max_new_tokens=max_new_tokens,
    )
    return wavs[0], sr, time.time() - t0


def _gen_with_oom_retry(bundle: TTSBundle, text: str, lang: str):
    try:
        return _gen_once(bundle, text, lang, MAX_NEW_TOKENS_PRIMARY)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        gc.collect()
        if bundle.device == "cuda":
            torch.cuda.empty_cache()
        return _gen_once(bundle, text, lang, MAX_NEW_TOKENS_FALLBACK)


def _synth_fil_eng(bundle: TTSBundle, text: str, lang: str):
    wav, sr, wall = _gen_with_oom_retry(bundle, text, lang)
    dur = len(wav) / sr
    return wav, sr, wall, (wall / dur if dur else None)


def _synth_ceb(bundle: TTSBundle, text: str, initialism_set: set):
    segments = ceb_full_pipeline.full_ceb_pipeline_segments(text, initialism_set, demo_fallback=False)

    def _gen():
        for seg_text, pause in segments:
            wav, _sr, wall = _gen_with_oom_retry(bundle, seg_text, "ceb")
            yield wav, pause, wall

    audio, timings = ceb_segmentation.concat_segments(_gen(), bundle.ref_sr)
    wall = sum(timings)
    dur = len(audio) / bundle.ref_sr
    return audio, bundle.ref_sr, wall, (wall / dur if dur else None)


def synthesize(bundle: TTSBundle, text: str, lang: str, initialism_set: set):
    """Native-language text -> (audio, sr, wall_sec, rtf). Raises on frontend
    failures by design; callers wanting a non-crashing turn should call
    synthesize_turn instead.
    """
    lang = lang.lower().strip()
    if lang not in bundle.ref_config:
        raise ValueError(f"Unknown lang {lang!r}, expected one of {list(bundle.ref_config)}")
    prepped = prep_tts_text(str(text))
    if lang == "ceb":
        return _synth_ceb(bundle, prepped, initialism_set)
    if lang == "fil":
        vtext = verbalize_fil_prc(prepped, initialism_set=initialism_set)
        return _synth_fil_eng(bundle, vtext, "fil")
    vtext = verbalize_eng_prc(prepped, initialism_set=initialism_set)
    return _synth_fil_eng(bundle, vtext, "eng")


@dataclass
class TTSResult:
    """Synthesis outcome for one turn. audio is None when ok is False."""

    id: object
    lang: str
    audio: np.ndarray | None
    sr: int | None
    wall_sec: float | None
    rtf: float | None
    ok: bool
    failure_reason: str | None = None


def synthesize_turn(
    bundle: TTSBundle, id: object, text: str, lang: str, initialism_set: set
) -> TTSResult:
    """Non-crashing entry point for the pipeline.

    The TTS frontend raises on some inputs by design (three CEB rows raised
    in the DS7 run and were excluded). A raise here must not crash the turn:
    the caller keeps the text answer regardless, audio is just None with the
    reason recorded.
    """
    try:
        audio, sr, wall, rtf = synthesize(bundle, text, lang, initialism_set)
        return TTSResult(id, lang, audio, sr, wall, rtf, True)
    except Exception as exc:
        return TTSResult(id, lang, None, None, None, None, False, str(exc)[:300])
