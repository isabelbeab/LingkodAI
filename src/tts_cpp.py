"""Stage 7 on qwentts.cpp: src/tts.py's text path, synthesized by a patched
qwentts.cpp `tts-server` child process instead of qwen-tts in this process.

Branch `tts-cpp` only. Same interface as src/tts.py: load(), synthesize(),
synthesize_turn() returning a TTSResult (here CppTTSResult, which adds
per-segment records and a simulated playback timeline), bundle.unload().

Two modes, chosen at load():

  parity     arm B of the A/B. src/tts.py's text path unchanged: prep_tts_text;
             ceb through full_ceb_pipeline_segments (demo_fallback=False) and
             concat_segments; fil and eng as one whole-answer generation
             through verbalize_fil_prc / verbalize_eng_prc. max_new_tokens
             4096, retried once at 2048 when the server reports out of memory.
             A cap hit is recorded, not acted on.
  segmented  arm C. Every language is split before verbalizing with the
             vendored ceb_segmentation.split_for_synthesis (140 chars, pauses
             0.20 / 0.45 / 0.60 s), the order ceb's own pipeline uses. "No.",
             "Nos.", "(s." "A.Y." and "R.A." stay attached to the number that
             follows (see _ABBREV_GUARD). For fil and eng the verbalizers'
             paren_conditional_strip runs once on the whole answer before the
             split, so a parenthetical acronym is dropped exactly when parity
             drops it (it judges against the whole text, which a segment
             alone would not have). Each segment is capped at
             2 x expected frames + 25, expected frames being 0.84 per
             verbalized character (DS6: correlation 0.998 ceb, 0.995 fil, 0.94
             eng). A segment that runs to its cap, or returns no audio, is
             retried once with a new seed; a second failure makes the whole
             turn text-only, the existing designed fallback.

The server (scripts/build_qwentts.sh builds it) runs with --icl-non-streaming
(the non_streaming_mode=True prompt layout src/tts.py uses) and --talker-ctx.
load() starts it on a free localhost port, waits for /health, checks its log
for the two lines only the patched build prints, and registers the three
padded reference voices once, resampled to 24 kHz with librosa exactly as
qwen-tts does. Language: "english" for eng, omitted (auto) for fil and ceb.

Failures inside a turn give a text-only turn, never a crashed conversation:
a frontend raise (as in src/tts.py), a non-OK server answer, a server crash
or a timeout. A crashed or killed server is restarted, voices included,
before the next turn. unload() kills the process, so staged mode gets the
VRAM back.

Frames are counted from the returned WAV: the buffered path decodes exactly
1920 samples (one 12.5 Hz codec frame) per generated frame, so a segment hit
its cap when frames == max_new_tokens.

Seeds are explicit and recorded. Each request's seed derives from the
bundle's base seed, the turn id, the segment index and the attempt, so a run
is reproducible from the base seed alone.

Linux only (process groups via os.killpg). Never use scoring-side normalizers
here; see src/tts.py.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# src.tts puts src/tts_frontend/ on sys.path, so the vendored modules import
# by top-level name below, the same way src/tts.py imports them.
from . import tts
from .tts import (
    MAX_NEW_TOKENS_FALLBACK,
    MAX_NEW_TOKENS_PRIMARY,
    REF_CLIP_FILENAMES,
    REF_TEXT_CONFIG,
    TTSResult,
)

import ceb_full_pipeline  # noqa: E402
import ceb_prc_digit_verbalizer  # noqa: E402
import ceb_respell_lab  # noqa: E402
import ceb_segmentation  # noqa: E402
import fil_prc_digit_verbalizer  # noqa: E402
import shared_domain_frontend  # noqa: E402
from eng_prc_digit_verbalizer import verbalize_eng_prc  # noqa: E402
from fil_prc_digit_verbalizer import verbalize_fil_prc  # noqa: E402
from lingkod_tts_utils import prep_tts_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

MODES = ("parity", "segmented")
SAMPLE_RATE = 24000
HOP_SAMPLES = 1920  # one 12.5 Hz codec frame at 24 kHz

# Length-aware cap (segmented mode), from DS6: about 14.9 verbalized chars per
# second of audio, so 12.5 / 14.9 = 0.84 frames per char; actual/expected p99
# is 1.2, and 2x catches the one DS6 runaway (ceb_01_mt_185, 3.08x) with no
# false positives on the other 2,234 rows.
FRAMES_PER_CHAR = 0.84
CAP_MULTIPLIER = 2.0
CAP_SLACK_FRAMES = 25

# Talker KV positions: 4096 generated frames plus room for the prompt prefix.
# The longest DS6 answer (fil_01_mt_328, 3,362 verbalized chars) goes out as
# one prompt in parity mode. 7168 positions is 1.5 GiB of f32 KV. A prompt
# that still does not fit is refused by the server with both numbers.
DEFAULT_TALKER_CTX = MAX_NEW_TOKENS_PRIMARY + 3072

DEFAULT_SERVER_BIN = REPO_ROOT / "build" / "qwentts.cpp" / "build" / "tts-server"
DEFAULT_STARTUP_TIMEOUT_S = 600.0
DEFAULT_REQUEST_TIMEOUT_S = 1800.0
WARMUP_MAX_NEW_TOKENS = 64  # same as src/tts.py's warm-up clone

VOICE_NAMES = {lang: f"lingkod-{lang}" for lang in REF_CLIP_FILENAMES}

# Abbreviations whose period must not end a segment when a number follows
# (directly, or as "R.A. No. 9298"). DS6 (rag_answer_native, split with
# split_for_synthesis): "No." stranded the number 35 ceb / 34 eng / 21 fil
# times ("Resolution No." then "02-2025." alone), "(s." 4 times ("(s." then
# "2008)"), "A.Y." 3 times.
_ABBREV_GUARD = re.compile(r"(?<![\w.])(No|Nos|s|A\.Y|R\.A)\.\s+(?=\d|Nos?\.\s*\d)")
# Stands in for the space after a guarded period while the splitter runs: not
# whitespace (so the sentence regex cannot split there), not touched by
# clean_rag_answer, restored to a space before verbalizing.
_GUARD_CHAR = ""

# Ordinal words handed to split_for_synthesis for "N." list markers, matching
# what each language's verbalizer speaks for the same marker in parity mode:
# ceb's CEB_ORDINALS (what full_ceb_pipeline_segments passes), fil's own
# list-marker table, and for eng the eng verbalizer's own reading of the bare
# number (verbalize_eng_prc has no list-marker pass, so "1." reads as "one.").
_SPLIT_ORDINALS = {
    "ceb": ceb_prc_digit_verbalizer.CEB_ORDINALS,
    "fil": fil_prc_digit_verbalizer._FIL_ORDINALS,
    "eng": {n: verbalize_eng_prc(str(n), initialism_set=set()) for n in range(1, 100)},
}

_PATCH_LOG_MARKERS = (
    "ICL prompt layout: non_streaming_mode=True",
    "Talker KV length: {talker_ctx} positions",
)


# ---------------------------------------------------------------------------
# Text path
# ---------------------------------------------------------------------------


def _split_guarded(text: str, lang: str) -> list[tuple[str, float]]:
    """split_for_synthesis with the abbreviation guard, vendored file untouched."""
    if _GUARD_CHAR in text:
        raise ValueError(f"answer text contains the guard placeholder U+{ord(_GUARD_CHAR):04X}")
    guarded = _ABBREV_GUARD.sub(lambda m: m.group(1) + "." + _GUARD_CHAR, text)
    pieces = ceb_segmentation.split_for_synthesis(guarded, ordinal_words=_SPLIT_ORDINALS[lang])
    return [(seg.replace(_GUARD_CHAR, " "), pause) for seg, pause in pieces]


def _verbalize_segment(seg_text: str, lang: str, initialism_set: set) -> str:
    if lang == "ceb":
        # Same three steps, same order, as the loop in
        # ceb_full_pipeline.full_ceb_pipeline_segments (load-bearing order,
        # see that module's docstring). Repeated here only because the guard
        # has to act on the text before the split that function does itself.
        t = ceb_prc_digit_verbalizer.verbalize_ceb_prc(seg_text)
        t = ceb_respell_lab.apply_loanword_fixes(t, demo_fallback=False)
        return shared_domain_frontend.apply_shared_frontend(t, "ceb", initialism_set)["output"]
    if lang == "fil":
        return verbalize_fil_prc(seg_text, initialism_set=initialism_set)
    return verbalize_eng_prc(seg_text, initialism_set=initialism_set)


def text_segments(text: str, lang: str, initialism_set: set, mode: str) -> list[tuple[str, float]]:
    """Native answer -> [(verbalized_text, pause_after_seconds), ...], the
    exact strings sent to the server. Raises on frontend failures by design."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")
    prepped = prep_tts_text(str(text))
    if mode == "parity":
        if lang == "ceb":
            return ceb_full_pipeline.full_ceb_pipeline_segments(prepped, initialism_set, demo_fallback=False)
        # One whole-answer generation with no trailing pause: concat_segments
        # then returns the waveform unchanged, as src/tts.py's fil/eng path does.
        if lang == "fil":
            return [(verbalize_fil_prc(prepped, initialism_set=initialism_set), 0.0)]
        return [(verbalize_eng_prc(prepped, initialism_set=initialism_set), 0.0)]
    if lang in ("fil", "eng"):
        # verbalize_fil_prc / verbalize_eng_prc strip a parenthetical only if
        # its words appear bare elsewhere in their input. Run per segment, that
        # test would only see the segment, and "(PRC)" would be spoken in a
        # segment that lacks a bare "PRC". Stripping once here, on the whole
        # answer, gives the parity decision; the per-segment pass then finds
        # nothing more to strip that parity kept.
        prepped, _ = shared_domain_frontend.paren_conditional_strip(prepped)
    return [(_verbalize_segment(seg, lang, initialism_set), pause) for seg, pause in _split_guarded(prepped, lang)]


def expected_frames(verbalized_text: str) -> float:
    return FRAMES_PER_CHAR * len(verbalized_text)


def segment_cap(verbalized_text: str) -> int:
    """Per-segment max_new_tokens in segmented mode, never above 4096."""
    cap = math.ceil(CAP_MULTIPLIER * expected_frames(verbalized_text)) + CAP_SLACK_FRAMES
    return min(cap, MAX_NEW_TOKENS_PRIMARY)


def request_seed(base_seed: int, turn_key: object, seg_idx: int, attempt: int) -> int:
    """Deterministic 31-bit seed per (turn, segment, attempt), independent of
    call order, so one recorded base seed reproduces a whole run."""
    ss = np.random.SeedSequence([base_seed, zlib.crc32(str(turn_key).encode("utf-8")), seg_idx, attempt])
    return int(ss.generate_state(1, dtype=np.uint32)[0] >> 1)


# ---------------------------------------------------------------------------
# Playback simulation
# ---------------------------------------------------------------------------


def playback_timeline(frontend_s: float, segments: list[dict]) -> dict:
    """Simulate sentence-level play-ahead for one turn.

    Segments are generated one after another; segment i is ready once its
    generation (retries included) ends cleanly, and plays only then, so a
    runaway is retried before anyone hears it. Playback of segment i starts
    at max(ready_i, end of segment i-1) and lasts its audio plus its pause.
    A gap between the two is a stall. no_stall_start_s is the smallest start
    delay that avoids every stall: what a client could wait up front, since
    audio length is predictable from text length.
    """
    ready, t = [], frontend_s
    for seg in segments:
        t += seg["wall_s"]
        ready.append(t)
    durations = [seg["audio_s"] + seg["pause_after_s"] for seg in segments]

    stalls, end = [], None
    for i, (r, d) in enumerate(zip(ready, durations)):
        start = r if end is None else max(r, end)
        if end is not None and r > end:
            stalls.append({"segment": i, "stall_s": r - end})
        end = start + d

    played_before = np.concatenate([[0.0], np.cumsum(durations)[:-1]]) if segments else np.zeros(0)
    no_stall_start = float(max(r - p for r, p in zip(ready, played_before))) if segments else None
    return {
        "frontend_s": frontend_s,
        "ttfa_s": ready[0] if ready else None,
        "ready_s": ready,
        "n_stalls": len(stalls),
        "total_stall_s": float(sum(s["stall_s"] for s in stalls)),
        "stalls": stalls,
        "no_stall_start_s": no_stall_start,
        "playback_end_s": end,
        "audio_total_s": float(sum(durations)),
    }


# ---------------------------------------------------------------------------
# Server process and HTTP
# ---------------------------------------------------------------------------


class ServerError(RuntimeError):
    """tts-server answered with a non-OK HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(f"tts-server returned HTTP {status}: {message}")
        self.status = status
        self.message = message


class ServerCrashed(RuntimeError):
    """The tts-server process died, or was killed after a timeout."""


@dataclass
class SpeechResponse:
    audio: np.ndarray
    frames: int
    wall_s: float


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _error_message(body: bytes) -> str:
    try:
        return json.loads(body.decode("utf-8"))["error"]["message"]
    except Exception:
        return body.decode("utf-8", errors="replace")[:300]


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TTSServer:
    """One tts-server child process plus the HTTP calls this module makes."""

    def __init__(
        self,
        binary: Path,
        talker_gguf: Path,
        codec_gguf: Path,
        *,
        talker_ctx: int,
        log_path: Path,
        extra_args: tuple[str, ...] = (),
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ):
        self.binary = Path(binary)
        self.talker_gguf = Path(talker_gguf)
        self.codec_gguf = Path(codec_gguf)
        self.talker_ctx = int(talker_ctx)
        self.log_path = Path(log_path)
        self.extra_args = tuple(extra_args)
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.cmd: list[str] | None = None
        self._log_fh = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n_bytes: int = 4000) -> str:
        if not self.log_path.exists():
            return ""
        with open(self.log_path, "rb") as fh:
            fh.seek(max(0, self.log_path.stat().st_size - n_bytes))
            return fh.read().decode("utf-8", errors="replace")

    def start(self) -> float:
        """Spawn the server and block until /health answers. Returns seconds."""
        for label, p in (("tts-server binary", self.binary), ("talker GGUF", self.talker_gguf),
                         ("codec GGUF", self.codec_gguf)):
            if not p.is_file():
                raise FileNotFoundError(f"{label} not found: {p}")
        if self.is_alive():
            raise RuntimeError("tts-server is already running")

        self.port = _free_port()
        self.cmd = [
            str(self.binary),
            "--model", str(self.talker_gguf),
            "--codec", str(self.codec_gguf),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--lang", "auto",
            "--icl-non-streaming",
            "--talker-ctx", str(self.talker_ctx),
            *self.extra_args,
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        self._log_fh = open(self.log_path, "ab")
        self._log_fh.write(f"\n=== tts_cpp start {time.strftime('%Y-%m-%dT%H:%M:%S')}: {' '.join(self.cmd)}\n".encode())
        self._log_fh.flush()

        t0 = time.perf_counter()
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.DEVNULL, stdout=self._log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.stop()
            raise
        deadline = t0 + self.startup_timeout_s
        while True:
            rc = self.proc.poll()
            if rc is not None:
                self.stop()
                raise ServerCrashed(f"tts-server exited with code {rc} during startup. Log tail:\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, http.client.HTTPException, TimeoutError):
                pass
            if time.perf_counter() > deadline:
                self.stop()
                raise ServerCrashed(
                    f"tts-server not healthy after {self.startup_timeout_s:.0f}s. Log tail:\n{self.log_tail()}"
                )
            time.sleep(0.5)
        startup_s = time.perf_counter() - t0

        with open(self.log_path, "rb") as fh:
            fh.seek(log_offset)
            this_start = fh.read().decode("utf-8", errors="replace")
        for marker in _PATCH_LOG_MARKERS:
            marker = marker.format(talker_ctx=self.talker_ctx)
            if marker not in this_start:
                self.stop()
                raise RuntimeError(
                    f"tts-server log lacks {marker!r}: this binary is not the patched build "
                    f"(scripts/build_qwentts.sh) or ignored a flag. Binary: {self.binary}"
                )
        return startup_s

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=15)
            except ProcessLookupError:
                pass
        self.proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def _post(self, path: str, payload: dict) -> bytes:
        if not self.is_alive():
            raise ServerCrashed("tts-server is not running")
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise ServerError(exc.code, _error_message(exc.read())) from None
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError) or isinstance(exc, TimeoutError):
                self.stop()
                raise ServerCrashed(
                    f"tts-server did not answer {path} within {self.request_timeout_s:.0f}s; killed it"
                ) from exc
            rc = self.proc.poll() if self.proc is not None else None
            self.stop()
            raise ServerCrashed(
                f"tts-server connection failed on {path} ({reason}); exit code {rc}. Log tail:\n{self.log_tail()}"
            ) from exc

    def register_voice(self, name: str, ref_text: str, audio_24k: np.ndarray) -> float:
        t0 = time.perf_counter()
        payload = {"name": name, "ref_text": ref_text, "wav_b64": _wav_b64(audio_24k, SAMPLE_RATE)}
        self._post("/v1/audio/voices", payload)
        return time.perf_counter() - t0

    def speech(self, text: str, voice: str, language: str | None, seed: int, max_new_tokens: int) -> SpeechResponse:
        payload = {
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "seed": int(seed),
            "max_new_tokens": int(max_new_tokens),
        }
        if language is not None:
            payload["language"] = language
        t0 = time.perf_counter()
        body = self._post("/v1/audio/speech", payload)
        wall = time.perf_counter() - t0
        audio, sr = sf.read(io.BytesIO(body), dtype="float32")
        if sr != SAMPLE_RATE or audio.ndim != 1:
            raise RuntimeError(f"tts-server returned {audio.ndim}-D audio at {sr} Hz, expected mono {SAMPLE_RATE} Hz")
        if len(audio) % HOP_SAMPLES:
            raise RuntimeError(
                f"tts-server returned {len(audio)} samples, not a whole number of {HOP_SAMPLES}-sample frames; "
                "frame counting (and cap detection) would be wrong"
            )
        return SpeechResponse(audio=audio, frames=len(audio) // HOP_SAMPLES, wall_s=wall)


# ---------------------------------------------------------------------------
# Load / unload
# ---------------------------------------------------------------------------


def _load_ref_24k(padded_path: Path) -> np.ndarray:
    """Padded reference clip -> mono float32 at 24 kHz, the way qwen-tts
    prepares it for both the codec and the speaker encoder: librosa.load at
    the native rate, then librosa.resample."""
    import librosa

    audio, sr = librosa.load(str(padded_path), sr=None, mono=True)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(y=audio.astype(np.float32), orig_sr=int(sr), target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


@dataclass
class CppTTSBundle:
    """A running tts-server plus the three registered reference voices."""

    server: TTSServer
    mode: str
    base_seed: int
    ref_sr: int
    ref_config: dict[str, dict]
    info: dict = field(default_factory=dict)
    restarts: int = 0

    def _start_and_register(self) -> dict:
        timing = {"startup_s": self.server.start()}
        timing["voice_register_s"] = {
            lang: self.server.register_voice(VOICE_NAMES[lang], cfg["ref_text"], cfg["ref_audio_24k"])
            for lang, cfg in self.ref_config.items()
        }
        return timing

    def ensure_server(self) -> None:
        """Restart a crashed or killed server, voices included."""
        if self.server.is_alive():
            return
        self.restarts += 1
        print(f"  [tts_cpp] tts-server not running; restart {self.restarts}, log {self.server.log_path}")
        timing = self._start_and_register()
        self.info.setdefault("restarts", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **timing})

    def unload(self) -> None:
        self.server.stop()


def _resolve(arg, env_name: str, default=None) -> Path:
    value = arg or os.environ.get(env_name) or default
    if not value:
        raise ValueError(f"{env_name} is not set and no path was passed; see src/tts_cpp.py")
    return Path(value).expanduser()


def load(
    ref_dir: str | Path | None = None,
    *,
    mode: str = "parity",
    server_bin: str | Path | None = None,
    talker_gguf: str | Path | None = None,
    codec_gguf: str | Path | None = None,
    talker_ctx: int = DEFAULT_TALKER_CTX,
    seed: int | None = None,
    extra_server_args: tuple[str, ...] = (),
    log_dir: str | Path | None = None,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> CppTTSBundle:
    """Start tts-server, register the three padded reference voices, warm up.

    Paths come from the arguments, else QWENTTS_SERVER_BIN (default
    build/qwentts.cpp/build/tts-server), QWENTTS_TALKER_GGUF,
    QWENTTS_CODEC_GGUF and TTS_REF_DIR. seed=None draws a base seed, which is
    printed and recorded; pass it back to reproduce a run.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")
    binary = _resolve(server_bin, "QWENTTS_SERVER_BIN", DEFAULT_SERVER_BIN)
    talker = _resolve(talker_gguf, "QWENTTS_TALKER_GGUF")
    codec = _resolve(codec_gguf, "QWENTTS_CODEC_GGUF")
    ref_dir = ref_dir or os.environ.get("TTS_REF_DIR")
    if not ref_dir:
        raise ValueError(
            "TTS_REF_DIR is not set. It must point at a directory holding the three "
            "locked reference clips (searched recursively); see .env.example."
        )
    log_dir = Path(log_dir) if log_dir else Path(os.environ.get("TMPDIR", "/tmp")) / "lingkodai_tts_cpp"
    base_seed = int(seed) if seed is not None else secrets.randbelow(2**31)

    ref_config: dict[str, dict] = {}
    for lang, filename in REF_CLIP_FILENAMES.items():
        padded = tts._pad_ref_audio_with_silence(tts.find_ref_clip(Path(ref_dir), filename))
        entry = REF_TEXT_CONFIG[lang]
        ref_config[lang] = {
            "ref_audio": padded,
            "ref_audio_24k": _load_ref_24k(padded),
            "ref_text": entry["ref_text"],
            "language": entry["language"],
        }

    build_info_path = binary.parent / "BUILD_INFO.txt"
    server = TTSServer(
        binary, talker, codec,
        talker_ctx=talker_ctx,
        log_path=log_dir / "tts-server.log",
        extra_args=tuple(extra_server_args),
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
    )
    bundle = CppTTSBundle(
        server=server,
        mode=mode,
        base_seed=base_seed,
        ref_sr=SAMPLE_RATE,
        ref_config=ref_config,
        info={
            "mode": mode,
            "base_seed": base_seed,
            "server_bin": str(binary),
            "talker_gguf": str(talker),
            "codec_gguf": str(codec),
            "talker_ctx": talker_ctx,
            "extra_server_args": list(extra_server_args),
            "server_log": str(server.log_path),
            "build_info": build_info_path.read_text() if build_info_path.is_file() else None,
        },
    )
    print("[tts_cpp] Stage 7 backend: qwentts.cpp tts-server")
    for key in ("mode", "base_seed", "server_bin", "talker_gguf", "codec_gguf", "talker_ctx", "server_log"):
        print(f"  {key:<12} {bundle.info[key]}")
    if bundle.info["build_info"] is None:
        print(f"  build_info   none at {build_info_path} (patched flags are still checked in the server log)")
    try:
        bundle.info.update(bundle._start_and_register())
        bundle.info["server_cmd"] = server.cmd
        warm = server.speech(
            "test", VOICE_NAMES["eng"], ref_config["eng"]["language"], base_seed, WARMUP_MAX_NEW_TOKENS
        )
        bundle.info["warmup_s"] = warm.wall_s
    except BaseException:
        server.stop()
        raise
    print(f"  startup      {bundle.info['startup_s']:.1f}s, voices "
          + ", ".join(f"{k} {v:.2f}s" for k, v in bundle.info["voice_register_s"].items())
          + f", warm-up {bundle.info['warmup_s']:.2f}s")
    return bundle


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@dataclass
class CppTTSResult(TTSResult):
    """TTSResult plus what the qwentts.cpp backend knows about the turn.

    segments holds one JSON-ready record per segment (the text sent,
    expected frames, cap, every attempt's seed, frames and wall time), also
    for text-only turns as far as generation got. segment_audio holds the
    per-segment waveforms (for listening; not JSON). timeline is the
    simulated play-ahead (playback_timeline), None for a text-only turn.
    """

    mode: str | None = None
    segments: list[dict] = field(default_factory=list)
    segment_audio: list[np.ndarray] = field(default_factory=list)
    timeline: dict | None = None
    server_restarts: int = 0


class _TextOnly(Exception):
    """A designed text-only outcome for the whole turn."""


def _is_oom_message(message: str) -> bool:
    m = message.lower()
    return "out of memory" in m or "malloc failed" in m


def _attempt(bundle: CppTTSBundle, rec: dict, text: str, lang: str, seed: int, max_new_tokens: int) -> SpeechResponse:
    cfg = bundle.ref_config[lang]
    att = {"seed": seed, "max_new_tokens": max_new_tokens, "frames": None, "cap_hit": None, "wall_s": None,
           "error": None}
    rec["attempts"].append(att)
    try:
        resp = bundle.server.speech(text, VOICE_NAMES[lang], cfg["language"], seed, max_new_tokens)
    except (ServerError, ServerCrashed) as exc:
        att["error"] = str(exc)[:300]
        raise
    att.update(frames=resp.frames, cap_hit=resp.frames >= max_new_tokens, wall_s=resp.wall_s)
    return resp


def _generate_segment(bundle: CppTTSBundle, turn_key: object, rec: dict, lang: str) -> np.ndarray:
    i, text = rec["index"], rec["text"]
    try:
        if bundle.mode == "parity":
            seed = request_seed(bundle.base_seed, turn_key, i, 0)
            try:
                resp = _attempt(bundle, rec, text, lang, seed, MAX_NEW_TOKENS_PRIMARY)
            except ServerError as exc:
                if not _is_oom_message(exc.message):
                    raise
                seed = request_seed(bundle.base_seed, turn_key, i, 1)
                resp = _attempt(bundle, rec, text, lang, seed, MAX_NEW_TOKENS_FALLBACK)
            return resp.audio

        for attempt in (0, 1):
            seed = request_seed(bundle.base_seed, turn_key, i, attempt)
            resp = _attempt(bundle, rec, text, lang, seed, rec["cap"])
            if 0 < resp.frames < rec["cap"]:
                return resp.audio
        seeds = [a["seed"] for a in rec["attempts"]]
        frames = [a["frames"] for a in rec["attempts"]]
        raise _TextOnly(
            f"segment {i}: no clean generation in two attempts (frames {frames}, cap {rec['cap']}, "
            f"seeds {seeds}); whole turn text-only by design"
        )
    except ServerError as exc:
        raise _TextOnly(f"segment {i}: {exc}") from exc
    except ServerCrashed as exc:
        raise _TextOnly(f"segment {i}: {exc}") from exc


def synthesize_turn(
    bundle: CppTTSBundle, id: object, text: str, lang: str, initialism_set: set
) -> CppTTSResult:
    """Non-crashing entry point for the pipeline, as in src/tts.py: any
    failure gives audio None plus the reason, and the text answer stands."""
    lang = str(lang).lower().strip()
    base = dict(id=id, lang=lang, mode=bundle.mode)
    if lang not in bundle.ref_config:
        return CppTTSResult(**base, audio=None, sr=None, wall_sec=None, rtf=None, ok=False,
                            failure_reason=f"Unknown lang {lang!r}, expected one of {list(bundle.ref_config)}")

    t0 = time.perf_counter()
    try:
        pieces = text_segments(text, lang, initialism_set, bundle.mode)
    except Exception as exc:
        return CppTTSResult(**base, audio=None, sr=None, wall_sec=None, rtf=None, ok=False,
                            failure_reason=str(exc)[:300])
    frontend_s = time.perf_counter() - t0
    if not pieces:
        return CppTTSResult(**base, audio=None, sr=None, wall_sec=None, rtf=None, ok=False,
                            failure_reason="no text to speak after the TTS frontend")

    records = []
    for i, (seg_text, pause) in enumerate(pieces):
        records.append({
            "index": i,
            "text": seg_text,
            "chars": len(seg_text),
            "pause_after_s": pause,
            "expected_frames": expected_frames(seg_text),
            "cap": segment_cap(seg_text) if bundle.mode == "segmented" else MAX_NEW_TOKENS_PRIMARY,
            "attempts": [],
            "frames": None,
            "cap_hit": None,
            "audio_s": None,
            "wall_s": None,
            "retries": None,
        })
    segment_audio: list[np.ndarray] = []

    def _gen():
        # Third element is a float, as in src/tts.py and the synthesis
        # notebook: concat_segments only collects it, so this is preserved.
        for rec in records:
            wav = _generate_segment(bundle, id, rec, lang)
            rec["frames"] = len(wav) // HOP_SAMPLES
            rec["cap_hit"] = rec["attempts"][-1]["cap_hit"]
            rec["audio_s"] = len(wav) / SAMPLE_RATE
            rec["wall_s"] = sum(a["wall_s"] or 0.0 for a in rec["attempts"])
            rec["retries"] = len(rec["attempts"]) - 1
            segment_audio.append(wav)
            yield wav, rec["pause_after_s"], rec["wall_s"]

    try:
        bundle.ensure_server()
        audio, timings = ceb_segmentation.concat_segments(_gen(), SAMPLE_RATE)
    except Exception as exc:
        # As in src/tts.py: nothing here may crash the turn. _TextOnly is the
        # designed outcome; anything else (a restart that failed, malformed
        # audio) is recorded with its type so it reads as unexpected.
        reason = str(exc) if isinstance(exc, _TextOnly) else f"{type(exc).__name__}: {exc}"
        return CppTTSResult(**base, audio=None, sr=None, wall_sec=None, rtf=None, ok=False,
                            failure_reason=reason[:600], segments=records, server_restarts=bundle.restarts)

    wall = float(sum(timings))
    dur = len(audio) / SAMPLE_RATE
    return CppTTSResult(
        **base,
        audio=audio,
        sr=SAMPLE_RATE,
        wall_sec=wall,
        rtf=(wall / dur if dur else None),
        ok=True,
        segments=records,
        segment_audio=segment_audio,
        timeline=playback_timeline(frontend_s, records),
        server_restarts=bundle.restarts,
    )


def synthesize(bundle: CppTTSBundle, text: str, lang: str, initialism_set: set):
    """(audio, sr, wall_sec, rtf), raising on any failure, like src/tts.py's
    synthesize. Callers wanting a non-crashing turn use synthesize_turn."""
    result = synthesize_turn(bundle, None, text, lang, initialism_set)
    if not result.ok:
        raise RuntimeError(result.failure_reason)
    return result.audio, result.sr, result.wall_sec, result.rtf
