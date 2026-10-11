"""Stand-in for qwentts.cpp's tts-server in CPU tests of src/tts_cpp.py.

Standard library only. Same command line, same endpoints, same log lines
the patched build prints, so tts_cpp drives it exactly as it drives the real
binary. Generates no speech: a reply is frames x 1920 samples of a quiet tone
at 24 kHz, 16-bit WAV, where frames = round(FRAMES_PER_CHAR x len(input)),
capped at max_new_tokens.

Behaviour comes from a JSON file named by FAKE_TTS_BEHAVIOR (all keys
optional). Each "*_on" key is a substring of the request input:

  frames_per_char   float, default 0.84
  sec_per_frame     float, sleep per generated frame (simulated speed)
  runaway_on        {substring: n}  the first n requests with that substring
                    run to max_new_tokens; use n = -1 for every request
  zero_on           {substring: n}  the first n such requests return 0 frames
  error_on          substring list -> HTTP 502 with a generic message
  oom_on            substring list -> HTTP 502 "fill_qt_audio: malloc failed"
                    unless max_new_tokens <= 2048
  crash_on          substring list -> the process exits mid-request
  hang_on           substring list -> sleep 3600 s before answering
  omit_markers      true -> do not print the patched-build log lines
  exit_at_startup   true -> exit 3 before listening

Every request is appended as one JSON line to FAKE_TTS_REQUEST_LOG when set.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import struct
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = 1920
SR = 24000


def _load_behavior() -> dict:
    path = os.environ.get("FAKE_TTS_BEHAVIOR")
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _log_request(entry: dict) -> None:
    path = os.environ.get("FAKE_TTS_REQUEST_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def _wav_info(data: bytes) -> dict:
    """Minimal RIFF parse: format tag, channels, rate, bits, sample count."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE payload")
    pos, info = 12, {}
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            info.update(format_tag=tag, channels=ch, sample_rate=rate, bits=bits)
        elif cid == b"data":
            info["n_samples"] = size // (info["channels"] * info["bits"] // 8)
        pos += 8 + size + (size & 1)
    return info


# One codec frame of a quiet tone, repeated: fast enough for 4096-frame replies.
_FRAME_PCM = b"".join(struct.pack("<h", int(800 * math.sin(i / 20.0))) for i in range(HOP))


def _wav_bytes(n_frames: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(_FRAME_PCM * n_frames)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    behavior: dict = {}
    counts: dict = {}
    voices: dict = {}

    def log_message(self, fmt, *args):  # keep the server log readable
        pass

    def _json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "server_error"}})

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._error(404, "not found")

    def _take(self, key: str, text: str) -> bool:
        for sub, n in self.behavior.get(key, {}).items():
            if sub in text:
                used = self.counts.get((key, sub), 0)
                if n < 0 or used < n:
                    self.counts[(key, sub)] = used + 1
                    return True
        return False

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        b = self.behavior
        if self.path == "/v1/audio/voices":
            info = _wav_info(base64.b64decode(payload["wav_b64"]))
            _log_request({"path": self.path, "name": payload["name"], "ref_text": payload.get("ref_text"), **info})
            self.voices[payload["name"].lower()] = payload.get("ref_text", "")
            self._json(200, {"name": payload["name"]})
            return
        if self.path != "/v1/audio/speech":
            self._error(404, "not found")
            return

        text = payload["input"]
        _log_request({"path": self.path, **payload})
        max_new = payload.get("max_new_tokens", 2048)
        if payload.get("voice", "").lower() not in self.voices:
            self._error(400, f"unknown voice '{payload.get('voice')}'")
            return
        if any(s in text for s in b.get("crash_on", [])):
            sys.stderr.write("[Fake] crashing on request\n")
            os._exit(1)
        if any(s in text for s in b.get("hang_on", [])):
            time.sleep(3600)
        if any(s in text for s in b.get("error_on", [])):
            self._error(502, "tts_engine_admit: generation failed")
            return
        if any(s in text for s in b.get("oom_on", [])) and max_new > 2048:
            self._error(502, "fill_qt_audio: malloc failed for 1234567 samples")
            return

        if self._take("runaway_on", text):
            frames = max_new
        elif self._take("zero_on", text):
            frames = 0
        else:
            frames = min(max_new, max(1, round(b.get("frames_per_char", 0.84) * len(text))))
        time.sleep(b.get("sec_per_frame", 0.0) * frames)
        body = _wav_bytes(frames)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--icl-non-streaming", action="store_true")
    ap.add_argument("--talker-ctx", type=int, default=4096)
    ap.add_argument("--clamp-fp16", action="store_true")
    args = ap.parse_args()

    behavior = _load_behavior()
    if behavior.get("exit_at_startup"):
        sys.stderr.write("[Fake] exiting at startup\n")
        return 3
    if not behavior.get("omit_markers"):
        layout = "True" if args.icl_non_streaming else "False"
        sys.stderr.write(f"[Qwen] ICL prompt layout: non_streaming_mode={layout}\n")
        sys.stderr.write(f"[Qwen] Talker KV length: {args.talker_ctx} positions\n")
    sys.stderr.flush()

    Handler.behavior = behavior
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f"[Server] listening on {args.host}:{args.port}\n")
    sys.stderr.flush()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
