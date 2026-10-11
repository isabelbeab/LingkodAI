"""CPU-only test of scripts/run_conversation.py's Stage 7 backend flags and
output files. run_staged is replaced by a fake; no stage runs for real."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_conversation  # noqa: E402

from src import pipeline, routing  # noqa: E402


def _fake_result(backend):
    t = pipeline.TurnRecord(turn_idx=0, audio_path="a.wav", final_lang="eng", native_answer="answer")
    t.audio_out, t.audio_out_sr, t.tts_ok, t.tts_backend = np.zeros(4800, dtype=np.float32), 24000, True, backend
    t.tts_wall_sec, t.tts_rtf = 0.5, 0.25
    if backend == "cpp":
        t.tts_segments = [
            {"index": 0, "text": "one.", "chars": 4, "pause_after_s": 0.45, "expected_frames": 3.4, "cap": 32,
             "frames": 1, "cap_hit": False, "retries": 0, "audio_s": 0.08, "wall_s": 0.2, "attempts": []},
            {"index": 1, "text": "two.", "chars": 4, "pause_after_s": 0.45, "expected_frames": 3.4, "cap": 32,
             "frames": 1, "cap_hit": False, "retries": 1, "audio_s": 0.08, "wall_s": 0.3, "attempts": []},
        ]
        t.tts_timeline = {"ttfa_s": 0.21, "n_stalls": 1, "total_stall_s": 0.1}
        t.tts_segment_audio = [np.zeros(1920, dtype=np.float32), np.zeros(1920, dtype=np.float32)]
    conv = routing.ConversationRouting.from_first_turn("eng", "eng")
    return pipeline.ConversationResult(turns=[t], routing=conv, tts_info={"base_seed": 7} if backend == "cpp" else None)


@pytest.fixture
def run(tmp_path, monkeypatch):
    calls = []

    def fake_run_staged(audio_paths, init_set, **kw):
        calls.append(kw)
        return _fake_result(kw["tts_backend"])

    monkeypatch.setattr(pipeline, "run_staged", fake_run_staged)
    audio = tmp_path / "turn.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    out = tmp_path / "out"

    def _run(*extra):
        monkeypatch.setattr(sys, "argv", ["run_conversation.py", "--audio", str(audio), "--out", str(out), *extra])
        return run_conversation.main()

    _run.calls, _run.out = calls, out
    return _run


def test_cpp_backend_writes_segments_timeline_and_manifest(run):
    assert run("--tts-backend", "cpp", "--tts-mode", "segmented", "--tts-seed", "7",
               "--tts-server-arg=--clamp-fp16") == 0

    opts = run.calls[0]["tts_options"]
    assert opts == {"mode": "segmented", "seed": 7, "extra_server_args": ("--clamp-fp16",), "log_dir": run.out}
    files = sorted(p.name for p in run.out.iterdir())
    assert files == ["manifest.json", "timeline.json", "turn_0.json", "turn_0.wav", "turn_0_seg00.wav",
                     "turn_0_seg01.wav"]
    manifest = json.loads((run.out / "manifest.json").read_text())
    assert manifest["tts_backend"] == "cpp" and manifest["tts"] == {"base_seed": 7}
    timeline = json.loads((run.out / "timeline.json").read_text())
    assert timeline[0]["timeline"]["n_stalls"] == 1
    assert [s["retries"] for s in timeline[0]["segments"]] == [0, 1]
    assert "attempts" not in timeline[0]["segments"][0] and "text" not in timeline[0]["segments"][0]
    turn = json.loads((run.out / "turn_0.json").read_text())
    assert turn["tts_segments"][1]["text"] == "two." and "tts_segment_audio" not in turn


def test_hf_backend_is_the_default_and_writes_no_timeline(run):
    assert run() == 0

    assert run.calls[0]["tts_backend"] == "hf" and run.calls[0]["tts_options"] is None
    assert not (run.out / "timeline.json").exists()
    assert json.loads((run.out / "manifest.json").read_text())["tts_backend"] == "hf"


def test_cpp_only_flags_are_refused_with_the_hf_backend(run, capsys):
    assert run("--tts-mode", "segmented") == 1
    assert "only valid with --tts-backend cpp: --tts-mode" in capsys.readouterr().err
    assert run.calls == []
