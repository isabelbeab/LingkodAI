"""CPU-only tests for scripts/run_folder.py. pipeline.run_staged is replaced
with a fake that builds TurnRecords from the file names, so no model loads.
The output files are written by run_conversation.write_conversation, the
same code the CLI uses."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_folder as rf  # noqa: E402

from src import pipeline  # noqa: E402

INIT_SET = Path(__file__).resolve().parents[1] / "data" / "init_set_ds6.json"


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return path


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Records each call; a path containing 'boom' raises like a failed stage,
    one containing 'mute' comes back text-only."""
    calls = []

    def fake_run_staged(paths, init_set, *, rag_data_dir=None):
        calls.append(list(paths))
        if any("boom" in p for p in paths):
            raise RuntimeError("MT in failed")
        turns = []
        for i, p in enumerate(paths):
            spoken = "mute" not in p
            turns.append(pipeline.TurnRecord(
                turn_idx=i, audio_path=p, transcript=f"heard {Path(p).name}", final_lang="fil",
                english_query="q", english_answer="a", native_answer=f"sagot {i}",
                chunk_ids=["EXT-073"], audio_out=np.zeros(240, dtype=np.float32) if spoken else None,
                audio_out_sr=24000, tts_ok=spoken, tts_failure_reason=None if spoken else "frontend raised"))
        return SimpleNamespace(turns=turns, routing=SimpleNamespace(final_lang="fil", was_overridden=False))

    monkeypatch.setattr(pipeline, "run_staged", fake_run_staged)
    return calls


def read_summary(out: Path) -> list[dict]:
    with open(out / "summary.csv", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_discover_files_are_single_questions_and_folders_are_conversations(tmp_path):
    touch(tmp_path / "b_question.m4a")
    touch(tmp_path / "a_conv" / "02_follow.wav")
    touch(tmp_path / "a_conv" / "01_first.MP3")
    touch(tmp_path / "a_conv" / "notes.txt")
    touch(tmp_path / "readme.md")
    touch(tmp_path / ".hidden.wav")
    (tmp_path / "empty").mkdir()
    found = rf.discover(tmp_path)
    assert [(name, [p.name for p in turns]) for name, turns in found] == [
        ("a_conv", ["01_first.MP3", "02_follow.wav"]),
        ("b_question", ["b_question.m4a"]),
    ]


def test_discover_refuses_two_inputs_with_the_same_output_name(tmp_path):
    touch(tmp_path / "q1.m4a")
    touch(tmp_path / "q1.wav")
    with pytest.raises(ValueError, match="q1"):
        rf.discover(tmp_path)


def test_discover_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rf.discover(tmp_path / "nope")


def test_run_folder_writes_each_conversation_and_a_summary(tmp_path, fake_pipeline):
    inp, out = tmp_path / "input", tmp_path / "output"
    touch(inp / "single.m4a")
    touch(inp / "conv" / "01.wav")
    touch(inp / "conv" / "02_mute.wav")
    assert rf.run_folder(inp, out, INIT_SET) == 0

    assert fake_pipeline == [[str(inp / "conv" / "01.wav"), str(inp / "conv" / "02_mute.wav")],
                             [str(inp / "single.m4a")]]
    for name in ("conv", "single"):
        assert (out / name / "manifest.json").exists()
    assert (out / "conv" / "turn_0.wav").exists()
    assert not (out / "conv" / "turn_1.wav").exists()

    rows = read_summary(out)
    assert [(r["conversation"], r["turn"], r["input_file"], r["status"], r["audio_file"]) for r in rows] == [
        ("conv", "1", "01.wav", "ok", "conv/turn_0.wav"),
        ("conv", "2", "02_mute.wav", "text only: frontend raised", ""),
        ("single", "1", "single.m4a", "ok", "single/turn_0.wav"),
    ]
    assert rows[2]["transcript"] == "heard single.m4a"
    assert rows[2]["native_answer"] == "sagot 0"
    assert rows[2]["chunk_ids"] == "EXT-073"


def test_finished_conversations_are_skipped_but_stay_in_the_summary(tmp_path, fake_pipeline):
    inp, out = tmp_path / "input", tmp_path / "output"
    touch(inp / "first.m4a")
    rf.run_folder(inp, out, INIT_SET)
    touch(inp / "second.m4a")
    assert rf.run_folder(inp, out, INIT_SET) == 0
    assert fake_pipeline == [[str(inp / "first.m4a")], [str(inp / "second.m4a")]]
    assert [r["conversation"] for r in read_summary(out)] == ["first", "second"]


def test_force_redoes_finished_conversations(tmp_path, fake_pipeline):
    inp, out = tmp_path / "input", tmp_path / "output"
    touch(inp / "first.m4a")
    rf.run_folder(inp, out, INIT_SET)
    rf.run_folder(inp, out, INIT_SET, force=True)
    assert len(fake_pipeline) == 2


def test_a_failed_conversation_is_reported_and_the_rest_still_run(tmp_path, fake_pipeline):
    inp, out = tmp_path / "input", tmp_path / "output"
    touch(inp / "a_boom.m4a")
    touch(inp / "b_fine.m4a")
    assert rf.run_folder(inp, out, INIT_SET) == 1
    assert not (out / "a_boom" / "manifest.json").exists()  # so the next run retries it
    rows = read_summary(out)
    assert [(r["conversation"], r["status"]) for r in rows] == [("a_boom", "failed"), ("b_fine", "ok")]
    assert rows[0]["error"] == "RuntimeError: MT in failed"


def test_empty_input_folder_does_nothing(tmp_path, fake_pipeline):
    (tmp_path / "input").mkdir()
    assert rf.run_folder(tmp_path / "input", tmp_path / "output", INIT_SET) == 0
    assert fake_pipeline == []
