"""CPU smoke tests for scripts/ab_asr_checkpoints.py (no models, no GPU)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ab_asr_checkpoints as ab  # noqa: E402


def _golden_dir(tmp_path: Path) -> Path:
    golden = tmp_path / "golden"
    golden.mkdir()
    for lang in ab.LANGS:
        for i in (1, 2):
            clip_id = f"{lang}_{i:02d}"
            wav = tmp_path / f"{clip_id}.wav"
            wav.write_bytes(b"")
            (golden / f"{clip_id}.json").write_text(
                json.dumps({"clip_id": clip_id, "audio_path": str(wav), "final_lang": lang})
            )
    (golden / "report.json").write_text("{}")
    return golden


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    roots = (tmp_path / "old", tmp_path / "new")
    for root in roots:
        for lang in ab.LANGS:
            (root / lang).mkdir(parents=True)
            (root / lang / "config.json").write_text("{}")
    return roots


def test_load_clips_groups_by_final_lang(tmp_path):
    clips = ab.load_clips(_golden_dir(tmp_path))
    assert {lang: [c for c, _ in items] for lang, items in clips.items()} == {
        "ceb": ["ceb_01", "ceb_02"], "fil": ["fil_01", "fil_02"], "eng": ["eng_01", "eng_02"],
    }


def test_load_clips_missing_audio_raises(tmp_path):
    golden = _golden_dir(tmp_path)
    (tmp_path / "fil_01.wav").unlink()
    with pytest.raises(FileNotFoundError):
        ab.load_clips(golden)


def test_load_clips_missing_language_raises(tmp_path):
    golden = _golden_dir(tmp_path)
    for p in golden.glob("eng_*.json"):
        p.unlink()
    with pytest.raises(ValueError):
        ab.load_clips(golden)


def _run(monkeypatch, tmp_path, fake_transcribe) -> int:
    golden = _golden_dir(tmp_path)
    old, new = _roots(tmp_path)
    monkeypatch.setattr(ab, "print_environment", lambda args: None)
    monkeypatch.setattr("src.audio.load_audio", lambda path: path)
    monkeypatch.setattr(ab, "transcribe_all", fake_transcribe)
    monkeypatch.setattr(sys, "argv", ["ab", "--old-root", str(old), "--new-root", str(new), "--golden-dir", str(golden)])
    return ab.main()


def test_identical_transcripts_exit_0(monkeypatch, tmp_path):
    def fake(checkpoint, lang, audio_by_clip):
        return {clip_id: f"text {clip_id}" for clip_id in audio_by_clip}

    assert _run(monkeypatch, tmp_path, fake) == 0


def test_one_diff_exit_2(monkeypatch, tmp_path, capsys):
    def fake(checkpoint, lang, audio_by_clip):
        out = {clip_id: f"text {clip_id}" for clip_id in audio_by_clip}
        if checkpoint.parent.name == "new" and lang == "ceb":
            out["ceb_02"] = "something else"
        return out

    assert _run(monkeypatch, tmp_path, fake) == 2
    out = capsys.readouterr().out
    assert "ceb_02: DIFFERS" in out
    assert "5/6 transcripts identical" in out


def test_missing_checkpoint_exit_1(monkeypatch, tmp_path):
    golden = _golden_dir(tmp_path)
    old, new = _roots(tmp_path)
    (new / "fil" / "config.json").unlink()
    monkeypatch.setattr(ab, "print_environment", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["ab", "--old-root", str(old), "--new-root", str(new), "--golden-dir", str(golden)])
    assert ab.main() == 1
