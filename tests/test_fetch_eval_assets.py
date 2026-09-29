"""CPU smoke tests for scripts/fetch_eval_assets.py (no network)."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_eval_assets as f  # noqa: E402


def _stage(root: Path, n_ref=3, n_audio=24, ds6=True) -> dict[str, bytes]:
    files = {f"tts_ref/ref{i}.wav": f"ref{i}".encode() for i in range(n_ref)}
    files.update({f"golden/audio/multi_turn/01_ceb_mt_audio/t{i}.m4a": f"a{i}".encode() for i in range(n_audio)})
    if ds6:
        files["golden/DS6_augmented.xlsx"] = b"ds6"
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    lines = [f"{hashlib.sha256(d).hexdigest()}  {rel}" for rel, d in files.items()]
    (root / f.MANIFEST).write_text("\n".join(lines) + "\n")
    return files


def test_clean_download_verifies(tmp_path):
    _stage(tmp_path)
    assert f.verify(tmp_path) == []


def test_extra_unlisted_files_are_ignored(tmp_path):
    _stage(tmp_path)
    (tmp_path / "README.md").write_text("card")
    (tmp_path / ".cache").mkdir()
    assert f.verify(tmp_path) == []


def test_changed_file_is_reported(tmp_path):
    _stage(tmp_path)
    (tmp_path / "golden/DS6_augmented.xlsx").write_bytes(b"tampered")
    problems = f.verify(tmp_path)
    assert len(problems) == 1 and problems[0].startswith("sha256 mismatch: golden/DS6_augmented.xlsx")


def test_missing_file_is_reported(tmp_path):
    _stage(tmp_path)
    (tmp_path / "tts_ref/ref0.wav").unlink()
    assert f.verify(tmp_path) == ["missing: tts_ref/ref0.wav"]


def test_short_manifest_fails_on_counts(tmp_path):
    _stage(tmp_path, n_audio=23)
    problems = f.verify(tmp_path)
    assert any("23 golden audio clips, expected 24" in p for p in problems)
    assert any("27 entries, expected 28" in p for p in problems)


def test_missing_manifest(tmp_path):
    assert f.verify(tmp_path) == [f"{tmp_path / f.MANIFEST} does not exist"]


def test_parse_manifest_rejects_malformed_and_duplicates():
    good = "a" * 64
    assert f.parse_manifest(f"{good}  x/y.wav\n\n") == {"x/y.wav": good}
    with pytest.raises(ValueError, match="malformed"):
        f.parse_manifest("abc  x.wav")
    with pytest.raises(ValueError, match="twice"):
        f.parse_manifest(f"{good}  x.wav\n{good}  x.wav")
