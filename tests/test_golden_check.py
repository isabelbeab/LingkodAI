"""CPU-only smoke test for scripts/golden_check.py's pure logic: parsing
DS6's clip_id scheme, picking which conversation_id to check, path
remapping, and the exact-match bookkeeping. Never loads the real DS6 xlsx,
runs the pipeline, or touches audio -- those need JOJIE (see the script's
own docstring)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import golden_check  # noqa: E402


REQUIRED_COLS = [
    "clip_id", "conversation_id", "turn_id", "final_lang",
    "source_audio_path", "audio_path_16k",
    "transcript_hypothesis", "translated_to_english", "rag_answer", "rag_answer_native",
]


def _row(clip_id, conversation_id, turn_id, final_lang="ceb"):
    return {
        "clip_id": clip_id,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "final_lang": final_lang,
        "source_audio_path": f"/fake/{clip_id}.m4a",
        "audio_path_16k": f"/fake16k/{clip_id}.wav",
        "transcript_hypothesis": f"transcript for {clip_id}",
        "translated_to_english": f"english for {clip_id}",
        "rag_answer": f"answer for {clip_id}",
        "rag_answer_native": f"native answer for {clip_id}",
    }


def write_xlsx(tmp_path, rows) -> Path:
    path = tmp_path / "ds6_fake.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


# --- load_ds6 --------------------------------------------------------------


def test_load_ds6_parses_clip_id_into_lang_and_turn(tmp_path):
    rows = [
        _row("ceb_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01"),
        _row("ceb_01_mt_02", "CONV-MT-001", "CONV-MT-001-T02"),
        _row("eng_01_mt_001", "CONV-MT-001", "CONV-MT-001-T01", final_lang="eng"),
    ]
    df = golden_check.load_ds6(write_xlsx(tmp_path, rows))

    assert list(df["clip_lang"]) == ["ceb", "ceb", "eng"]
    assert list(df["clip_turn"]) == [1, 2, 1]


def test_load_ds6_raises_on_missing_column(tmp_path):
    rows = [_row("ceb_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01")]
    df = pd.DataFrame(rows).drop(columns=["rag_answer_native"])
    path = tmp_path / "ds6_bad.xlsx"
    df.to_excel(path, index=False)

    with pytest.raises(RuntimeError, match="missing expected column"):
        golden_check.load_ds6(path)


def test_load_ds6_raises_on_unparseable_clip_id(tmp_path):
    rows = [_row("not_a_valid_clip_id", "CONV-MT-001", "CONV-MT-001-T01")]
    with pytest.raises(RuntimeError, match="clip_id"):
        golden_check.load_ds6(write_xlsx(tmp_path, rows))


# --- pick_conversation_id ----------------------------------------------


def test_pick_conversation_id_auto_selects_first_with_all_three_langs(tmp_path):
    rows = [
        _row("ceb_01_mt_01", "CONV-MT-002", "CONV-MT-002-T01"),  # missing fil/eng
        _row("ceb_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01"),
        _row("fil_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01"),
        _row("eng_01_mt_001", "CONV-MT-001", "CONV-MT-001-T01"),
    ]
    df = golden_check.load_ds6(write_xlsx(tmp_path, rows))

    assert golden_check.pick_conversation_id(df, None) == "CONV-MT-001"


def test_pick_conversation_id_respects_explicit_override(tmp_path):
    rows = [
        _row("ceb_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01"),
        _row("fil_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01"),
        _row("eng_01_mt_001", "CONV-MT-001", "CONV-MT-001-T01"),
    ]
    df = golden_check.load_ds6(write_xlsx(tmp_path, rows))

    assert golden_check.pick_conversation_id(df, "CONV-MT-999") == "CONV-MT-999"


def test_pick_conversation_id_raises_when_none_has_all_three(tmp_path):
    rows = [_row("ceb_01_mt_01", "CONV-MT-001", "CONV-MT-001-T01")]
    df = golden_check.load_ds6(write_xlsx(tmp_path, rows))

    with pytest.raises(RuntimeError, match="No conversation_id"):
        golden_check.pick_conversation_id(df, None)


# --- resolve_audio_path ------------------------------------------------


def test_resolve_audio_path_remaps_matching_prefix():
    result = golden_check.resolve_audio_path(
        "/home2/msds2026/ibucayan/clip.m4a", {"/home2/msds2026/ibucayan": "/mnt/jojie"}
    )
    assert result == "/mnt/jojie/clip.m4a"


def test_resolve_audio_path_leaves_unmatched_path_unchanged():
    result = golden_check.resolve_audio_path("/other/clip.m4a", {"/home2/msds2026/ibucayan": "/mnt/jojie"})
    assert result == "/other/clip.m4a"


# --- FieldComparison -----------------------------------------------------


def test_field_comparison_tracks_matches_and_diffs():
    comp = golden_check.FieldComparison(field="transcript")
    comp.record("clip1", "hello", "hello")
    comp.record("clip2", "hello", "goodbye")
    comp.record("clip3", "hello", None)  # None actual counts as a mismatch, not a crash

    assert comp.n_total == 3
    assert comp.n_match == 1
    assert comp.exact_match_rate == pytest.approx(1 / 3)
    assert [d["clip_id"] for d in comp.diffs] == ["clip2", "clip3"]
