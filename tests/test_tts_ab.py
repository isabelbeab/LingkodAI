"""CPU-only tests for the Stage 7 A/B helpers: scripts/tts_ab.py (selection,
normalization, summary), scripts/validate_qwentts_patch.py (dump checks on
synthetic dumps) and scripts/fetch_qwentts_weights.py (hashing). No model,
judge, GPU or download is involved."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_qwentts_weights  # noqa: E402
import tts_ab  # noqa: E402
import validate_qwentts_patch as vqp  # noqa: E402


# --- select ----------------------------------------------------------------------


def _fake_ds6(n_per_lang=20):
    rows = []
    for lang in ("ceb", "fil", "eng"):
        for i in range(n_per_lang):
            cid = f"{lang}_01_mt_{i:02d}"
            rows.append({
                "clip_id": cid,
                "conversation_id": "CONV-MT-001" if i < 2 else f"CONV-MT-{i:03d}",
                "clip_lang": lang,
                "final_lang": lang,
                "verbalized_text": " ".join(["word"] * (10 + 10 * i)),
                "tts_rt_wer": 2.5 if (lang, i) == ("fil", 7) else 0.4,
                "duration_sec": 5.0 + i,
                "utmos22": 3.0,
            })
    rows.append({**rows[5], "clip_id": "ceb_01_mt_185", "final_lang": "eng", "conversation_id": "CONV-MT-185"})
    return pd.DataFrame(rows).set_index("clip_id", drop=False)


def test_select_tags_golden_runaway_high_wer_and_strata():
    sel = tts_ab.select_rows(_fake_ds6(), per_stratum=2, seed=0).set_index("clip_id")

    golden = sel.index[sel["tags"].str.contains("golden")]
    assert sorted(golden) == sorted(f"{l}_01_mt_0{i}" for l in ("ceb", "fil", "eng") for i in (0, 1))
    assert sel.loc["ceb_01_mt_185", "tags"] == "runaway"
    assert "high_wer" in sel.loc["fil_01_mt_07", "tags"]
    for lang in ("ceb", "fil", "eng"):
        for q in ("q1", "q2", "q3", "q4"):
            assert ((sel["final_lang"] == lang) & (sel["tags"].str.contains(f"sample_{q}"))).sum() == 2
    # A row is never drawn twice: a sampled row carries one sample tag only.
    assert not sel["tags"].str.contains(r"sample_q\d\+sample").any()


def test_select_is_deterministic_and_seed_dependent():
    df = _fake_ds6()
    a = tts_ab.select_rows(df, 2, seed=0)
    assert a.equals(tts_ab.select_rows(df, 2, seed=0))
    assert set(a["clip_id"]) != set(tts_ab.select_rows(df, 2, seed=1)["clip_id"])


def test_wer_eligibility_is_75_reference_words():
    sel = tts_ab.select_rows(_fake_ds6(), 2, 0)
    assert (sel["wer_eligible"] == (sel["ref_words"] <= 75)).all()
    assert sel["wer_eligible"].any() and not sel["wer_eligible"].all()
    assert tts_ab.ref_words("one two  three") == 3


# --- normalization and summary -------------------------------------------------------------


def test_hyp_normalize_applies_eval_side_to_eng_only():
    assert tts_ab.hyp_normalize("The pee are see office.", "eng", {"PRC"}).split()[1] == "prc"
    fil = tts_ab.hyp_normalize("Ang pi-ar-si office.", "fil", {"PRC"})
    assert "prc" not in fil.split()  # fil keeps the judge's spelling, as DS6's hyp_norm does


def _scores():
    return pd.DataFrame([
        # label, lang, ok, wer_eligible, ref, hyp, utmos, ratio, rtf, ttfa, stalls, cap hits, retries
        ("A", "eng", True, True, "a b c d", "a b c d", 4.0, 1.0, 2.3, 30.0, 0, np.nan, np.nan),
        ("A", "eng", True, True, "a b", "a x", 3.0, 1.2, 2.3, 10.0, 0, np.nan, np.nan),
        ("A", "eng", True, False, "a b c", "zzz", 3.5, 3.1, 2.3, 50.0, 0, np.nan, np.nan),
        ("C", "eng", True, True, "a b c d", "a b c d", 4.1, 1.0, 0.3, 1.0, 0, 0, 0),
        ("C", "eng", False, True, "a b", None, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 1),
    ], columns=["label", "lang", "ok", "wer_eligible", "reference_norm", "hyp_norm", "utmos", "length_ratio", "rtf",
                "ttfa_s", "n_stalls", "cap_hit_attempts", "retries"])


def test_summary_counts_and_corpus_wer():
    ds6 = pd.DataFrame([("eng", True, "a b c d", "a b c x", 3.9, 1.1)],
                       columns=["lang", "wer_eligible", "reference_norm", "hyp_norm", "utmos", "length_ratio"])
    by_lang, overall = tts_ab.summarize(_scores(), ds6)

    a, c, d = overall.loc["A"], overall.loc["C"], overall.loc["DS6 (stored)"]
    assert a["rows"] == 3 and a["text_only"] == 0 and a["wer_rows"] == 2
    assert a["corpus_wer"] == pytest.approx(1 / 6)  # one error over six eligible reference words
    assert a["ratio_over_2"] == 1 and a["ratio_max"] == pytest.approx(3.1)
    assert c["text_only"] == 1 and c["wer_rows"] == 1 and c["corpus_wer"] == 0.0
    assert c["cap_hit_attempts"] == 2 and c["retries"] == 1
    assert d["corpus_wer"] == pytest.approx(0.25)
    assert by_lang.loc[("A", "eng"), "ttfa_median_s"] == pytest.approx(30.0)


def test_load_runs_rejects_duplicates(tmp_path):
    for name in ("r1", "r2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "rows.jsonl").write_text(json.dumps({"label": "X", "clip_id": "c1", "ok": True}) + "\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        tts_ab.load_runs([tmp_path / "r1", tmp_path / "r2"])


def test_synth_refuses_cpp_options_for_arm_a(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tts_ab.py", "synth", "--ds6", "x.xlsx", "--selection", "s.csv", "--arm", "A",
                                      "--label", "A", "--out", "o", "--seed", "1"])
    assert tts_ab.main() == 1
    assert "arm A takes no cpp options" in capsys.readouterr().err


# --- validate_qwentts_patch.check_dumps ---------------------------------------------


def _dump(path: Path, arr) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("i", arr.ndim))
        for s in arr.shape:
            fh.write(struct.pack("i", s))
        fh.write(arr.tobytes())


def _case(tmp_path, *, non_streaming=True, cpp_rows=None, perturb_text=False, prefix=8):
    rng = np.random.default_rng(0)
    n_text, n_ref, ref_t, hidden = 6, 4, 10, 16
    text_lens, codec_lens = n_ref + n_text + 1, 1 + ref_t
    rows = prefix + (text_lens + codec_lens if non_streaming else codec_lens)
    emb = rng.standard_normal((rows, hidden))
    emb_cpp = emb.copy()
    if perturb_text:
        emb_cpp[prefix + 2] = rng.standard_normal(hidden)
    if cpp_rows is not None:
        emb_cpp = emb_cpp[:cpp_rows]
    logits = rng.standard_normal(64)
    for side, e in (("cpp", emb_cpp), ("py", emb)):
        d = tmp_path / side
        _dump(d / "prompt-ids.bin", np.arange(n_text + 8))
        _dump(d / "ref-ids.bin", np.arange(n_ref + 5))
        _dump(d / "ref-codes.bin", np.zeros((16, ref_t)))
        _dump(d / "talker-input-embed.bin", e)
        _dump(d / "talker-logits-prefill.bin", logits)
    return vqp.check_dumps(tmp_path / "cpp", tmp_path / "py", non_streaming)


def test_check_dumps_passes_a_matching_true_layout(tmp_path):
    r = _case(tmp_path)
    assert r["passed"], r["checks"]
    assert r["prefix_rows"] == 8 and r["text_lens"] == 11 and r["text_rows_cos_min"] == pytest.approx(1.0)


def test_check_dumps_passes_the_false_layout_control(tmp_path):
    r = _case(tmp_path, non_streaming=False, prefix=9)
    assert r["passed"] and "text_rows_cos" not in r["checks"]


def test_check_dumps_catches_a_short_embedding_the_harness_cosine_would_miss(tmp_path):
    # The unpatched layout gives the C++ side fewer rows; a truncating cosine
    # over the shared prefix would still read 1.0.
    r = _case(tmp_path, cpp_rows=8 + 11)
    assert not r["passed"] and r["checks"]["embed_shape_equal"] is False


def test_check_dumps_catches_a_wrong_text_row(tmp_path):
    r = _case(tmp_path, perturb_text=True)
    assert not r["passed"] and r["checks"]["text_rows_cos"] is False


# --- fetch_qwentts_weights ----------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "blob"
    p.write_bytes(b"x" * 100_003)
    assert fetch_qwentts_weights.sha256_of(p, chunk=4096) == hashlib.sha256(b"x" * 100_003).hexdigest()


def test_weight_table_is_pinned_and_complete():
    assert len(fetch_qwentts_weights.REVISION) == 40
    for quant, files in fetch_qwentts_weights.FILES.items():
        assert set(files) == {f"qwen-talker-1.7b-base-{quant}.gguf", f"qwen-tokenizer-12hz-{quant}.gguf"}
        assert all(len(h) == 64 and int(h, 16) >= 0 for h in files.values())


# --- score, end to end with the judge and UTMOS faked ----------------------------------


def test_score_end_to_end_with_fake_judge_and_utmos(tmp_path, monkeypatch):
    import types

    ds6 = pd.DataFrame([
        {"clip_id": "eng_01_mt_01", "final_lang": "eng", "verbalized_text": "bring your ID",
         "reference_norm": "bring your id", "hyp_norm": "bring your id", "tts_rt_wer": 0.0, "utmos22": 4.2,
         "duration_sec": 1.2},
        {"clip_id": "fil_01_mt_01", "final_lang": "fil", "verbalized_text": "dalhin ang ID",
         "reference_norm": "dalhin ang id", "hyp_norm": "dalhin ang id", "tts_rt_wer": 0.0, "utmos22": 3.5,
         "duration_sec": 1.1},
    ]).set_index("clip_id", drop=False)
    monkeypatch.setattr(tts_ab, "load_ds6", lambda path: ds6)

    for label, ok_fil in (("A-hf", True), ("C-seg", False)):
        run = tmp_path / label
        (run / "wavs").mkdir(parents=True)
        rows = []
        for cid, lang, ok in (("eng_01_mt_01", "eng", True), ("fil_01_mt_01", "fil", ok_fil)):
            rows.append({"arm": label[0], "label": label, "clip_id": cid, "lang": lang, "tags": "golden", "ok": ok,
                         "failure_reason": None if ok else "segment 0: no clean generation", "wall_sec": 1.0,
                         "rtf": 0.5, "audio_s": 2.0, "chars": 13, "expected_s": 0.87, "length_ratio": 2.3,
                         "timeline": {"ttfa_s": 0.4, "n_stalls": 1}, "cap_hit_attempts": 0, "retries": 0})
            if ok:
                (run / "wavs" / f"{cid}.wav").write_bytes(b"")
        (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    judged = []
    monkeypatch.setattr(tts_ab, "build_judge", lambda lang, device: (lang, None))
    monkeypatch.setattr(tts_ab, "judge_transcribe",
                        lambda paths, proc, model, device: judged.extend((proc, p.name) for p in paths)
                        or ["Bring your I D." if proc == "eng" else "Dalhin ang ID."] * len(paths))
    monkeypatch.setattr(tts_ab, "utmos_score", lambda path, utmos, device: 3.0)
    monkeypatch.setitem(sys.modules, "utmos_pytorch",
                        types.SimpleNamespace(UTMOSScoreTorch=lambda device: object()))
    out = tmp_path / "scores"
    monkeypatch.setattr(sys, "argv", ["tts_ab.py", "score", "--ds6", "x.xlsx", "--runs", str(tmp_path / "A-hf"),
                                      str(tmp_path / "C-seg"), "--out", str(out)])

    assert tts_ab.main() == 0

    assert sorted(judged) == [("eng", "eng_01_mt_01.wav"), ("eng", "eng_01_mt_01.wav"), ("fil", "fil_01_mt_01.wav")]
    scores = pd.read_csv(out / "scores.csv")
    assert len(scores) == 4 and scores["ok"].sum() == 3
    summary = pd.read_csv(out / "summary.csv", index_col=0)
    assert list(summary.index) == ["A-hf", "C-seg", "DS6 (stored)"]
    assert summary.loc["C-seg", "text_only"] == 1 and summary.loc["A-hf", "text_only"] == 0
    assert summary.loc["A-hf", "utmos_mean"] == pytest.approx(3.0)
    assert summary.loc["DS6 (stored)", "corpus_wer"] == 0.0
    # A rerun reuses the cached judge output and UTMOS values.
    judged.clear()
    assert tts_ab.main() == 0 and judged == []
