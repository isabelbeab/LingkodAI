"""CPU-only tests for src/tts_cpp.py. Never runs qwentts.cpp or loads a model.

Text path: the real vendored tts_frontend modules, checked against
src/tts.py's own path (driven through test_tts.py's fake qwen-tts model).
Server path: tests/fake_tts_server.py, a stdlib stand-in launched exactly the
way tts_cpp launches the real tts-server (same flags, same endpoints, same
log lines), with scripted runaways, errors, crashes and hangs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import tts, tts_cpp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tts import make_bundle  # noqa: E402

FAKE_SERVER = Path(__file__).resolve().parent / "fake_tts_server.py"
REF_SR = 16000
REF_SECONDS = 1.0

FIL_TEXT = (
    "Ayon sa PRB Resolution No. 02-2025, kailangan ng dalawang kopya ng aplikasyon. "
    "Dalhin din ang isang valid ID at ang resibo ng bayad."
)
ENG_TEXT = "Bring these documents. 1. Your application form. 2. One valid ID."
CEB_TEXT = "Dad-a ang imong aplikasyon. Ang proseso molungtad og usa ka semana."


# --- text path -----------------------------------------------------------


@pytest.mark.parametrize("lang,text", [("eng", ENG_TEXT), ("fil", FIL_TEXT), ("ceb", CEB_TEXT)])
def test_parity_sends_exactly_what_src_tts_sends(tmp_path, lang, text):
    hf_bundle = make_bundle(tmp_path)
    tts.synthesize(hf_bundle, text, lang, set())
    sent_by_hf = [call["text"] for call in hf_bundle.model.calls]

    pieces = tts_cpp.text_segments(text, lang, set(), "parity")

    assert [t for t, _ in pieces] == sent_by_hf


def test_parity_fil_eng_is_one_segment_without_pause():
    for lang, text in (("eng", ENG_TEXT), ("fil", FIL_TEXT)):
        pieces = tts_cpp.text_segments(text, lang, set(), "parity")
        assert len(pieces) == 1 and pieces[0][1] == 0.0


def test_segmented_splits_eng_before_verbalizing_and_speaks_list_numbers():
    pieces = tts_cpp.text_segments(ENG_TEXT, "eng", set(), "segmented")

    assert [t for t, _ in pieces] == ["Bring these documents.", "one, Your application form.", "two, One valid ID."]
    assert [p for _, p in pieces] == [0.45, 0.60, 0.60]


def test_segmented_fil_list_markers_use_fil_ordinals():
    pieces = tts_cpp.text_segments("Kailangan: 1. Aplikasyon. 2. Resibo.", "fil", set(), "segmented")

    assert [t for t, _ in pieces] == ["Kailangan:", "una, Aplikasyon.", "pangalawa, Resibo."]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Basaha ang PRB Resolution No. 02-2025. Sunod nga lakang.", "Basaha ang PRB Resolution No. 02-2025."),
        ("Under R.A. No. 9298 the rules apply. Next step.", "Under R.A. No. 9298 the rules apply."),
        ("See CMO 17 (s. 2008) for details. Next step.", "See CMO 17 (s. 2008) for details."),
        ("Graduates of A.Y. 2018 qualify. Next step.", "Graduates of A.Y. 2018 qualify."),
    ],
)
def test_abbreviation_guard_keeps_number_attached(text, expected):
    plain = [s for s, _ in tts_cpp.ceb_segmentation.split_for_synthesis(text)]
    guarded = [s for s, _ in tts_cpp._split_guarded(text, "eng")]

    assert len(guarded) < len(plain)
    assert expected in guarded
    assert " ".join(guarded).split() == " ".join(plain).split()


@pytest.mark.parametrize(
    "text",
    [
        "Pay the fee. 2 copies are needed. Submit them today.",  # sentence end before a digit
        "They said No. Then they left.",  # "No." not followed by a number
        "Check the days. 3 offices are open.",  # "s." inside a word, not the series abbreviation
    ],
)
def test_abbreviation_guard_leaves_ordinary_sentences_alone(text):
    assert tts_cpp._split_guarded(text, "eng") == tts_cpp.ceb_segmentation.split_for_synthesis(
        text, ordinal_words=tts_cpp._SPLIT_ORDINALS["eng"]
    )


def test_segmented_ceb_matches_parity_when_guard_does_not_fire():
    assert tts_cpp.text_segments(CEB_TEXT, "ceb", set(), "segmented") == tts_cpp.text_segments(
        CEB_TEXT, "ceb", set(), "parity"
    )


def test_guard_placeholder_in_input_raises():
    with pytest.raises(ValueError, match="placeholder"):
        tts_cpp.text_segments("bad  text", "eng", set(), "segmented")


def test_segment_cap_math():
    assert tts_cpp.segment_cap("x" * 100) == 168 + 25  # ceil(2 * 0.84 * 100) + 25
    assert tts_cpp.segment_cap("x" * 3) == 31  # ceil(5.04) + 25
    assert tts_cpp.segment_cap("x" * 5000) == 4096  # never above the evaluated max_new_tokens


def test_request_seed_is_deterministic_and_distinct():
    a = tts_cpp.request_seed(7, 3, 0, 0)
    assert a == tts_cpp.request_seed(7, 3, 0, 0)
    others = {tts_cpp.request_seed(7, 3, 0, 1), tts_cpp.request_seed(7, 3, 1, 0), tts_cpp.request_seed(7, 4, 0, 0),
              tts_cpp.request_seed(8, 3, 0, 0)}
    assert a not in others and len(others) == 4
    assert 0 <= a < 2**31


# --- playback timeline -----------------------------------------------------


def _seg(wall, audio, pause):
    return {"wall_s": wall, "audio_s": audio, "pause_after_s": pause}


def test_timeline_stall_when_a_long_segment_follows_a_short_one():
    tl = tts_cpp.playback_timeline(0.1, [_seg(1.0, 2.0, 0.2), _seg(3.0, 2.0, 0.2), _seg(1.0, 2.0, 0.2)])

    assert tl["ttfa_s"] == pytest.approx(1.1)
    assert tl["ready_s"] == pytest.approx([1.1, 4.1, 5.1])
    # seg 0 plays 1.1 to 3.3; seg 1 is ready at 4.1: a 0.8 s stall; seg 2 waits for seg 1 to end.
    assert tl["n_stalls"] == 1
    assert tl["stalls"][0]["segment"] == 1 and tl["stalls"][0]["stall_s"] == pytest.approx(0.8)
    assert tl["playback_end_s"] == pytest.approx(8.5)
    # Starting at 1.9 s covers every segment: max(1.1 - 0, 4.1 - 2.2, 5.1 - 4.4).
    assert tl["no_stall_start_s"] == pytest.approx(1.9)
    assert tl["audio_total_s"] == pytest.approx(6.6)


def test_timeline_no_stall_when_generation_outpaces_playback():
    tl = tts_cpp.playback_timeline(0.0, [_seg(0.5, 3.0, 0.45), _seg(0.5, 3.0, 0.45)])

    assert tl["n_stalls"] == 0 and tl["total_stall_s"] == 0.0
    assert tl["no_stall_start_s"] == pytest.approx(tl["ttfa_s"]) == pytest.approx(0.5)


# --- lead-in trim ----------------------------------------------------------


def _tone(seconds, sr=24000, amp=0.1):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_lead_in_keeps_20_ms_before_the_first_sound():
    wav = np.concatenate([np.zeros(int(0.30 * 24000), np.float32), _tone(0.5)])

    out, cut = tts_cpp.trim_lead_in(wav)

    assert cut == pytest.approx(0.28)
    assert np.array_equal(out, wav[int(0.28 * 24000):])


def test_trim_lead_in_treats_sound_under_the_threshold_as_silence():
    hiss = np.full(int(0.20 * 24000), 5e-4, np.float32)  # RMS 5e-4, under 1e-3

    out, cut = tts_cpp.trim_lead_in(np.concatenate([hiss, _tone(0.5)]))

    assert cut == pytest.approx(0.18)


@pytest.mark.parametrize(
    "wav",
    [_tone(0.5), np.concatenate([np.zeros(200, np.float32), _tone(0.5)]), np.zeros(24000, np.float32),
     np.zeros(100, np.float32), np.zeros(0, np.float32)],
    ids=["starts-at-once", "within-20-ms", "all-silent", "shorter-than-a-frame", "empty"],
)
def test_trim_lead_in_leaves_alone_audio_with_nothing_to_trim(wav):
    out, cut = tts_cpp.trim_lead_in(wav)

    assert cut == 0.0 and np.array_equal(out, wav)


# --- fake server harness ------------------------------------------------------


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Paths for a fake tts-server run; .load(**kw) starts one and registers
    it for teardown, .requests() reads what the server received."""
    refs = tmp_path / "refs"
    refs.mkdir()
    rng = np.random.default_rng(0)
    for filename in tts.REF_CLIP_FILENAMES.values():
        sf.write(refs / filename, (rng.standard_normal(int(REF_SR * REF_SECONDS)) * 0.05).astype(np.float32), REF_SR)
    (tmp_path / "talker.gguf").write_bytes(b"")
    (tmp_path / "codec.gguf").write_bytes(b"")
    wrapper = tmp_path / "tts-server"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SERVER}" "$@"\n')
    wrapper.chmod(0o755)
    behavior_path = tmp_path / "behavior.json"
    request_log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_TTS_BEHAVIOR", str(behavior_path))
    monkeypatch.setenv("FAKE_TTS_REQUEST_LOG", str(request_log))
    bundles = []

    class Fake:
        dir = tmp_path

        def load(self, behavior=None, **kw):
            behavior_path.write_text(json.dumps(behavior or {}))
            kw.setdefault("mode", "segmented")
            kw.setdefault("seed", 1234)
            bundle = tts_cpp.load(
                refs, server_bin=wrapper, talker_gguf=tmp_path / "talker.gguf", codec_gguf=tmp_path / "codec.gguf",
                log_dir=tmp_path / "logs", startup_timeout_s=30, **kw,
            )
            bundles.append(bundle)
            return bundle

        def requests(self, path=None):
            if not request_log.exists():
                return []
            rows = [json.loads(line) for line in request_log.read_text().splitlines()]
            return [r for r in rows if path is None or r["path"] == path]

    yield Fake()
    for b in bundles:
        b.unload()


def _speech(fake):
    return [r for r in fake.requests("/v1/audio/speech") if r["input"] != "test"]  # drop the warm-up


# --- load ------------------------------------------------------------------


def test_load_registers_three_voices_at_24k_float(fake):
    bundle = fake.load()

    regs = fake.requests("/v1/audio/voices")
    assert sorted(r["name"] for r in regs) == ["lingkod-ceb", "lingkod-eng", "lingkod-fil"]
    padded_samples = round((REF_SECONDS + tts.REF_PAD_SILENCE_SEC) * 24000)
    for r in regs:
        assert (r["sample_rate"], r["channels"], r["format_tag"], r["bits"]) == (24000, 1, 3, 32)
        assert abs(r["n_samples"] - padded_samples) <= 2
        assert r["ref_text"] == tts.REF_TEXT_CONFIG[r["name"].split("-")[1]]["ref_text"]
    assert bundle.info["base_seed"] == 1234 and bundle.info["mode"] == "segmented"
    assert "--icl-non-streaming" in bundle.info["server_cmd"]
    assert bundle.info["server_cmd"][bundle.info["server_cmd"].index("--talker-ctx") + 1] == "7168"


def test_load_rejects_an_unpatched_binary(fake):
    with pytest.raises(RuntimeError, match="not the patched build"):
        fake.load({"omit_markers": True})


def test_load_raises_when_the_server_exits_at_startup(fake):
    with pytest.raises(tts_cpp.ServerCrashed, match="exited with code 3"):
        fake.load({"exit_at_startup": True})


def test_load_raises_on_missing_binary(fake, tmp_path):
    with pytest.raises(FileNotFoundError, match="tts-server binary"):
        tts_cpp.load(tmp_path, server_bin=tmp_path / "nope", talker_gguf=tmp_path / "talker.gguf",
                     codec_gguf=tmp_path / "codec.gguf")


def test_load_requires_gguf_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("QWENTTS_TALKER_GGUF", raising=False)
    with pytest.raises(ValueError, match="QWENTTS_TALKER_GGUF"):
        tts_cpp.load(tmp_path, server_bin=tmp_path / "x")


def test_load_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="Unknown mode"):
        tts_cpp.load(tmp_path, mode="fast")


def test_unload_kills_the_server_process(fake):
    bundle = fake.load()
    pid = bundle.server.proc.pid

    bundle.unload()

    assert not bundle.server.is_alive()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# --- synthesis -----------------------------------------------------------------


def test_language_mapping(fake):
    bundle = fake.load()
    for lang, text in (("eng", ENG_TEXT), ("fil", FIL_TEXT), ("ceb", CEB_TEXT)):
        assert tts_cpp.synthesize_turn(bundle, lang, text, lang, set()).ok

    by_voice = {}
    for r in _speech(fake):
        by_voice.setdefault(r["voice"], set()).add(r.get("language", "<omitted>"))
    assert by_voice == {"lingkod-eng": {"english"}, "lingkod-fil": {"<omitted>"}, "lingkod-ceb": {"<omitted>"}}


def test_segmented_turn_order_pauses_and_audio(fake):
    bundle = fake.load()

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok and r.sr == 24000 and r.mode == "segmented"
    assert [s["text"] for s in r.segments] == [r_["input"] for r_ in _speech(fake)]
    frames = [round(0.84 * s["chars"]) for s in r.segments]
    assert [s["frames"] for s in r.segments] == frames
    expected_len = sum(f * 1920 + int(s["pause_after_s"] * 24000) for f, s in zip(frames, r.segments))
    assert len(r.audio) == expected_len
    assert [len(a) for a in r.segment_audio] == [f * 1920 for f in frames]
    assert all(s["retries"] == 0 and s["cap_hit"] is False for s in r.segments)
    assert [q["max_new_tokens"] for q in _speech(fake)] == [s["cap"] for s in r.segments]
    assert r.timeline["n_stalls"] == 0 and r.rtf > 0


def test_cap_hit_retries_once_with_a_new_seed(fake):
    bundle = fake.load({"runaway_on": {"application form": 1}})

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok
    seg = r.segments[1]
    assert seg["retries"] == 1
    first, second = seg["attempts"]
    assert first["cap_hit"] is True and first["frames"] == seg["cap"]
    assert second["cap_hit"] is False and second["seed"] != first["seed"]
    assert seg["frames"] == second["frames"]


def test_second_cap_hit_makes_the_whole_turn_text_only(fake):
    bundle = fake.load({"runaway_on": {"application form": -1}})

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert not r.ok and r.audio is None and r.timeline is None
    assert "segment 1" in r.failure_reason and "text-only" in r.failure_reason
    assert [len(s["attempts"]) for s in r.segments] == [1, 2, 0]  # stops at the failing segment
    assert len(_speech(fake)) == 3


def test_zero_frames_counts_as_a_failed_attempt(fake):
    bundle = fake.load({"zero_on": {"Bring": 1}})

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok and r.segments[0]["retries"] == 1 and r.segments[0]["attempts"][0]["frames"] == 0


def test_parity_records_a_cap_hit_without_retrying(fake):
    bundle = fake.load({"runaway_on": {"Bring": 1}}, mode="parity")

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok and len(r.segments) == 1
    assert r.segments[0]["cap_hit"] is True and r.segments[0]["frames"] == 4096
    assert r.segments[0]["retries"] == 0
    assert [q["max_new_tokens"] for q in _speech(fake)] == [4096]


def test_parity_retries_at_2048_on_out_of_memory(fake):
    bundle = fake.load({"oom_on": ["Bring"]}, mode="parity")

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok
    assert [q["max_new_tokens"] for q in _speech(fake)] == [4096, 2048]
    assert "malloc failed" in r.segments[0]["attempts"][0]["error"]


def test_server_error_is_text_only_and_the_server_keeps_serving(fake):
    bundle = fake.load({"error_on": ["application form"]})

    bad = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())
    good = tts_cpp.synthesize_turn(bundle, 1, FIL_TEXT, "fil", set())

    assert not bad.ok and "HTTP 502" in bad.failure_reason
    assert good.ok and bundle.restarts == 0


def test_crash_is_text_only_and_the_next_turn_restarts_the_server(fake):
    bundle = fake.load({"crash_on": ["application form"]})

    bad = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())
    assert not bad.ok and "connection failed" in bad.failure_reason
    assert not bundle.server.is_alive()

    good = tts_cpp.synthesize_turn(bundle, 1, FIL_TEXT, "fil", set())
    assert good.ok and good.server_restarts == 1 and bundle.restarts == 1
    assert len(fake.requests("/v1/audio/voices")) == 6  # voices registered again
    assert len(bundle.info["restarts"]) == 1


def test_timeout_kills_the_server_and_the_turn_is_text_only(fake):
    bundle = fake.load({"hang_on": ["application form"]}, request_timeout_s=2)

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert not r.ok and "did not answer" in r.failure_reason
    assert not bundle.server.is_alive()


def test_frontend_raise_is_text_only(fake, monkeypatch):
    bundle = fake.load()

    def boom(*a, **k):
        raise ValueError("frontend refused this row")

    monkeypatch.setattr(tts_cpp, "text_segments", boom)
    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert not r.ok and r.failure_reason == "frontend refused this row"
    assert _speech(fake) == []


def test_unknown_lang_is_text_only(fake):
    bundle = fake.load()
    r = tts_cpp.synthesize_turn(bundle, 0, "text", "xyz", set())
    assert not r.ok and "Unknown lang" in r.failure_reason


def test_seeds_reproduce_across_runs(fake):
    texts = []
    for _ in range(2):
        bundle = fake.load(seed=99)
        tts_cpp.synthesize_turn(bundle, 5, ENG_TEXT, "eng", set())
        bundle.unload()
    seeds = [r["seed"] for r in _speech(fake)]
    assert seeds[:3] == seeds[3:] and len(set(seeds[:3])) == 3


def test_synthesize_raises_where_synthesize_turn_reports(fake):
    bundle = fake.load({"error_on": ["Bring"]})
    with pytest.raises(RuntimeError, match="HTTP 502"):
        tts_cpp.synthesize(bundle, ENG_TEXT, "eng", set())


def test_segmented_drops_a_parenthetical_acronym_exactly_when_parity_does():
    # "(PRC)" sits in a different segment from the bare "PRC", so a per-segment
    # strip alone would keep it; the whole-answer strip drops it, as parity does.
    text = "Mag-file sa Professional Regulation Commission (PRC). Pumunta sa PRC office bukas."
    seg = " ".join(t for t, _ in tts_cpp.text_segments(text, "fil", {"PRC"}, "segmented"))
    par = " ".join(t for t, _ in tts_cpp.text_segments(text, "fil", {"PRC"}, "parity"))

    assert "(" not in seg and "(" not in par
    assert seg.split() == par.split()


# --- lead-in trim through the server --------------------------------------

# The fake answers with 3 silent frames, then its tone: 5760 samples of
# silence, of which 20 ms (480 samples) are kept.
LEAD_CUT = 3 * 1920 - 480


def test_lead_in_is_trimmed_from_every_segment(fake):
    bundle = fake.load({"lead_silence_frames": 3})

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok and bundle.info["trim_lead_in"] is True and len(r.segments) > 1
    frames = [round(0.84 * s["chars"]) for s in r.segments]
    assert [s["frames"] for s in r.segments] == frames  # generated frames, counted before the trim
    assert all(s["lead_in_trimmed_s"] == pytest.approx(LEAD_CUT / 24000) for s in r.segments)
    assert [len(a) for a in r.segment_audio] == [f * 1920 - LEAD_CUT for f in frames]
    assert [s["audio_s"] for s in r.segments] == pytest.approx([(f * 1920 - LEAD_CUT) / 24000 for f in frames])
    assert len(r.audio) == sum(f * 1920 - LEAD_CUT + int(s["pause_after_s"] * 24000)
                               for f, s in zip(frames, r.segments))


def test_parity_ceb_pause_is_the_frontend_pause_plus_20_ms(fake):
    bundle = fake.load({"lead_silence_frames": 3}, mode="parity")

    r = tts_cpp.synthesize_turn(bundle, 0, CEB_TEXT, "ceb", set())

    assert r.ok and len(r.segments) == 2
    end = len(r.segment_audio[0])
    silence = int(r.segments[0]["pause_after_s"] * 24000) + 480
    assert r.segments[0]["pause_after_s"] > 0
    assert not r.audio[end:end + silence].any() and r.audio[end + silence:end + silence + 240].any()


def test_trim_lead_in_false_keeps_the_server_audio(fake):
    bundle = fake.load({"lead_silence_frames": 3}, mode="parity", trim_lead_in=False)

    r = tts_cpp.synthesize_turn(bundle, 0, ENG_TEXT, "eng", set())

    assert r.ok and bundle.info["trim_lead_in"] is False
    assert r.segments[0]["lead_in_trimmed_s"] == 0.0
    assert len(r.audio) == r.segments[0]["frames"] * 1920
    assert not r.audio[:3 * 1920].any()
