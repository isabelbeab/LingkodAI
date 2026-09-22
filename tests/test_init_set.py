"""Asserts the frozen TTS initialism set's count and a few known members, so
a regenerated set (accidental or deliberate) cannot silently change what
synthesis spells out. See scripts/build_init_set.py's docstring for why this
set must never be derived at runtime, per answer.

If this test ever fails because the count or a member genuinely changed
(e.g. build_init_set.py was rerun against a new source file), that is a real
change to synthesis behavior and must be re-listened to before updating this
test, not just made to pass."""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load(name: str) -> dict:
    path = DATA_DIR / name
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_init_set_ds6_exists_and_has_expected_count():
    d = _load("init_set_ds6.json")
    assert d["label"] == "ds6"
    assert d["n_tokens"] == 231
    assert len(d["tokens"]) == 231


def test_init_set_ds6_contains_known_members():
    d = _load("init_set_ds6.json")
    tokens = set(d["tokens"])
    # PRC and COR are core domain acronyms that must be in any real corpus
    # sample; CPA and PIC are the specific plural-acronym cases
    # scripts/build_init_set.py's own docstring uses to explain why
    # per-answer derivation is unsafe (see its "Plural-acronym check").
    for expected in ("PRC", "COR", "CPA", "PIC"):
        assert expected in tokens, f"{expected!r} missing from the frozen DS6 initialism set"


def test_init_set_ds6_excludes_us_on_purpose():
    """A known quirk, preserved on purpose: the notebook's STOP list removed
    'US' after finding it was always the country abbreviation in the corpus.
    See scripts/build_init_set.py's module docstring."""
    d = _load("init_set_ds6.json")
    assert "US" not in set(d["tokens"])
    assert "US" in set(d["stop_list"])


def test_init_set_ds6_provenance_recorded():
    d = _load("init_set_ds6.json")
    assert d["source_rows"] == 2235
    assert d["source_sha256"]
    assert d["text_column"] == "rag_answer_native"


def test_init_set_live_is_superset_of_ds6_and_ds7():
    ds6 = set(_load("init_set_ds6.json")["tokens"])
    ds7 = set(_load("init_set_ds7.json")["tokens"])
    live = set(_load("init_set_live.json")["tokens"])
    assert ds6 <= live
    assert ds7 <= live
    assert live == ds6 | ds7
