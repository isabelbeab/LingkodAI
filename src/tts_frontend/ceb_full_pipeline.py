"""
ceb_full_pipeline.py

Combines, IN ORDER:
1. ceb_segmentation.split_for_synthesis -- splits a row's text into
   segments, list-marker-aware, so mid-string numbered checklists (found
   2026-08-13, ~41 corpus rows) get proper ordinal/pause treatment instead
   of falling through as stray digits.
2. ceb_prc_digit_verbalizer.verbalize_ceb_prc -- per segment. MUST run
   before step 4, never after -- see that module's header for the PHP/
   initialism-spellout collision this order avoids (confirmed 2026-08-13
   against 100 real PHP-bearing rows: wrong order clears 0/100, right
   order clears 52/100).
3. ceb_respell_lab.apply_loanword_fixes -- per segment. MUST run BEFORE
   step 4, not after. REORDERED 2026-08-13: originally ran last, which
   was a real bug, not a style choice -- by the time
   shared_domain_frontend's -tion/x_to_ks rules and initialism spellout
   have run, a word like "exchange" is already "ekschange" and "SEC" is
   already "es-i-si", so a lexicon lookup for the ORIGINAL word never
   matches. Caught by testing a sentence containing both "exchange" and
   "SEC" against real confirmed lexicon entries and seeing the wrong
   output. Running this step first means a confirmed lexicon entry
   genuinely overrides the general rules and the initialism algorithm,
   matching what ceb_respell_lab.py's own Layer A/Layer B docstring always
   claimed, but which the pipeline itself did not actually enforce until
   now.
4. shared_domain_frontend.apply_shared_frontend -- per segment.

Sandbox, same status as its four components. Not promoted to
lingkod_tts_utils.py, not wired into NB1/NB3.
"""
import ceb_segmentation as seg
import ceb_prc_digit_verbalizer as prc
import shared_domain_frontend as sdf
import ceb_respell_lab as crl


def full_ceb_pipeline_segments(text: str, initialism_set: set,
                               demo_fallback: bool = False) -> list[tuple[str, float]]:
    """Returns [(processed_segment_text, pause_after_seconds), ...], the
    shape synthesize_segments expects. Raises on the FIRST segment that
    fails either digit verbalization or the shared frontend -- same
    fail-fast contract as the rest of this project's raise-by-default
    design, so a coverage check still gets one clear ValueError per row.

    demo_fallback: passed straight through to
    ceb_respell_lab.apply_loanword_fixes. False (default) is the tested,
    regression-verified path (399/553 real corpus rows). True substitutes
    a best-guess for the handful of words with no confirmed heard-ok
    entry yet, instead of leaving them as raw English -- for demo use
    under deadline pressure, not for anything being reported as a result."""
    out = []
    for seg_text, pause in seg.split_for_synthesis(text, ordinal_words=prc.CEB_ORDINALS):
        t = prc.verbalize_ceb_prc(seg_text)
        t = crl.apply_loanword_fixes(t, demo_fallback=demo_fallback)
        t = sdf.apply_shared_frontend(t, "ceb", initialism_set)["output"]
        out.append((t, pause))
    return out


def full_ceb_pipeline(text: str, initialism_set: set, demo_fallback: bool = False) -> str:
    """Convenience wrapper: joined text only, no pause timing. See
    full_ceb_pipeline_segments for the demo_fallback explanation. Real
    synthesis should call full_ceb_pipeline_segments directly and feed it
    to synthesize_segments / concat_segments so pauses actually get
    inserted -- this wrapper is for quick coverage/text checks only."""
    segments = full_ceb_pipeline_segments(text, initialism_set, demo_fallback=demo_fallback)
    return " ".join(t for t, _pause in segments)
