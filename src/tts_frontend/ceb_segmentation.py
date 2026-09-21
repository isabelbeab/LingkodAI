"""
ceb_segmentation.py

Ported 2026-08-13 from NB_DEMO_ceb_prc_frontend.ipynb (2026-08-11 build,
segmentation cell), same file verbalize_ceb_prc came from. Same status:
sandbox, not promoted to lingkod_tts_utils.py, not wired into NB1/NB3.

WHY THIS MATTERS BEYOND LIST MARKERS: normalize_inline_list_markers fixes
the mid-string numbered-list problem found in the 2026-08-13 coverage
categorization pass (roughly 40 of the 553 corpus rows have a checklist
like "1. Item... 2. Item... 3. Item..." typed inline with no real
newlines -- the same shape as the original sentence-4 bug from
2026-08-11, just not yet applied corpus-wide). Only the FIRST "1." at a
string's true start was ever visible to verbalize_ceb_prc's own
list-marker handling; every later "2.", "3." etc mid-string fell through
to the bare-cardinal pass as a stray digit.

Checked 2026-08-13: this was a coverage gap, not a silent-correctness bug
like the reference-code case -- every mid-string-list row in the current
corpus happens to contain at least one item numbered 8+, which always
raises somewhere and blocks the row, just under a misleading reason.
Still worth fixing for coverage, and for a second reason: this same cell
is also the pause-insertion and RTF-per-segment machinery needed for the
formal latency reporting requirement, so porting it serves both.

KNOWN, INTENTIONAL BEHAVIOR NOT CHANGED BY THIS PORT: split_for_synthesis
only speaks an ordinal word for list items 1-3 (CEB_ORDINALS' coverage).
For item 4 and beyond, the "N. " marker is stripped but no ordinal word
is substituted -- the item's sequence number is conveyed only by the
pause between segments (PAUSE_LIST_ITEM), not spoken aloud. This was
already the demo notebook's own design choice, not something introduced
by this port; flagged here so it isn't mistaken for an oversight.
"""

import re, time
import numpy as np
# NOTE 2026-08-13: torch is imported INSIDE synthesize_segments below, not
# here at module level. First fix (module-level import) was itself wrong:
# it made this whole module fail to import in any environment without
# torch installed, breaking text-only work (list-marker splitting, pause
# calculation) that never touches the GPU at all -- caught immediately
# when trying to test real corpus text conversion without a GPU present.
# A lazy, function-local import keeps split_for_synthesis usable
# everywhere while still fixing the original NameError on JOJIE.

PAUSE_CLAUSE    = 0.20
PAUSE_SENTENCE  = 0.45
PAUSE_LIST_ITEM = 0.60
MAX_SEG_CHARS   = 140

_MD_BOLD    = re.compile(r"\*{1,2}")
_NUM_MARKER = re.compile(r"^\s*(\d+)\.\s+")
_BULLET     = re.compile(r"^\s*[-*\u2022]\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE     = re.compile(r"(?<=[,;:])\s+")

def clean_rag_answer(text: str) -> str:
    """Strip formatting artifacts that came from the RAG layer, not from
    language. Runs BEFORE the verbalizer / frontend passes, never after."""
    text = _MD_BOLD.sub("", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

_INLINE_LIST_MARKER = re.compile(r"(:\s*|\.\s+)(?=\d{1,2}\.\s+[A-Z])")

def normalize_inline_list_markers(text: str) -> str:
    """Insert a real newline before a "N. " list marker that appears
    mid-string after a colon or a prior list item's closing period, so
    split_for_synthesis's block-start detection actually sees it.

    Why this exists: found 2026-08-11 testing sentence 4 of the CEB demo
    set. Tim's translation has no real newlines (that's genuinely how he
    typed it back, not a transcription error), so "...dokumento: 1. Usa
    ka sulat..." arrived as one continuous string. Without this pass,
    split_for_synthesis's sentence-splitter treats "1." as its own
    sentence, stranding the numeral, and verbalize_ceb_prc's bare-cardinal
    fallback then SILENTLY converts the orphaned "1" to the counting word
    "usa" instead of the ordinal "una" -- wrong reading, no warning. This
    is a heuristic (colon-or-period immediately before a small
    digit-dot-capital pattern reads as list-start), not a general list
    parser; the (?!\s*\.) guard on _BARE_DIGIT_RUN in verbalize_ceb_prc
    is the second line of defense that raises rather than guesses wrong
    if something unusual still slips past this.
    """
    return _INLINE_LIST_MARKER.sub(lambda m: m.group(1).rstrip() + "\n", text)


def split_for_synthesis(text: str, max_chars: int = MAX_SEG_CHARS,
                        ordinal_words: dict | None = None):
    """Return [(segment_text, pause_after_seconds), ...]."""
    segments = []
    for block in re.split(r"\n\s*\n|\n", normalize_inline_list_markers(clean_rag_answer(text))):
        block = block.strip()
        if not block:
            continue

        is_item = False
        m = _NUM_MARKER.match(block)
        if m:
            is_item = True
            n = int(m.group(1))
            block = _NUM_MARKER.sub("", block)
            if ordinal_words and n in ordinal_words:
                block = f"{ordinal_words[n]}, {block}"
        elif _BULLET.match(block):
            is_item = True
            block = _BULLET.sub("", block)

        sents = [s for s in _SENT_SPLIT.split(block) if s.strip()]
        for si, sent in enumerate(sents):
            pieces = [sent]
            if len(sent) > max_chars:
                pieces, buf = [], ""
                for cl in _CLAUSE.split(sent):
                    if buf and len(buf) + len(cl) > max_chars:
                        pieces.append(buf.strip()); buf = cl
                    else:
                        buf = f"{buf} {cl}".strip()
                if buf:
                    pieces.append(buf.strip())

            for pi, piece in enumerate(pieces):
                last_piece = (pi == len(pieces) - 1)
                last_sent  = (si == len(sents) - 1)
                if not last_piece:
                    pause = PAUSE_CLAUSE
                elif last_sent and is_item:
                    pause = PAUSE_LIST_ITEM
                else:
                    pause = PAUSE_SENTENCE
                segments.append((piece.strip(), pause))
    return segments


def synthesize_segments(segments, model, tokenizer, sampling_rate,
                        preprocess=None, device="cuda", voice_seed=None):
    """Yield (waveform, pause_after, timing) per segment. `preprocess` is
    applied to each segment's text before tokenization -- pass
    prc_preprocess (defined below) to run verbalize_ceb_prc then
    ceb_frontend_apply. Timing wraps the forward pass only.

    voice_seed: UNTESTED, hypothesis-tier, ADDED 2026-08-13 in response to
    HK reporting a voice shift (deep vs higher-pitched) between segments
    of the same synthesized row. If set to an int, the seed is reset to
    the SAME value immediately before EVERY segment's forward pass, not
    once before the loop. This distinction matters and was wrong in an
    earlier explanation: seeding once before the loop only makes an
    entire rerun reproducible, since each call still advances the RNG
    stream, so segment 2 draws different random state than segment 1
    regardless. Resetting per segment is what actually gives every
    segment the same starting point.

    THIS IS A HYPOTHESIS, NOT A CONFIRMED FIX. It rests on the assumption
    that VITS's stochastic decoder (which samples noise at inference,
    even in eval mode, by design) is the source of the voice variation.
    That is a reasonable read of the architecture but has not been
    verified against this specific checkpoint or this specific symptom --
    no GPU access to test it directly. There's a real alternative
    explanation seeding would NOT fix: if this checkpoint is genuinely
    multi-speaker (not the single-speaker model the project has assumed)
    and something is randomly selecting a speaker embedding via a
    non-torch random source, this seed would do nothing. Worth checking
    model.config for anything indicating multiple speakers before
    trusting this is the whole answer -- see the diagnostic snippet in
    chat. Report back whether this actually helps; if it doesn't, that
    result is as informative as if it does."""
    import torch  # lazy: only this function needs it, see module header
    for text, pause in segments:
        if voice_seed is not None:
            torch.manual_seed(voice_seed)
            torch.cuda.manual_seed_all(voice_seed)
        prepared = preprocess(text) if preprocess else text
        inputs = tokenizer(prepared, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = model(**inputs).waveform[0].detach().cpu().numpy()
        t1 = time.perf_counter()
        audio_seconds = len(wav) / sampling_rate
        wall_seconds = t1 - t0
        timing = {
            "text": text, "prepared": prepared,
            "wall_seconds": wall_seconds, "audio_seconds": audio_seconds,
            "rtf": (wall_seconds / audio_seconds) if audio_seconds > 0 else float("nan"),
        }
        yield wav, pause, timing


def concat_segments(gen, sampling_rate):
    out, timings = [], []
    for wav, pause, timing in gen:
        out.append(wav)
        timings.append(timing)
        if pause > 0:
            out.append(np.zeros(int(pause * sampling_rate), dtype=wav.dtype))
    audio = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return audio, timings
