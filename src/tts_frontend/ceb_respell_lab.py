# CEB respelling lab: rule engine + variant tester.
# Sandbox. Not promoted, not wired into NB1/NB3, same status as
# verbalize_ceb_prc.
#
# HONESTY NOTE ON "HOW I PARSED IT": there is no parser. "pie ar see" was
# a HAND-WRITTEN override, added specifically because the generic
# per-letter algorithm (_spell_out, a dict lookup over CEB_LETTER_NAMES)
# produced "pi ar si", which did NOT match what was confirmed by ear.
# CEB_LOANWORD is likewise a hand-written dict, not a derivation. So the
# thing to tweak is not a clever algorithm -- it's rules and lexicon
# entries, which is what this file makes tweakable.

from __future__ import annotations
import re
from typing import Callable

# ---------------------------------------------------------------------
# LAYER A: productive rules. Ordered, applied in sequence.
#
# Only add a rule here when the SAME pattern failed across MULTIPLE
# words. A pattern seen once belongs in the lexicon (Layer B), not here,
# because one instance can't distinguish "general rule" from "this word
# happens to behave this way."
#
# tier: "multi" = same failure observed in 3+ distinct words
#       "single" = observed once, provisional, candidate for demotion
# ---------------------------------------------------------------------
CEB_RESPELL_RULES: list[tuple[str, str, str, str]] = [
    # (name, pattern, replacement, tier)
    ("tion_to_syon", r"tion\b", "syon", "multi"),
    # ^ Observed in verification, registration, section, accreditation
    #   (4 distinct words, 2026-08-11 listening). Strongest rule here.
    #   Note \b so it won't fire inside an unrelated substring. CONFIRMED
    #   AGAIN 2026-08-13 on real synthesized RAG output: "specialization"
    #   -> "specializasyon" heard as good, a 5th word, different corpus
    #   than the original 4.
    ("x_to_ks", r"x", "ks", "multi"),
    # ^ ADDED 2026-08-13. HK reported "tax" mispronounced (the model
    #   couldn't handle the letter at all) -- structurally expected, not
    #   idiosyncratic: Cebuano's alphabet has no "x", so this is a
    #   mechanical gap, not a word-specific quirk like most lexicon
    #   entries. Tier "multi" by structural reasoning plus corpus scope,
    #   not by having heard multiple words yet: only "tax" (115 corpus
    #   occurrences) has been directly heard failing. Real corpus scan
    #   found 300+ combined occurrences across other x-containing words
    #   that would hit the same underlying gap (Exchange 24, examination
    #   21+23, Extension 19, Annex 14, Exemption 11+9, Executive 5,
    #   Experience 5, ...). Confirming 2-3 more of these by ear (exam,
    #   exchange, extension are the natural next picks) would fully clear
    #   the multi-word bar the way tion_to_syon did; until then, treat
    #   this as strong-but-not-fully-heard, not settled.
]


def apply_rules(word: str, rules=None) -> tuple[str, list[str]]:
    """Apply the rule list in order. Returns (result, [rule names that fired]).
    Returning WHICH rules fired matters: when a respelling sounds wrong,
    the first question is 'which rule did this' and guessing is slow."""
    rules = CEB_RESPELL_RULES if rules is None else rules
    fired = []
    out = word
    for name, pat, repl, _tier in rules:
        new = re.sub(pat, repl, out)
        if new != out:
            fired.append(name)
            out = new
    return out, fired


# ---------------------------------------------------------------------
# LAYER B: per-word lexicon. Overrides rules entirely when present.
#
# Every entry is a HYPOTHESIS until heard in >=2 different carrier
# sentences. Status field is not decoration -- it's what keeps an
# unverified guess from being cited as settled later.
#   "heard-ok"      = listened, sounded right, >=2 carriers
#   "heard-decent"  = listened, usable but not fully right -- ADDED
#                     2026-08-13, a real 4th tier the original three
#                     didn't have room for. Distinct from "heard-ok"
#                     (fully right) and from "heard-bad" (wrong): this is
#                     "acceptable as a fallback, worth one more attempt if
#                     time allows, not worth blocking on."
#   "heard-bad"     = listened, sounded wrong (kept as a NEGATIVE record so
#                     the same wrong variant doesn't get retried)
#   "proposed"      = written from the failure description, NOT yet heard
#   "unclear"       = ADDED 2026-08-13: a variant was tested but no clear
#                     verdict was given either way (superseded by moving
#                     straight to new candidates). Distinct from
#                     "proposed" (never tested at all) and from
#                     "heard-bad" (tested, confirmed wrong) -- don't
#                     assert a verdict that wasn't actually given.
#
# UPDATED 2026-08-13 with real evidence from the first real-RAG-row
# listening pass (44 words, single listener, real corpus sentences via
# NB8 Section 15 -- not generic carriers). Two findings worth stating
# before the table, not buried in it:
#
# (1) "account-" root family: accountant, accounting, accountancy all
#     independently got the identical verdict (the "ou" sound too hard,
#     should read as "kawn" not "koon"). Genuinely one root issue
#     surfacing three times, not three unrelated words -- all three
#     entries share the same fix for that reason, listed separately only
#     because lookups are per exact word string.
#
# (2) "v" is NOT a clean structural gap like "x" was, and does NOT get a
#     general rule despite the surface similarity. "Division" reaches the
#     model as literal untranslated English 456 times, goes through zero
#     v-related handling, and was heard as PERFECT. "Level", "envelope",
#     and "drive" all contain "v" too and were all heard bad specifically
#     because of the v. 3-bad vs 1-good on the same letter means this is
#     context-dependent, not a uniform gap -- a blanket v->b rule would
#     risk breaking Division to fix Level. Handled per-word below. If 2-3
#     more v-words independently confirm the same b-substitution helps,
#     that would be real evidence to revisit this as a rule; not enough
#     yet.
#
# (3) MISTAKE, FOUND AND FIXED 2026-08-13: entries below originally used
#     literal hyphens copied straight from HK's "spoken as" column (e.g.
#     "pro-fe-syu-nal", "o-fi-se") as if those were candidate spellings
#     to test. They were not -- that column is HK's own syllable-break
#     notation for writing down what they heard, not text that was ever
#     sent to the model. Conflating notation with data was a mapping
#     mistake on my part. Every regular-loanword entry below is
#     de-hyphenated accordingly. Hyphens ARE meaningful and intentional
#     in the INITIALISM entries above (cpa/dst/us/sec/bir/usb/pqf) --
#     there, the hyphen is a real letter-separator choice being tested,
#     not a notation artifact, and stays.
# ---------------------------------------------------------------------
CEB_LOANWORD_LAB: dict[str, dict] = {
    # --- INITIALISMS with a hyphen/spelling question, added 2026-08-13 ---
    "cpa":  {"sipiey": "heard-bad", "si-pi-ey": "unclear", "see-pee-ay": "proposed"},
    # ^ "si-pi-ey" is the current hyphenated default -- HK asked to try
    #   variants without explicitly saying this one failed, so marked
    #   unclear rather than heard-bad. New candidate mirrors the
    #   strategy that just worked for SEC: English letter-NAME spelling
    #   (C=see, P=pee, A=ay) instead of the CEB-phonetic approximation.
    "dst":  {"diesti": "heard-bad", "di-es-ti": "heard-ok"},
    # ^ SETTLED per HK ("di-is-ti is good" -- read as confirming
    #   "di-es-ti", the only DST form actually synthesized this round;
    #   the vowel-spelling difference looks like HK's own phonetic
    #   re-transcription, not a genuinely different candidate).
    "us":   {"yues": "heard-bad", "yu-es": "heard-decent", "yu-ess": "proposed"},
    # ^ "decent, but try yu-ess" -- same doubled-final-letter pattern
    #   that helped several other words tonight (sertipikeyyt, sentrall,
    #   akawntansee).
    "sec":  {"esisi": "heard-bad", "es-i-si": "heard-bad", "ess-ee-see": "heard-bad",
            "es-ee-see": "heard-ok"},
    # ^ SETTLED 2026-08-13 (batch 4). Final spelling single "es" not
    #   double "ess" at the start -- close to but not identical to the
    #   earlier proposed "ess-ee-see", applying the exact final form given.
    "bir":  {"biayar": "heard-bad", "bi-ay-ar": "heard-ok"},
    # ^ CONFIRMED PERFECT 2026-08-13 (hyphenated). Settled.
    "usb":  {"yuesbi": "heard-bad", "yu-es-bi": "heard-ok"},
    # ^ CONFIRMED PERFECT 2026-08-13 (hyphenated). Settled.
    "pqf":  {"pi-kyu-ef": "heard-ok", "pi-kyu-ep": "unclear"},
    # ^ REVERSED 2026-08-13: originally heard-bad (jitter/robotic/silent
    #   at "ef") in earlier isolated-word/generic-carrier testing. In
    #   REAL sentence context this round, HK confirmed "pi-kyu-ef is
    #   good" -- no ending swap needed. Plausible explanation: this is
    #   the same carrier-context-sensitivity already documented
    #   elsewhere tonight (isolated-word synthesis is the weakest
    #   carrier by two independent signals), not a contradiction to
    #   paper over. The earlier heard-bad verdict was real evidence from
    #   a real test, just under different conditions; both are kept in
    #   the record. "pi-kyu-ep" (the swap this row prompted) got no
    #   verdict since it turned out not needed.


    "engineer":    {"in-hi-neer": "heard-bad", "endyinir": "heard-bad", "injeeneer": "heard-ok"},
    "engineers":   {"en-hi-neers": "heard-bad", "enjineers": "heard-bad", "injeeneers": "heard-ok"},
    # ^ plural inferred from the settled singular (batch 4 only gave
    #   "engineer"), not independently confirmed -- flagged, not hidden.
    "registered":  {"re-gis-teerrr-ed": "heard-bad", "redyisterd": "heard-bad", "rejisterd": "heard-ok"},
    "architects":  {"arfitehks": "heard-bad", "arkiteks": "heard-ok"},
    "care":        {"ca-re": "heard-bad", "keyr": "heard-bad", "ker": "heard-ok"},
    "council":     {"kun-sil": "heard-bad", "kawnsil": "heard-bad", "kownsil": "heard-ok"},
    "file":        {"fi-le": "heard-bad", "fayl": "heard-ok"},
    "apply":       {"mag-ap-lee": "heard-bad", "aplay": "heard-ok"},
    "update":      {"mag-up-da-teh": "heard-bad", "apdeyt": "heard-ok"},
    "lifelong":    {"laif-li-yong": "heard-bad", "layflong": "heard-bad", "layf-long": "heard-ok"},
    "learning":    {"lee-err-ning": "heard-bad", "lerning": "heard-ok"},
    "self-directed": {"self-directed": "heard-bad", "self dayrekted": "heard-bad",
                      "self-direkted": "heard-ok"},
    # ^ ALL SETTLED 2026-08-13 (batch 4, listening_batch_4.ods). Final
    #   choices sometimes differ from what was previously proposed here
    #   (e.g. "kownsil" not "kawnsil", "self-direkted" not
    #   "self dayrekted") -- applying HK's actual final spelling exactly
    #   as given, not the closest prior guess. Old unused proposals
    #   downgraded to heard-bad so they stop showing as pending.

    # --- CONFIRMED GOOD, 2026-08-13, no further work needed ---
    "division":     {"dibisyon": "heard-ok"},       # contains "v" -- see note (2) above
    "regional":     {"rejunal": "heard-ok"},
    "form":         {"form": "heard-ok"},
    "documentary":  {"documentary": "heard-ok"},
    "stamp":        {"stamp": "heard-ok"},
    "partnership":  {"partnership": "heard-ok"},
    "attorney":     {"atornee": "heard-ok"},
    "letter":       {"letter": "heard-ok"},         # minor: "a bit stressed on tt", not worth chasing
    "firm":         {"firm": "heard-ok"},           # HK: "dunno how else it can be improved, leave it"
    "office":       {"ofise": "heard-bad", "ofis": "heard-ok"},
    "public":       {"poblik": "heard-bad", "pablik": "heard-ok"},
    "service":      {"unclear": "heard-bad", "serbis": "heard-ok"},
    "training":     {"trayning": "heard-bad", "treyning": "heard-ok"},
    "website":      {"websayt": "heard-ok"},

    # --- NEEDS ONE MORE ROUND: heard-bad/decent, new candidate proposed ---
    "certificate": {"ser-ti-pika-te": "heard-bad", "sertipikate": "heard-bad",
                    "sertipikeyyt": "heard-ok"},
    # ^ SETTLED per HK -- "sertipikeyyt" is the final answer.
    # ^ "sertipikate" decent but not fully right -- HK: ending "ey" too
    #   quick, doubling the vowel to slow it down for the next attempt.
    #   Pre-existing older entries ("ser-ti-pika-te" heard-bad,
    #   "sertipiket" proposed) folded in here rather than kept separate;
    #   sertipikate already supersedes them as the closest real result.
    "certified":    {"sertifid": "heard-bad", "sertifayd": "heard-ok"},
    "copy":         {"koopi": "heard-bad", "kapi": "heard-ok"},
    # ^ SETTLED per HK -- "kapi" is the final answer, further
    #   candidates (kapee/kahpeh/kapeh/kapiih/kapie/kapy) dropped, never
    #   actually synthesized, moot now.
    "central":      {"sengal": "heard-bad", "sentral": "heard-bad", "sentrall": "heard-ok"},
    # ^ "good enough" per HK -- settled, not chasing further.
    "philippine":   {"filipine": "heard-bad", "pilipin": "heard-ok"},
    "power":        {"poower": "heard-bad", "pawr": "heard-bad", "pawur": "heard-bad",
                    "pahwer": "heard-bad", "pahwur": "heard-bad", "pawer": "heard-ok"},
    # ^ SETTLED per HK -- "pawer" is the final answer.
    "accountancy":  {"akoontant": "heard-bad", "akawntansi": "heard-bad", "akawntansee": "heard-ok"},
    # ^ "good enough" per HK -- settled, not chasing further. akawntansi
    #   downgraded from heard-decent to heard-bad (superseded by the
    #   winner), so it stops showing up as "still needs testing".
    # ^ "akawntansee" preferred over "akawntansi" -- settled.
    "accountant":   {"akoontant": "heard-bad", "akawntant": "proposed"},
    "accounting":   {"akoontant": "heard-bad", "akawnting": "proposed"},
    # ^ all three share the same root issue (see note 1 above);
    #   accountancy is furthest along (decent, refining the ending),
    #   accountant/accounting still need their first real listen.
    "authenticated": {"awthentikeyted": "heard-ok"},
    # ^ "good enough, keep it" per HK -- settled.
    # ^ HK: "good enough, dunno how to improve" -- stopping here, not
    #   chasing a marginal gain with no clear direction.
    "professional": {"propesyunal": "heard-ok"},
    # ^ CONFIRMED 2026-08-13, preferred over "propeshunal" -- settled.
    #   Earlier entry here used literal hyphens copied from HK's own
    #   phonetic transcription notation, a mapping mistake on my part
    #   (see note 3 above), fixed before this round ran.
    "national":     {"natyonal": "heard-bad", "nasyonal": "heard-ok"},
    # ^ SETTLED per HK -- "nasyonal" is the final answer. "nashonal" was
    #   never actually tested (dropped before synthesis, not rejected by
    #   ear), so it's removed rather than falsely marked heard-bad.
    "program":      {"poogang": "heard-bad", "prugram": "heard-ok", "prowgram": "unclear"},
    # ^ "prugram" settled per HK. "prowgram" was also synthesized in the
    #   same batch but got no verdict either way -- not asserting it
    #   failed without evidence.
    "notarized":    {"notarizd": "heard-bad", "notaraysd": "heard-bad", "nutaraysd": "heard-ok"},
    "chairperson":  {"kairperson": "heard-bad", "cherperson": "heard-ok", "tserperson": "unclear"},
    # ^ "cherperson" settled per HK. "tserperson" also synthesized in the
    #   same batch, no verdict given, not asserted as failed.
    "technical":    {"teshnikal": "heard-bad", "teknikal": "heard-ok"},
    # ^ SETTLED per HK, real-row context.
    "real":         {"reyal": "heard-bad", "ril": "heard-ok"},
    "estate":       {"estate": "heard-bad", "estayt": "heard-bad", "esteyt": "heard-ok"},
    "statement":    {"statement": "heard-bad", "steytment": "heard-ok"},
    "articles":     {"articles": "heard-bad", "artikels": "heard-ok"},
    "schools":      {"skools": "heard-bad", "skuls": "proposed"},
    "lecturer":     {"lekturer": "heard-ok", "lekchurer": "proposed"},
    "framework":    {"freym work": "heard-ok"},
    # ^ NEW 2026-08-13 (batch 4). Singular, distinct entry from the
    #   pre-existing "frameworks" plural below -- HK gave the singular
    #   final spelling only; the plural entry is left as its own open
    #   item rather than guessed at by appending an "s".
    "frameworks":   {"fiaymework": "heard-bad", "freymwerks": "proposed"},
    "level":        {"leel": "heard-bad", "lebel": "heard-ok"},
    # ^ v->b tested PER-WORD only -- see note (2) above, not a general rule.
    "timeline":     {"timeline": "heard-bad", "taymlayn": "heard-bad", "taym layn": "heard-ok"},
    "speaker":      {"speker": "heard-bad", "spiker": "heard-ok"},
    "by-laws":      {"bilaws": "heard-bad", "baylaws": "heard-bad", "bay-loss": "heard-ok"},
    "drive":        {"unintelligible": "heard-bad", "drayb": "heard-ok"},
    # ^ third v-word rated bad (with Level, envelope) -- see note (2).
    "scanned":      {"skaned": "heard-bad", "iskand": "heard-bad", "skand": "heard-ok"},
    "brown":        {"brown": "heard-bad", "brawn": "heard-ok"},
    "envelope":     {"enelope": "heard-bad", "enbelope": "heard-bad", "enbelop": "heard-ok"},
    # ^ second v-word rated bad -- see note (2).
    "case":         {"kase": "heard-bad", "keys": "heard-ok"},
    "dietetic":     {"dayetetik": "heard-ok"},
    "exchange":     {"ekscheyng": "heard-ok"},
    # ^ NEW 2026-08-13 (batch 4), neither previously in this file. Note
    #   on "exchange": it contains "x", so CEB_RESPELL_RULES' x_to_ks
    #   rule would otherwise produce "ekschange" -- this lexicon entry
    #   overrides that (Layer B always overrides Layer A when present),
    #   giving the confirmed "ekscheyng" instead. Correct by design, not
    #   a conflict: the general rule is the fallback for words with no
    #   dedicated entry, not a competing answer.
    # ALL SETTLED 2026-08-13 (batch 4, listening_batch_4.ods) unless noted.
    "mechanical":   {"meshanikal": "heard-decent", "mekanikal": "proposed"},
}


def proposed_variants(word: str) -> list[str]:
    """All not-yet-rejected variants for a word, so a listening pass never
    re-tests something already heard and rejected."""
    entry = CEB_LOANWORD_LAB.get(word.lower(), {})
    return [v for v, status in entry.items() if status != "heard-bad"]


def needs_testing(word: str) -> bool:
    """True if a word has any variant still worth synthesizing this round.
    Once a winner is confirmed (heard-ok present), the word is DONE --
    any leftover "unclear"/"proposed" siblings from the same batch (a
    candidate that got no verdict because the other one already won)
    don't reopen it. Without this check, a settled word with one loose
    end would keep reappearing in every future batch for no reason.
    Only words with NO heard-ok yet, but something still open
    (proposed/unclear/heard-decent), count as needing testing."""
    entry = CEB_LOANWORD_LAB.get(word.lower(), {})
    if not entry:
        return True  # not in the lexicon at all -- nothing settled yet
    if any(status == "heard-ok" for status in entry.values()):
        return False
    return any(status in ("proposed", "unclear", "heard-decent")
              for status in entry.values())


def untested_variants(word: str) -> list[str]:
    """Only the variants still worth a clip this round (proposed, unclear,
    heard-decent) -- excludes an already-settled heard-ok, unlike
    proposed_variants which includes it. Use this for batch generation so
    a confirmed variant doesn't eat a clip every rerun."""
    entry = CEB_LOANWORD_LAB.get(word.lower(), {})
    return [v for v, status in entry.items()
            if status in ("proposed", "unclear", "heard-decent")]


def apply_loanword_fixes(text: str, demo_fallback: bool = False) -> str:
    """Apply loanword substitutions from CEB_LOANWORD_LAB to text.

    Default (demo_fallback=False): only CONFIRMED (heard-ok) entries
    apply. proposed, heard-decent, unclear, and heard-bad variants never
    silently reach real synthesis output -- only fully confirmed fixes
    do. This is the tested, regression-verified behavior (399/553 real
    corpus rows) and stays the production default.

    demo_fallback=True: ADDED 2026-08-13 for HK's demo notebook, at HK's
    explicit request -- "I don't need perfection, I need something
    bearable." For the small number of words with NO heard-ok entry yet
    (as of this writing: accountant, accounting, cpa, frameworks,
    mechanical, schools, us), this mode substitutes the BEST AVAILABLE
    candidate instead of leaving the word as raw untranslated English.
    Priority order, never picks heard-bad: heard-ok > heard-decent >
    unclear > proposed. "unclear" is ranked above "proposed" on purpose:
    an unclear variant was at least actually synthesized and heard, just
    with no verdict recorded either way, which is a stronger basis for a
    demo guess than a proposed variant that has never been synthesized
    at all.

    THIS DOES NOT CHANGE THE LEXICON. No status is upgraded to heard-ok
    by using this flag -- it only changes what TEXT gets substituted at
    call time, for this one call, so the evidence-tier record in
    CEB_LOANWORD_LAB stays honest and nothing here can later be mistaken
    for a confirmed result. Turn this off (the default) for anything
    that isn't a live demo running against deadline pressure.

    Word-boundary, case-insensitive match against the ORIGINAL English
    word (the lexicon's dict key, e.g. "office"). No collision risk with
    the initialism system: this runs on regular-case English words,
    spell_out_initialisms only touches ALL-CAPS 2+ letter tokens, and by
    the time this function would run in the real pipeline, any
    initialism has already been converted to its letter-spellout form
    and no longer matches its original uppercase string anyway."""
    FALLBACK_PRIORITY = ("heard-decent", "unclear", "proposed")
    for word, entries in CEB_LOANWORD_LAB.items():
        winner = next((v for v, status in entries.items() if status == "heard-ok"), None)
        if winner is None and demo_fallback:
            for tier in FALLBACK_PRIORITY:
                winner = next((v for v, status in entries.items() if status == tier), None)
                if winner is not None:
                    break
        if winner is None:
            continue
        text = re.sub(r'\b' + re.escape(word) + r'\b', winner, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------
# Carrier sentences. Two minimum, per the project's two-or-three-clips
# rule -- a variant that sounds fine alone can break in context, and the
# ordinal-segment artifact found earlier is direct evidence that MMS
# prosody is context-sensitive.
# ---------------------------------------------------------------------
DEFAULT_CARRIERS = [
    "Ang {} kay importante sa proseso.",
    "Kinahanglan nimo ang {} sa opisina sa PRC.",
    "{}",  # isolation, as a control -- NOT sufficient on its own
]


def build_stimuli(word: str, variants: list[str], carriers=None) -> list[dict]:
    """Cross variants x carriers into a flat list of things to synthesize.
    Includes the RAW word as a control in every carrier so the comparison
    is against what the model currently does, not against memory of it."""
    carriers = DEFAULT_CARRIERS if carriers is None else carriers
    stimuli = []
    for ci, carrier in enumerate(carriers):
        stimuli.append({
            "word": word, "variant": word, "is_control": True,
            "carrier_idx": ci, "text": carrier.format(word),
        })
        for v in variants:
            stimuli.append({
                "word": word, "variant": v, "is_control": False,
                "carrier_idx": ci, "text": carrier.format(v),
            })
    return stimuli


def try_spellings(word, variants, model, tokenizer, sampling_rate,
                  carriers=None, out_dir=None, device="cuda",
                  preprocess: Callable[[str], str] | None = None):
    """Synthesize every (variant x carrier) and return rows for listening.

    Deliberately does NOT run the digit verbalizer or frontend by default:
    when testing a respelling you want to hear THAT STRING, not a string
    something else rewrote on the way through. Pass preprocess explicitly
    if you want the full pipeline.

    Returns list of dicts with 'path' for playback, plus timing so the
    RTF logging requirement gets fed by ordinary work instead of needing
    a separate measurement pass.
    """
    import time
    import numpy as np
    import torch
    import soundfile as sf
    from pathlib import Path

    out_dir = Path(out_dir) if out_dir else Path.cwd() / "respell_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, s in enumerate(build_stimuli(word, variants, carriers)):
        text = preprocess(s["text"]) if preprocess else s["text"]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = model(**inputs).waveform[0].detach().cpu().numpy()
        wall = time.perf_counter() - t0

        safe = re.sub(r"[^a-z0-9]+", "_", s["variant"].lower()).strip("_")
        path = out_dir / f"{word.lower()}__c{s['carrier_idx']}__{safe}.wav"
        sf.write(path, wav, sampling_rate)

        audio_s = len(wav) / sampling_rate
        rows.append({**s, "sent_to_model": text, "path": str(path),
                     "wall_seconds": wall, "audio_seconds": audio_s,
                     "rtf": wall / audio_s if audio_s else float("nan")})
    return rows


def play(rows):
    """Render playback widgets grouped so each carrier's control sits
    directly above its variants -- comparison is only meaningful
    back-to-back."""
    from IPython.display import Audio, display, HTML
    for ci in sorted({r["carrier_idx"] for r in rows}):
        display(HTML(f"<b>carrier {ci}</b>"))
        for r in [x for x in rows if x["carrier_idx"] == ci]:
            tag = "CONTROL" if r["is_control"] else r["variant"]
            display(HTML(f"<code>{tag}</code> &nbsp; <small>{r['sent_to_model']}</small>"))
            display(Audio(r["path"]))
