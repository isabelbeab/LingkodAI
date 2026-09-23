# CHANGELOG

A dated record of the decisions made during the end-to-end integration, and
the reasoning behind each. Where a decision was later revisited, the original
entry is kept for the record and marked superseded rather than rewritten.

Context: the integration began with Bea out sick and Francis having handed
over his RAG material; several early decisions were made with what was in
hand rather than waiting on people, and are flagged as such below.

---

## 2026-09-19 -- ASR: use the GitHub STT script, not the local notebook

Use the STT code in Bea's repo on GitHub. Ignore any local `02_stt` notebook.

Bea's reason, verbatim in substance: the Cebuano token id is already inserted
into the checkpoint published on Hugging Face, the syntax in the GitHub
version is cleaner, and its audio preprocessing is consistent with the audio
LID path. That last point matters most, because it means one `load_audio()`
produces the mel features that both the LID encoder and the ASR encoder
consume, so the two stages cannot drift apart on resampling or normalisation.

The three checkpoints in use are `beabucayan/lingkodai-whisper-{ceb,fil,eng}`,
roughly 3GB each on disk in fp32, loaded in fp16 on CUDA. Cebuano carries a
custom `<|cebuano|>` token at id 51865, which is why `src/asr.py` asserts on
vocab and embedding size. Keep those assertions.

Would change this: Bea saying a newer checkpoint has been pushed.

---

## 2026-09-19 -- MT settings, both directions (stages 4 and 6)

Identical in and out. `facebook/nllb-200-3.3B`, fp16 on CUDA and fp32 on CPU,
6.7 GB allocated. Greedy decoding: `generate()` receives only
`forced_bos_token_id` and `max_new_tokens=256`, nothing else. Tokenizer
called with `padding=True, truncation=True`. Batch size 8, flattening chunks
across rows. Sentence-boundary chunking at 800 characters via
`re.split(r"(?<=[.!?])\s+", ...)`. `eng` rows pass through untouched with no
NLLB call. Language codes: `{"ceb": "ceb_Latn", "fil": "tgl_Latn",
"eng": "eng_Latn"}`.

On a whole-batch failure, retry per chunk so partial progress isn't lost. A
row counts as successful only if it gets back the same number of chunks it
was split into.

NLLB's generation config carries `max_length=200`; `max_new_tokens` takes
precedence and transformers warns about it on every call. That warning is
expected, not a defect -- don't "fix" it by suppressing or matching it.

Verified against both `reference/bea_stage4_mt_multi_turn.ipynb` (native to
English) and `reference/bea_stage5_backtranslate_w_lid.ipynb` (English to
native), which use an identical chunk-batch-reassemble pattern in both
directions. Implemented in `src/mt.py`.

---

## 2026-09-19 -- RAG quantisation uses fp16 compute, not bf16

Francis's notebook sets `bnb_4bit_compute_dtype=torch.bfloat16`, which is
correct on the Colab GPU he ran it on. JOJIE's card is compute capability
7.5, Turing, with no bf16 tensor cores. Set
`bnb_4bit_compute_dtype=torch.float16` for the JOJIE run.

This is a deliberate deviation from the evaluated Colab configuration and is
recorded as such in run output, since generation can differ slightly under a
different compute dtype. It is not optional: bf16 on Turing either errors or
falls back to something slow.

Would change this: running on Ampere or newer hardware, where bf16 should be
restored to match Francis's configuration.

---

## 2026-09-19 -- Use Francis's `charter_chunks` from the handover zip

The zip contains `charter_chunks (7).json`. A separate TTS project knowledge
source contains `charter_chunks_v7.json`. They differ in 6 of 79 chunks,
across `patch_log`, `page_start`, `full_entry_text`, `fee_schedule`,
`retrieval_text`, `steps`, `situational_requirements` and `total_fee`. Since
`retrieval_text` is among them, the two files can return different chunks
for the same query, which propagates through the RAG answer and everything
downstream of it.

Use the copy from Francis's zip. The reasoning is consistency of provenance
rather than a judgement about which is better: the other twelve chunk files
all come from that zip, and mixing one file from a different source into
eleven from another is the version of this that nobody remembers later. Put
`charter_chunks_v7.json` in `reference/` with a note, and record the SHA-256
of whichever file the pipeline actually loads in the run output.

The consequence to state plainly: until someone confirms which file produced
the evaluated DS6 results, no byte-exact DS6 reproduction is claimed for
charter-derived answers. The demo works either way. This affects a claim,
not the build.

Would change this: Francis confirming which file he ran.

---

## 2026-09-19 -- Hugging Face access: a scoped token, not Bea's account

Do not use Bea's account login. A shared password cannot authenticate an API
call anyway, and a token minted from a shared account exposes her entire
account if it leaks. Use a fine-grained token scoped read-only to the three
ASR repositories, set as `HF_TOKEN` in `.env`, and ask Bea to add HK as a
collaborator when she is back so the interim token can be revoked.

---

## 2026-09-19 -- Stage 5-7 implementation notes carried over from the source material

A handful of ported-code decisions worth keeping visible outside the module
docstrings that actually enforce them (`src/rag.py`, `src/tts.py`):

- **RAG drops LangChain.** `RunnableWithMessageHistory` only ever stored a
  human/ai message list; a plain per-conversation list reproduces it exactly,
  with one less dependency.
- **The TTS initialism set is frozen**, generated once by
  `scripts/build_init_set.py` into `data/init_set_ds6.json`, and must be
  loaded, never derived per answer. The notebook's regex
  (`\b[A-Z]{2,}[0-9]*\b`) does not match "CPAs" at all, so the singular "CPA"
  only enters the set by appearing elsewhere in the corpus, and
  `spell_out_initialisms` then strips the plural and finds it. Deriving the
  set per answer breaks that silently and leaves acronyms unspelled.
- **`ceb_segmentation.concat_segments` is fed a generator whose third element
  is a float**, not the timing dict its signature suggests. This works only
  because `concat_segments` merely collects it. Preserve as-is; this is not a
  bug to tidy.
- **The TTS frontend raises on some inputs by design** (three CEB rows raised
  in the DS7 run and were excluded). `synthesize()` preserves that --
  `synthesize_turn()` is the non-crashing entry point the pipeline actually
  calls: a raise is caught, audio is set to `None`, and the turn's text
  answer is unaffected.
- **Never use scoring-side normalizers (`scoring_normalize*`) in synthesis**
  -- those exist for WER evaluation, not for what gets spoken.

---

## 2026-09-21 -- TTS reference clips read from local disk, not a private HF repo

An earlier plan described the three locked reference clips and
`ref_config.json` as coming from a private HF repo named by `TTS_REF_REPO`.
That repo was never created; the variable was speculative. The GPU pipeline
only runs on JOJIE, where the locked clips already sit on disk, and
uploading CC-BY-NC research audio to a personal HF account would gain
nothing.

**Decision.** `TTS_REF_REPO` is dropped. `src/tts.py` reads the three locked
clips from the directory in `TTS_REF_DIR`, finding each by filename at any
depth and raising with the resolved path if a clip is missing or ambiguous.
`ref_text` and `language` are constants in `src/tts.py`
(`REF_TEXT_CONFIG`), copied verbatim from `REF_CONFIG` in the synthesis
notebook, including the deliberate "siyete" substitution in the CEB text.
The 0.5s trailing-silence padding and the locked clip filenames are
unchanged.

The clip location is not hard-coded because two layouts have been claimed:
the notebook used `<DEMO_DIR>/ref_audio/` (flat), the other is
`data/audio_preprocessing/{ENG,FIL,CEB}/<id>/`. Recursive lookup by filename
covers both.

Would change this: the team creating a shared reference repo for real.

---

## 2026-09-22 -- Encoder sharing across the three ASR checkpoints: verified safe

Bea's suggestion was that the audio LID encoder and all three STT encoders
are frozen, so the pipeline could hold a single encoder instead of four. The
underlying claim is right in principle, with one qualification worth
spelling out: "frozen" has two meanings, and only one makes sharing safe.
Frozen at inference time just means no gradients, true of every model
loaded, and implies nothing. Frozen during fine-tuning means the encoder
weights were never updated, so every fine-tuned checkpoint still contains
the original `openai/whisper-medium` encoder, byte for byte -- sharing is
safe only under this second meaning. If one language had been fine-tuned
with the encoder unfrozen, sharing would silently swap in the wrong weights
for that language: a degraded transcript, not an exception, so nothing in
the logs would show it.

**Verification result** (`scripts/verify_shared_encoder.py`, run on JOJIE,
CPU, fp32, `lingkod-e2e`). The gate is cleared: all three fine-tunes carry
the base encoder byte for byte.

```text
  name   encoder sha256         enc params    vocab   dec emb
  ceb    c16520ef83cb19de...   307,216,384   51,866    51,866
  fil    c16520ef83cb19de...   307,216,384   51,865    51,865
  eng    c16520ef83cb19de...   307,216,384   51,865    51,865
  base   c16520ef83cb19de...   307,216,384   51,865    51,865
```

The 51,866 vocab on ceb is the custom `<|cebuano|>` token at id 51865, a
decoder change only. Models were fetched with the fine-grained read-only
token from the 2026-09-19 HF-access decision above.

**The audio LID encoder stays separate, in fp32, unshared regardless.** Two
reasons: its head was trained against fp32 encoder outputs, and LID decides
the language for an entire conversation, so a borderline prediction flipping
under fp16 costs the whole conversation rather than one turn. The remaining
saving is about 600MB, not worth spending the evaluated behaviour on.

**Scope note.** Sharing saves about 1.2GB of VRAM only if all three ASR
models are held at once. `run_staged` loads only one, because the
conversation language is locked by audio LID before ASR loads
(`src/pipeline.py`, `asr.load(lid_result.lang)`). So sharing saves nothing in
`staged` mode and matters only for a future `resident` mode. No sharing code
was written this pass.

---

## 2026-09-22 -- ASR tokenizers loaded from patched local copies

Found on JOJIE during the first golden-check run. All three
`beabucayan/lingkodai-whisper-*` repos store `extra_special_tokens` in
`tokenizer_config.json` as a list (ceb 1 entry, fil and eng 107 each).
`transformers` 4.57.3, which `qwen-tts` pins and this project may not change,
expects a dict and raises `AttributeError: 'list' object has no attribute
'keys'` when `asr.load` builds the processor. The earlier encoder check
missed this because it loaded weights only, not the tokenizer.

**Decision.** `src/asr.py` is Bea's and stays untouched. It already reads
`ASR_CEB`, `ASR_FIL` and `ASR_ENG` from the environment, so
`scripts/patch_asr_tokenizers.py` builds a local copy of each checkpoint
(weights symlinked from the HF cache, no second download) with the list
moved to `additional_special_tokens`, and prints the exports that point
`asr.py` at those copies. The script raises unless the patched tokenizer's
special-token ids match base `openai/whisper-medium` and `<|cebuano|>` sits
at id 51865. Tokens themselves are unchanged; only the config key differs.

This is a deviation from loading straight from the Hub. The proper fix is
for Bea to re-save the tokenizers with the pinned version; ask when she is
back, and drop the `ASR_*` overrides once she has.

---

## 2026-09-22 -- Missing `gc.collect()` caused a phase-5 TTS OOM on JOJIE

The first real golden-check attempt got through phases 1-4 for the `ceb`
conversation and OOM'd in phase 5 (TTS) on JOJIE's 11GB RTX 2080 Ti, inside
`tts.load()`'s warm-up clone -- only 166MiB short with 10.46GB already in
use.

Root cause: `mt.py` and `rag.py`'s `unload()` had `del` plus
`torch.cuda.empty_cache()` but were missing `gc.collect()`. Without it,
accelerate/bitsandbytes hooks and NLLB buffers held references that blocked
memory from fully returning between phases, so free VRAM got tighter each
phase instead of resetting.

**Fix:** added `gc.collect()` to both `unload()`s, plus a one-line GPU
free/total memory print at the start of every phase in `src/pipeline.py`
(`_print_gpu_mem`) so a future OOM can be diagnosed from the log instead of
guessed at. Confirmed fixed: phase-3 start now consistently shows about
11.10GB free regardless of how many phases have already run, and the
affected `ceb` conversation completed all 5 phases including TTS for every
turn with no OOM.

---

## 2026-09-22 -- Routing: audio LID picks the ASR checkpoint, text LID decides `final_lang`

**Superseded 2026-09-22, same day, after Bea's direct review** -- see the
entry immediately below. Kept here for the record.

Bea's Stage 1 to 3 multi-turn notebooks were not in hand at integration
time, so the exact override rule between audio LID and text LID was not
reproducible from source. Decided as follows: audio LID runs on the first
turn's audio and its label selects the ASR checkpoint (forced rather than
chosen -- ASR cannot run until a checkpoint is picked, and a checkpoint
cannot be picked from a transcript that does not exist yet). Text LID then
runs on the resulting transcript and its label is recorded. Where the two
disagree, **audio LID wins**, the disagreement is written to the turn's
output record as `was_overridden`, and ASR is not re-run.

The language is then locked for the rest of the conversation -- confirmed
twice over in the source material: an explicit comment in Bea's notebook,
and independently by row-count arithmetic across the with-LID and no-LID
runs, where 62 differing rows over roughly 7.8 turns per conversation works
out to about 8 whole conversations flipping rather than individual turns.

### Superseding update, same day: text LID wins instead

Bea reviewed this rule and said it was wrong: **text LID should win
`final_lang` on disagreement**, since it runs on the actual transcript,
which audio LID never sees. Audio LID still picks the turn-1 ASR checkpoint
-- that ordering is still forced -- but the checkpoint choice no longer also
decides `final_lang` when the two disagree. ASR is still not re-run under
the corrected language.

This explains a real symptom from the same golden-check run: the `ceb`
conversation's audio LID mispredicted `eng` (confidence 0.769) while text
LID correctly said `ceb` (0.714); under the old rule `eng` won and the whole
conversation was misrouted. Under this rule it locks to `ceb` correctly.

Implemented in `src/routing.py` (`decide()`), tests updated in
`tests/test_routing.py`. This is Bea's word, not a reproduced source -- her
Stage 1-3 notebooks are still not in `reference/` -- so treat this as
authoritative but not yet independently verified from source. The golden
check has not yet been re-run against this fix; that is how to confirm it
end to end.

Would change this again: Bea's actual notebooks surfacing and showing a
third, different rule.
