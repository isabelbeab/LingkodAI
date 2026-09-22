# DECISIONS.md

Decisions taken for the end-to-end integration, with the reasoning behind each.
Read this alongside CLAUDE.md. Where the two disagree, this file is newer and
wins, and CLAUDE.md should be corrected to match.

Status: settled 2026-09-19. Bea is out sick this week, Francis has handed over
his RAG material, and the deadline is Tuesday 2026-09-22. Everything below was
decided with what is in hand rather than waiting on people. Each entry says what
would change the decision, so a later correction is a small edit rather than a
rewrite.

**For Claude Code: these are decided. Do not ask about them again. Ask only if
you find evidence in the source files that contradicts one, in which case say
which file and which line.**

---

## 1. The GitHub STT script is the source of truth, not the local notebook

Use the STT code in Bea's repo on GitHub. Ignore any local `02_stt` notebook.

Bea's reason, verbatim in substance: the Cebuano token id is already inserted
into the checkpoint published on Hugging Face, the syntax in the GitHub version
is cleaner, and its audio preprocessing is consistent with the audio LID path.
That last point is the one that matters most for us, because it means one
`load_audio()` produces the mel features that both the LID encoder and the ASR
encoder consume, so the two stages cannot drift apart on resampling or
normalisation.

The three checkpoints in use are `beabucayan/lingkodai-whisper-{ceb,fil,eng}`,
roughly 3GB each on disk in fp32, loaded in fp16 on CUDA. Cebuano carries a
custom `<|cebuano|>` token at id 51865, which is why `src/asr.py` asserts on
vocab and embedding size. Keep those assertions.

Changes this decision: Bea saying a newer checkpoint has been pushed.

---

## 2. Share one encoder across the three ASR checkpoints, but only after verifying

Bea's suggestion was that the audio LID encoder and all three STT encoders are
frozen, so the pipeline could hold a single encoder instead of four. The
underlying claim is right in principle and worth acting on, with two
qualifications.

**The word "frozen" carries two meanings and only one of them makes sharing
safe.** Frozen at inference time just means no gradients, which is true of every
model we load and implies nothing. Frozen during fine-tuning means the encoder
weights were never updated, so every fine-tuned checkpoint still contains the
original `openai/whisper-medium` encoder, byte for byte. Sharing is safe only
under the second meaning. If one language was fine-tuned with the encoder
unfrozen, sharing swaps in the wrong weights for that language and the failure
mode is a degraded transcript, not an exception, so nothing in the logs would
show it.

**Decision: verify first, then share across the three ASR checkpoints only.**
Run `scripts/verify_shared_encoder.py` before writing any sharing code. It
fingerprints the encoder state dict of all three fine-tunes plus base
`openai/whisper-medium` and reports which are identical. It runs on CPU and does
not touch the GPU, so it is safe to run while JOJIE is busy.

If the three fine-tune fingerprints match, implement sharing this way: load each
model on CPU, assign the one shared encoder module to `model.model.encoder` on
all three, then move to CUDA. Do not precompute encoder outputs and pass them
through `generate(encoder_outputs=...)`. The memory saving is identical, the
compute saving is not worth the integration risk this week, and the simple
version is three lines.

Expected saving is around 1.2GB of VRAM, which matters because the 11GB card has
to hold NLLB-3.3B and Qwen3-4B elsewhere in the run.

**The audio LID encoder stays separate, in fp32, unshared.** Two reasons. Its
head was trained against fp32 encoder outputs, and LID decides the language for
an entire conversation, so a borderline prediction flipping under fp16 costs the
whole conversation rather than one turn. The remaining saving is about 600MB,
which is not worth spending the evaluated behaviour on. Revisit after Tuesday by
checking whether fp16 produces identical labels across the eval set.

Changes this decision: the verification script reporting a mismatch, in which
case drop sharing entirely and say so rather than working around it.

**Verification result (run 2026-09-22 on JOJIE, CPU, fp32, `lingkod-e2e`).**
The gate is cleared: all three fine-tunes carry the base encoder byte for byte.

```text
  name   encoder sha256         enc params    vocab   dec emb
  ceb    c16520ef83cb19de...   307,216,384   51,866    51,866
  fil    c16520ef83cb19de...   307,216,384   51,865    51,865
  eng    c16520ef83cb19de...   307,216,384   51,865    51,865
  base   c16520ef83cb19de...   307,216,384   51,865    51,865
```

The 51,866 vocab on ceb is the custom `<|cebuano|>` token at id 51865, a decoder
change only. Models were fetched with a fine-grained read-only token (item 6).

**Scope note.** The 1.2GB saving above assumes all three ASR models are held at
once. `run_staged` loads only one, because the conversation language is locked
by audio LID before ASR loads (`src/pipeline.py`, `asr.load(lid_result.lang)`).
So sharing saves nothing in `staged` mode and matters only for `resident` mode.

---

## 3. Audio LID decides the language, text LID confirms it

Bea's Stage 1 to 3 multi-turn notebooks are not in hand, so the exact override
rule between audio LID and text LID is not reproducible from source. Decided as
follows.

Audio LID runs on the first turn's audio and its label selects the ASR
checkpoint. This ordering is forced rather than chosen: ASR cannot run until a
checkpoint is picked, and a checkpoint cannot be picked from a transcript that
does not exist yet. Text LID then runs on the resulting transcript and its label
is recorded. Where the two disagree, audio LID wins, the disagreement is written
to the turn's output record, and ASR is not re-run.

The language is then locked for the rest of the conversation. This is not an
assumption. It is confirmed twice over in the source material: an explicit
comment in Bea's notebook, and independently by row-count arithmetic across the
with-LID and no-LID runs, where 62 differing rows over roughly 7.8 turns per
conversation works out to about 8 whole conversations flipping rather than
individual turns.

Keep this rule in `src/routing.py` and nowhere else, so that if Bea's notebooks
later show a different override it is one function to change.

Changes this decision: Bea's Stage 1 to 3 notebooks arriving and showing a
re-run on disagreement.

---

## 4. Use Francis's `charter_chunks` from the handover zip

The zip contains `charter_chunks (7).json`. The TTS project knowledge contains
`charter_chunks_v7.json`. They differ in 6 of 79 chunks, across `patch_log`,
`page_start`, `full_entry_text`, `fee_schedule`, `retrieval_text`, `steps`,
`situational_requirements` and `total_fee`. Since `retrieval_text` is among them,
the two files can return different chunks for the same query, which propagates
through the RAG answer and everything downstream of it.

Use the copy from Francis's zip. The reasoning is consistency of provenance
rather than a judgement about which is better: the other twelve chunk files all
come from that zip, and mixing one file from a different source into eleven from
another is the version of this that nobody remembers in three weeks. Put
`charter_chunks_v7.json` in `reference/` with a note, and record the SHA-256 of
whichever file the pipeline actually loads in the run output.

The consequence to state plainly: until someone confirms which file produced the
evaluated DS6 results, we do not claim byte-exact DS6 reproduction for
charter-derived answers. The demo works either way. This affects a claim, not the
build.

Changes this decision: Francis confirming which file he ran.

---

## 5. RAG quantisation uses fp16 compute, not bf16

Francis's notebook sets `bnb_4bit_compute_dtype=torch.bfloat16`, which is correct
on the Colab GPU he ran it on. JOJIE's card is compute capability 7.5, Turing,
with no bf16 tensor cores. Set `bnb_4bit_compute_dtype=torch.float16` for the
JOJIE run.

This is a deliberate deviation from the evaluated Colab configuration and should
be recorded as such in the run output, since generation can differ slightly under
a different compute dtype. It is not optional: bf16 on Turing either errors or
falls back to something slow, and neither is a good discovery to make on Monday.

Changes this decision: running on Ampere or newer hardware, where bf16 should be
restored to match Francis's configuration.

---

## 6. Hugging Face access

Do not use Bea's account login. A shared password cannot authenticate an API call
anyway, and a token minted from a shared account exposes her entire account if it
leaks. Use a fine-grained token scoped read-only to the three ASR repositories,
set as `HF_TOKEN` in `.env`, and ask Bea to add HK as a collaborator when she is
back so the interim token can be revoked.

This is not a code blocker. Nothing in the build waits on it.

---

## 8. TTS reference clips load from local disk, not a private HF repo

CLAUDE.md describes the clips and `ref_config.json` as coming from a private HF
repo named by `TTS_REF_REPO`. That repo was never created; the variable was
invented in `.env.example`. The GPU pipeline only runs on JOJIE, where the
locked clips already sit on disk, and uploading CC-BY-NC research audio to a
personal HF account would gain nothing. This entry supersedes the CLAUDE.md
wording (see the precedence rule under Open items there).

**Decision.** `TTS_REF_REPO` is dropped. `src/tts.py` reads the three locked
clips from the directory in `TTS_REF_DIR`, finding each by filename at any depth
and raising with the resolved path if a clip is missing or ambiguous. `ref_text`
and `language` are constants in `src/tts.py` (`REF_TEXT_CONFIG`), copied verbatim
from `REF_CONFIG` in the synthesis notebook, including the deliberate "siyete"
substitution in the CEB text. The 0.5 s trailing-silence padding is unchanged.
The locked clip filenames are unchanged.

The clip location is not hard-coded because two layouts have been claimed. The
notebook used `<DEMO_DIR>/ref_audio/` (flat); the other is
`data/audio_preprocessing/{ENG,FIL,CEB}/<id>/`. Recursive lookup by filename
covers both. Verify on JOJIE with `find` before the golden check.

Changes this decision: the team actually creating a shared reference repo.

---

## 9. ASR tokenizers are loaded from patched local copies

Found on JOJIE, 2026-09-22, during the first golden-check run. All three
`beabucayan/lingkodai-whisper-*` repos store `extra_special_tokens` in
`tokenizer_config.json` as a list (ceb 1 entry, fil and eng 107 each).
`transformers` 4.57.3, which `qwen-tts` pins and we may not change, expects a
dict and raises `AttributeError: 'list' object has no attribute 'keys'` when
`asr.load` builds the processor. The earlier encoder check missed it because it
loaded weights only.

**Decision.** `src/asr.py` is Bea's and stays untouched. It already reads
`ASR_CEB`, `ASR_FIL` and `ASR_ENG`, so `scripts/patch_asr_tokenizers.py` builds a
local copy of each checkpoint (weights symlinked from the HF cache, no second
download) with the list moved to `additional_special_tokens`, and prints the
exports that point `asr.py` at those copies. The script raises unless the patched
tokenizer's special-token ids match base `openai/whisper-medium` and `<|cebuano|>`
sits at id 51865. Tokens are unchanged; only the config key differs.

Observed on JOJIE: text tokens are identical to base for fil, and the saved
prefix differs (fil and eng carry `<|transcribe|>`, id 50359, base does not).
That is expected, because `asr.py` sets language and task itself, so the check
compares text tokens and reports the prefix rather than comparing it.

This is a deviation from loading straight from the Hub. The proper fix is for
Bea to re-save the tokenizers with the pinned version; ask when she is back.

Changes this decision: Bea pushing repaired tokenizer configs, after which the
ASR_* overrides can be dropped.

---

## 7. Nothing above blocks Tuesday

Two items affect what we can claim rather than what we can ship. Item 4 means no
byte-exact DS6 reproduction claim for charter-derived answers. Item 5 means the
RAG generation configuration differs from Francis's evaluated run in one
documented way. Both belong in the write-up as stated deviations.

The item 2 verification gate was cleared on 2026-09-22 (output recorded in item
2). Sharing is safe to implement, but see the scope note there.

Everything else is decided.
