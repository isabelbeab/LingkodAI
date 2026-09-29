# CHANGELOG

Dated record of the decisions made during the end-to-end integration. Where a
decision was later revisited, the original entry is kept and marked
superseded rather than rewritten.

---

## 2026-09-19 -- ASR source

**Issue:** Two versions of the STT code existed: a local `02_stt` notebook and
the script in the ASR author's GitHub repo.

**Reason:** The GitHub version's checkpoints already carry the custom
`<|cebuano|>` token (id 51865), and its audio preprocessing is shared with
audio LID, so one `load_audio()` feeds both encoders and the two stages cannot
drift apart on resampling or normalisation.

**Decision:** Use the GitHub script and the `beabucayan/lingkodai-whisper-{ceb,fil,eng}`
checkpoints, fp16 on CUDA. Keep `src/asr.py`'s vocab and embedding-size
assertions.
The repos moved to the `lingkodai` org on 2026-09-28; see that entry.

---

## 2026-09-19 -- MT decoding settings (stages 4 and 6)

**Issue:** The earlier Streamlit demos disagreed with each other on NLLB
decoding, and one used beam search plus a repetition penalty, which is not the
evaluated configuration.

**Reason:** Both MT notebooks (native to English, English to native) use the
same greedy, chunk-batch-reassemble pattern. They are the source of truth.

**Decision:** `facebook/nllb-200-3.3B`, fp16 on CUDA. Greedy: `generate()` gets
only `forced_bos_token_id` and `max_new_tokens=256`. Batch size 8, 800-character
sentence-boundary chunks, per-chunk retry on batch failure, `eng` passes
through. The transformers `max_length` warning on every call is expected.
Implemented in `src/mt.py`.

---

## 2026-09-19 -- RAG compute dtype on JOJIE (superseded)

**Issue:** The RAG notebook sets `bnb_4bit_compute_dtype=torch.bfloat16`, but
JOJIE's GPUs are Turing (compute capability 7.5) with no bf16 support.

**Reason:** bf16 on Turing either errors or falls back to something slow.

**Decision:** Use `torch.float16` on JOJIE and record it in run output as a
deviation from the evaluated Colab run. Restore bf16 on Ampere or newer.
Never implemented; corrected 2026-09-29, below.

---

## 2026-09-19 -- Which `charter_chunks` file to load

**Issue:** Two copies of the charter chunks exist (`charter_chunks (7).json`
from the RAG handover zip, `charter_chunks_v7.json` from elsewhere). They differ
in 6 of 79 chunks, including `retrieval_text`, so they can retrieve different
chunks for the same query.

**Reason:** The other twelve chunk files all come from the handover zip; mixing
sources is the choice nobody remembers later. This is about provenance, not
which file is better.

**Decision:** Load the zip copy and record its SHA-256 in run output. No
byte-exact DS6 reproduction is claimed for charter-derived answers until the
RAG author confirms which file produced DS6.

---

## 2026-09-19 -- Hugging Face access (superseded)

**Issue:** The ASR checkpoints are private, and the only access on hand was a
teammate's shared account login.

**Reason:** A password cannot authenticate an API call, and a token from a
shared account exposes the whole account if it leaks.

**Decision:** Use a fine-grained read-only token scoped to the three ASR repos,
set as `HF_TOKEN` in `.env`. Replace it with proper collaborator access and
revoke it when the repo owner is back.
Superseded 2026-09-28 by the move to the `lingkodai` org, below.

---

## 2026-09-19 -- Ported-code rules for stages 5 to 7

**Issue:** Several behaviours in the source notebooks look like bugs or
unnecessary complexity and would be easy to "tidy" wrongly.

**Reason:** Each one is load-bearing for matching the evaluated run.

**Decision:**
- RAG drops LangChain; a plain message list reproduces
  `RunnableWithMessageHistory` exactly.
- The TTS initialism set is frozen in `data/init_set_ds6.json` and never
  derived per answer (per-answer derivation silently misses plurals like
  "CPAs").
- `concat_segments` keeps receiving a float where its signature suggests a
  dict; it only collects the value.
- A TTS frontend raise returns a text-only turn, never a crash.
- Scoring-side normalizers are never used in synthesis.

---

## 2026-09-21 -- TTS reference clips from local disk (superseded)

**Issue:** The plan had the reference clips coming from a private HF repo
(`TTS_REF_REPO`) that was never created.

**Reason:** The GPU pipeline runs where the clips already sit on disk, and
uploading CC-BY-NC research audio to a personal account gains nothing.

**Decision:** Drop `TTS_REF_REPO`. `src/tts.py` finds the three locked clips by
filename anywhere under `TTS_REF_DIR` and raises if one is missing or
ambiguous. `ref_text` values are constants copied verbatim from the notebook,
including the deliberate "siyete" substitution.
Superseded 2026-09-29 by the private eval-assets dataset repo, below.

---

## 2026-09-22 -- Sharing one encoder across the three ASR checkpoints

**Issue:** The audio LID and three ASR models each hold a Whisper-medium
encoder. Sharing one saves VRAM, but only if fine-tuning left every encoder
unchanged; otherwise one language silently gets the wrong weights.

**Reason:** `scripts/verify_shared_encoder.py` (run on JOJIE, CPU, fp32) found
all three fine-tuned encoders byte-identical to base `openai/whisper-medium`
(SHA-256 `c16520ef83cb19de...` for all four). The ceb vocab difference (51,866)
is the decoder-only Cebuano token.

**Decision:** Sharing is verified safe for the three ASR models. The audio LID
encoder stays separate in fp32, since its head was trained on fp32 outputs.
No sharing code written: staged mode only ever loads one ASR model, so the
saving (about 1.2GB) only matters for a future resident mode.

---

## 2026-09-22 -- ASR tokenizers fail to load under transformers 4.57.3

**Issue:** All three ASR repos store `extra_special_tokens` as a list; the
pinned transformers 4.57.3 expects a dict and raises `AttributeError` when
building the processor.

**Reason:** The transformers pin is fixed by `qwen-tts`, and `src/asr.py` is
owned by another teammate and must not be edited.

**Decision:** `scripts/patch_asr_tokenizers.py` builds local copies (weights
symlinked, not re-downloaded) with the list moved to
`additional_special_tokens`, verified against base Whisper's special-token ids.
`asr.py` is pointed at them through its existing `ASR_*` environment variables.
Drop this once the tokenizers are re-saved upstream.

---

## 2026-09-22 -- TTS out-of-memory in staged mode

**Issue:** The first golden-check run OOM'd loading TTS in phase 5 on an 11GB
card, 166MiB short.

**Reason:** `mt.py` and `rag.py` unloaded with `del` and `empty_cache()` but no
`gc.collect()`, so lingering references kept memory from returning between
phases.

**Decision:** Add `gc.collect()` to both `unload()`s and print free GPU memory
at every phase start. Confirmed: every phase now starts with about 11.10GB
free, and all 24 golden-check turns synthesize.

---

## 2026-09-22 -- Routing: audio LID wins `final_lang` (superseded)

**Issue:** The stage 1 to 3 notebooks were not available, so the rule for
audio LID versus text LID disagreement could not be reproduced from source.

**Reason:** Audio LID must pick the turn-1 ASR checkpoint anyway (there is no
transcript yet), so letting it also decide `final_lang` was the simplest rule.

**Decision:** Audio LID wins, the disagreement is recorded as
`was_overridden`, ASR is not re-run, and the language locks after turn 1.
Superseded the same day by the entry below.

---

## 2026-09-22 -- Routing: text LID wins `final_lang`

**Issue:** In the golden check, audio LID mispredicted the ceb conversation as
`eng` (0.769) while text LID correctly said `ceb` (0.714). Under the old rule
the whole conversation was misrouted: no MT, English answers, English voice.

**Reason:** The ASR/LID author reviewed the rule directly: text LID runs on
the actual transcript, which audio LID never sees, so it should win.

**Decision:** Audio LID, run on the first turn, still picks the ASR
checkpoint used for every turn, but text LID decides `final_lang` (used from
MT in onward) on disagreement. `was_overridden` is still recorded and
ASR is still not re-run. Implemented in `src/routing.py`. Confirmed
2026-09-23: `final_lang` now matches DS6 for all three conversations.

---

## 2026-09-28 -- Password gate and cloned-voice disclosure

**Issue:** The demo's synthesized voice is a cloned speaker from a CC-BY-NC,
research-only corpus, and the team had not yet signed off on how that is
handled.

**Reason:** Stakeholders reviewed the approach and accepted it.

**Decision:** Keep the `APP_PASSWORD` gate and the on-page disclosure that the
voice is a cloned research-corpus speaker. Closed.

---

## 2026-09-28 -- Fly.io demo hosting retired

**Issue:** The CPU demo had been hosted on Fly.io, with an optional
`GPU_BACKEND_URL` link to a separately hosted GPU backend for live answers.
The GPU-backend split was abandoned after it proved unstable, which left a
"Live pipeline" tab on the hosted page with nothing to call.

**Reason:** The hosted demo no longer served a purpose the local CPU image and
the standalone GPU demo (`app/gpu_chat_demo.py`) do not already cover, and a
half-working public page is worse than none.

**Decision:** Remove `fly.toml`, `app/gpu_backend_api.py`, the "Live pipeline"
tab and `GPU_BACKEND_URL`. The CPU image is run locally with `docker run`.
The Fly app itself is shut down separately.

---

## 2026-09-28 -- ASR checkpoints moved to the `lingkodai` Hugging Face org

**Issue:** The three ASR checkpoints lived in the ASR author's personal
Hugging Face account. The stakeholders created a `lingkodai` org and the ASR
author uploaded every model there: `lingkodai/lingkodai-whisper-{ceb,fil,eng}`
and `lingkodai/lingkodai-audio-lid-head`.

**Reason:** The project's models should not depend on one person's account.
Before switching, `scripts/compare_hf_repos.py` compared the Hub's file hashes
without downloading weights. Every file in all three ASR org repos matches the
personal repo, including `model.safetensors`, `tokenizer.json` and
`tokenizer_config.json`. The org's `deeper_50chunks.pt` has the same SHA-256 as
the committed `models/deeper_50chunks.pt`
(`101c32f02a254a1596617d44a98c9eccbc0f761862232b787ddf31e2b066dc99`).
Org commits checked: ceb `b90ec682`, fil `25722b97`, eng `9b9d4ffa`, LID head
`0ac45810`.

**Decision:** `src/asr.py`, `scripts/verify_shared_encoder.py` and
`scripts/patch_asr_tokenizers.py` default to the `lingkodai/` repo IDs,
approved by the ASR author. `HF_TOKEN` is now a fine-grained read-only token
from an org member's own account, scoped to the three ASR org repos. The audio
LID head is still loaded from the committed `models/deeper_50chunks.pt`, at the
ASR author's call: the org copy is an archive, and `src/lid_audio.py` is
unchanged. The tokenizer patch (2026-09-22 entry) still applies, since the org
copies carry the same unpatched `tokenizer_config.json`.

**Verified 2026-09-29 on JOJIE (CPU only, no GPU assigned):**
`scripts/patch_asr_tokenizers.py --out outputs/asr_patched_org` downloaded and
patched all three org repos (ceb 51,866 tokens with the custom token at 51865;
fil and eng 51,865; 107 base special tokens match in each), at the same org
commits listed above. `scripts/ab_asr_checkpoints.py` then transcribed the 24
golden-check clips (8 per language, each with its own language's checkpoint)
with the old and the org copies, fp32 on CPU: 24/24 transcripts identical. A
full golden check was not rerun, since only stage 2 changed.

---

## 2026-09-29 -- RAG compute dtype: correction

**Issue:** The 2026-09-19 entry says RAG runs with
`bnb_4bit_compute_dtype=torch.float16` on JOJIE. A fresh-clone audit found
`src/rag.py` has used `torch.bfloat16`, the notebook's value, since it was
written (commit `ee33b55`, 2026-09-21), and no run output records a dtype
deviation.

**Reason:** The golden-check runs on JOJIE (2026-09-22 and 2026-09-23) ran
this code, so they used bf16 and completed. The record should say what
actually ran.

**Decision:** No code change. `src/rag.py` keeps bf16, matching the
evaluated Colab run; the 2026-09-19 entry is marked superseded.

---

## 2026-09-29 -- Private eval assets on Hugging Face

**Issue:** Reproducing the golden check needs files that cannot go in the
GitHub repo, which is public: the three TTS reference clips (CC-BY-NC,
research only), `DS6_augmented.xlsx`, and the golden conversation's 24
source clips. They existed only on JOJIE, so no one else could run the
golden check or TTS. The 2026-09-21 entry kept the clips on local disk.

**Reason:** A private repo in the `lingkodai` org is not a personal account,
and access goes through the same fine-grained read-only tokens already used
for the ASR checkpoints.

**Decision:** One private Hugging Face dataset repo,
`lingkodai/lingkodai-eval-assets`: `tts_ref/` (the three clips, unpadded),
`golden/DS6_augmented.xlsx`, `golden/audio/multi_turn/...` (conversation
`CONV-MT-001` in all three languages, in the same layout as under the JOJIE
`PRC_synthetic_qa` folder) and a `MANIFEST.sha256` of every file.
`scripts/fetch_eval_assets.py` downloads it, verifies the manifest and prints
`TTS_REF_DIR` and the golden-check command with its `--audio-root-map`.
`src/tts.py` is unchanged: it still searches `TTS_REF_DIR` by filename.

**Verified 2026-09-29:** uploaded from JOJIE by an org member, private,
commit `e995094b`, 31 files (the 28 assets, the manifest, the dataset card
and `.gitattributes`). `scripts/fetch_eval_assets.py` then downloaded it
with a fine-grained read-only token and all 28 manifest entries matched, on
JOJIE and again on Colab.

---

## 2026-09-29 -- Golden check on Colab

**Issue:** The golden check had only ever run on JOJIE, which needs a JOJIE
account and an assigned GPU. No one else could reproduce it.

**Reason:** `notebooks/colab_golden_check.ipynb` runs it from a fresh clone
on Colab, pulling the private assets from Hugging Face.

**Decision:** Keep the notebook as the documented way to reproduce the
golden check without JOJIE. Result of the first full run (NVIDIA L4 24 GB,
Python 3.13.15, torch 2.11.0+cu128,
transformers 4.57.3), compared with the 2026-09-23 JOJIE run:

- `final_lang`: 3/3 match (ceb, fil, eng), as on JOJIE.
- eng: 8/8 exact on every field, as on JOJIE.
- ceb: transcript 4/8, English query 4/8, English answer 1/8, native answer
  1/8, the same counts as JOJIE, including the two known follow-ups
  (`ceb_01_mt_04` falls back to no-match; `ceb_01_mt_07` contradicts DS6 on
  the processing time). The `ceb_01_mt_01` transcript diff is DS6's own
  mojibake (`serviciÃ³n`), not a pipeline difference.
- fil: transcript and English query 7/8, English and native answer 4/8
  (JOJIE: 8/8 and 5/8). The one new diff is `fil_01_mt_05`: ASR on this GPU
  heard `pamenta'y ang` where DS6 has `pamenta ang`, MT then rendered the
  question as "What are the first two sentences?", and RAG answered with a
  general description of the service instead of listing the first two
  documents. The other fil answer diffs are wording drift.

Greedy decoding shifting slightly across GPUs is expected (see README,
Testing); the fil_01_mt_05 cascade is an example of how one token of ASR
drift can change a later answer.

TTS: 24/24 turns synthesized, but `ceb_01_mt_05` failed by ear: about 12
seconds in, one segment never produced a stop token and ran to the
`max_new_tokens=4096` cap (about 5.7 minutes at 12 tokens per second) of
garbled audio. Its text fields all match DS6 exactly, and the evaluated DS7
run synthesized the same answer normally, so this is run-to-run variation in
Qwen3-TTS generation (no seed is set, in the source notebook or the port),
not a frontend problem. The evaluated DS7 run may have hit it too: 8 of its
2,235 rows (6 eng, 2 ceb) have a round-trip WER above 2, meaning the
transcribed audio had far more words than the text, which fits this failure
but was not confirmed by listening. Known issue, not fixed: a guard that treats a
segment reaching the cap as a TTS failure (text-only turn, the existing
designed fallback) is a candidate follow-up, but it is a deviation from the
source notebook and needs a GPU test.
