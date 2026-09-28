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

## 2026-09-19 -- RAG compute dtype on JOJIE

**Issue:** The RAG notebook sets `bnb_4bit_compute_dtype=torch.bfloat16`, but
JOJIE's GPUs are Turing (compute capability 7.5) with no bf16 support.

**Reason:** bf16 on Turing either errors or falls back to something slow.

**Decision:** Use `torch.float16` on JOJIE and record it in run output as a
deviation from the evaluated Colab run. Restore bf16 on Ampere or newer.

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

## 2026-09-19 -- Hugging Face access

**Issue:** The ASR checkpoints are private, and the only access on hand was a
teammate's shared account login.

**Reason:** A password cannot authenticate an API call, and a token from a
shared account exposes the whole account if it leaks.

**Decision:** Use a fine-grained read-only token scoped to the three ASR repos,
set as `HF_TOKEN` in `.env`. Replace it with proper collaborator access and
revoke it when the repo owner is back.

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

## 2026-09-21 -- TTS reference clips from local disk

**Issue:** The plan had the reference clips coming from a private HF repo
(`TTS_REF_REPO`) that was never created.

**Reason:** The GPU pipeline runs where the clips already sit on disk, and
uploading CC-BY-NC research audio to a personal account gains nothing.

**Decision:** Drop `TTS_REF_REPO`. `src/tts.py` finds the three locked clips by
filename anywhere under `TTS_REF_DIR` and raises if one is missing or
ambiguous. `ref_text` values are constants copied verbatim from the notebook,
including the deliberate "siyete" substitution.

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

**Decision:** Audio LID still picks the turn-1 ASR checkpoint, but text LID
decides `final_lang` on disagreement. `was_overridden` is still recorded and
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
