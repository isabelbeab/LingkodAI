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
but was not confirmed by listening. (Corrected 2026-10-11: those rows' audio
has normal length; see "DS7 rows with round-trip WER above 2: correction".) Known issue, not fixed: a guard that treats a
segment reaching the cap as a TTS failure (text-only turn, the existing
designed fallback) is a candidate follow-up, but it is a deviation from the
source notebook and needs a GPU test.

---

## 2026-10-11 -- Stage 7 on qwentts.cpp (branch `tts-cpp`)

**Issue:** Stage 7 takes about 2.3 times the answer's audio length to
synthesize on an A100 (DS6 synthesis notebook, batch 1: RTF 2.30 ceb over 742
rows, 2.39 fil). A median answer, about 20 s of audio, waits about 48 s for
TTS alone, too slow for live use.

**Reason:** qwentts.cpp, a GGML port of the same Qwen3-TTS model, ran at RTF
0.34 to 0.41 on a 6 GB laptop GPU with Q8_0 weights in an earlier trial, and
its own test logs show 0.28 (F32) and 0.22 (Q8_0) on a workstation GPU. It
runs as a separate process: that trial hit a CUDA out-of-memory error that
segfaulted, and a crash in a child process can become a text-only turn
instead of a dead conversation.

**Decision:** A second Stage 7 backend, `src/tts_cpp.py`, selected with
`--tts-backend cpp`; the default stays `hf` (`src/tts.py`, the evaluated
path). `tts-server` runs on a free localhost port; the three padded
reference voices are registered once per server start, resampled to 24 kHz
with librosa as qwen-tts does and sent as float WAV. Language is `english`
for eng and omitted (auto) for fil and ceb, matching `language=None`
upstream. Sampling defaults are the same as qwen-tts (temperature 0.9, top-k
50, top-p 1.0, repetition penalty 1.05, the same for the sub-talker). Known
differences that remain: the server returns 16-bit WAV where qwen-tts returns
float; qwentts.cpp trims the reference to whole 1920-sample codec frames
before encoding it (under 80 ms of the padding silence). Weights:
`Serveurperso/Qwen3-TTS-GGUF` at revision `b7ee2e8`, F32 by default and Q8_0
as an extra benchmark arm, each file checked against its SHA-256
(`scripts/fetch_qwentts_weights.py`). The server is built from source
(`scripts/build_qwentts.sh`) because upstream's prebuilt images cannot carry
the two patches below. Turn JSON now records `tts_wall_sec` and `tts_rtf`
for both backends. Built and tested on CPU with a fake server only; not yet
run on a GPU.

---

## 2026-10-11 -- qwentts.cpp patch: ICL prompt layout

**Issue:** qwentts.cpp builds the voice-clone (ICL) prompt like upstream's
`generate_icl_prompt` with `non_streaming_mode=False`: text and reference
codes aligned and summed. `src/tts.py` and the DS6/DS7 synthesis use
`non_streaming_mode=True`: the whole text over `codec_pad`, then the
reference codes over `tts_pad`. A different prompt conditions the model
differently.

**Reason:** Parity with the evaluated path is the point of arm B, and the
`True` layout is a small, separate branch of the same function.

**Decision:** `patches/qwentts/0001-icl-non-streaming-layout.patch` adds the
`True` layout behind `--icl-non-streaming` (off by default, so upstream
behaviour is unchanged) and the matching option on qwentts.cpp's own cossim
harness. `src/tts_cpp.py` always passes the flag and refuses a server whose
startup log does not confirm it. `scripts/validate_qwentts_patch.py` checks
the layout against qwen-tts on Colab (prompt ids exact, embedding shapes
equal, text rows at cosine 0.9999 or more, prefill logits at 0.999 or more);
not yet run.

---

## 2026-10-11 -- qwentts.cpp patch: talker KV length

**Issue:** qwentts.cpp's talker KV cache is fixed at 4096 positions. Stage 7
allows `max_new_tokens=4096` frames on top of the ICL prompt, so a long
generation would overflow the cache and fail partway through.

**Reason:** The cache must hold the prompt plus every generated frame.

**Decision:** `patches/qwentts/0002-talker-kv-length.patch` adds
`--talker-ctx` and refuses, before any compute, a request whose prompt plus
`max_new_tokens` cannot fit (HTTP 400 with both numbers).
`src/tts_cpp.py` uses 7168 positions: 4096 frames plus 3072 for the prompt,
since parity mode sends a whole fil or eng answer as one prompt and the
longest DS6 answer (`fil_01_mt_328`) has 3,362 verbalized characters. The
cache is f32, about 224 KiB per position, so 7168 positions take 1.5 GiB.

---

## 2026-10-11 -- Segmented mode: fil and eng split before verbalizing

**Issue:** Only ceb answers are split into segments; fil and eng answers are
synthesized whole (up to 327 s of audio in DS7). Nothing can play before the
whole answer is done, and a runaway anywhere costs the whole answer.

**Reason:** `ceb_segmentation.split_for_synthesis` (140 characters, pauses
0.20 / 0.45 / 0.60 s) already serves ceb and works on any language. Splitting
before verbalizing, ceb's own order, keeps list numbers attached to their
item; splitting after verbalizing turned them into tiny spoken segments
("one." alone; 96 eng segments under 10 characters against 30).

**Decision:** In `segmented` mode (arm C) every language is split first, with
the vendored file unedited, then each segment is verbalized. A list marker at
the start of a block is spoken with the language's own reading: ceb's
`CEB_ORDINALS` as before, fil's verbalizer's own ordinal table, and for eng
the eng verbalizer's reading of the number ("one, ..."). For fil and eng,
`paren_conditional_strip` runs once on the whole answer before the split: it
drops a parenthetical only when its words appear bare elsewhere, and per
segment it would have kept, for example, "(PRC)" that parity drops. Checked
on all DS6 answers against DS6's `verbalized_text`: the words differ only by
"." against "," after list ordinals, markdown `**` removed by the splitter's
own cleanup, numbers before a capitalised sentence no longer misread as list
ordinals by the fil verbalizer ("Dokumento ng Kandidato 2. Samakatuwid" now
reads "two", where parity reads "pangalawa"), and the "No." readings in the
next entry. Some clauses have no split points; the longest verbalized
segments are 813 ceb, 1,261 fil and 777 eng characters, left whole and
covered by the length cap. `parity` mode (arm B) and the `hf` backend are
unchanged.

---

## 2026-10-11 -- Segmented mode: "No." kept with its number

**Issue:** `split_for_synthesis` ends a sentence at "No.", so "PRB Resolution
No." and "02-2025." become separate segments with a 0.45 s pause between
them. In DS6 that happens 35 times in ceb answers, 34 in eng and 21 in fil;
"(s. 2008)" splits the same way 4 times and "A.Y. 2018" 3 times.

**Reason:** The number belongs to the abbreviation; ceb's evaluated audio
already has these pauses, so the fix applies to arm C only.

**Decision:** Approved 2026-10-11 for "No." and "R.A."; applied as well to
"Nos.", "(s." and "A.Y.", the same failure. A guard in `src/tts_cpp.py` (the
vendored splitter stays unedited) replaces the space after these
abbreviations with a placeholder while the splitter runs, when a number, or
"No." and a number, follows. On DS6 it removes all 98 such splits and
changes no word. Because the ceb verbalizer now sees "No." and the number
together, it reads some of those numbers through its own "No." rule, for
example "2019-1146" after "Resolution No."; worth a listen.

---

## 2026-10-11 -- Segmented mode: length cap, retry, text-only turn

**Issue:** The 2026-09-29 known issue: a segment occasionally never produces a
stop token and runs to `max_new_tokens=4096` (4096 frames at 12.5 per second,
327.7 s of garbled audio). In DS7, `ceb_01_mt_185` ran 327.6 s, 3.08 times the expected length,
UTMOS 1.54.

**Reason:** Audio length is predictable from verbalized text (DS6:
correlation 0.998 ceb, 0.995 fil, 0.94 eng; about 14.9 characters per second,
so 0.84 frames per character; actual over expected has a p99 of 1.2). A cap
at twice the expected length catches `ceb_01_mt_185` with no false positives
on the other 2,234 rows. Runaways are random per generation (Colab's
`ceb_01_mt_05` runaway was a short segment), so a new seed is a fair retry.

**Decision:** In `segmented` mode each segment's `max_new_tokens` is
`ceil(2 x 0.84 x characters) + 25`, at most 4096. A segment that reaches its
cap, or returns no audio, is retried once with a new seed; a second failure
makes the whole turn text-only, the existing designed fallback. A segment is
only played after it ends under its cap, so a runaway is never heard. `parity`
mode keeps `max_new_tokens=4096` and only records cap hits. Its retry at 2048
fires on an out-of-memory error from the server; a GPU out-of-memory in
qwentts.cpp crashes the server instead, which gives a text-only turn, and the
server restarts before the next turn.

---

## 2026-10-11 -- Seeds for the qwentts.cpp backend

**Issue:** Neither the synthesis notebook nor `src/tts.py` sets a seed, so no
TTS run can be repeated, and a retry needs a seed different from the attempt
it replaces.

**Reason:** `tts-server` takes a seed per request.

**Decision:** The cpp backend takes a base seed (`--tts-seed`; drawn and
recorded when omitted). Each request's seed derives from the base seed, the
turn id, the segment index and the attempt, and every one is recorded in the
turn JSON. This is meant to repeat a run on the same build and GPU; not yet
checked on a GPU. The `hf` backend stays unseeded, as evaluated.

---

## 2026-10-11 -- DS7 rows with round-trip WER above 2: correction

**Issue:** The 2026-09-29 entry suggested the 8 DS7 rows with round-trip WER
above 2 fit the runaway failure.

**Reason:** Their audio has normal length: 0.85 to 1.15 times the length
expected from their verbalized text (DS6 average, 15.1 characters per
second), where a runaway runs three times or more. Their judge transcripts
have 2.8 to 7 times the reference's words, and the four inspected
(`ceb_01_mt_85`, `ceb_01_mt_697`, `eng_01_mt_157`, `eng_01_mt_734`) repeat a
phrase until they stop ("as specified as specified ..."): the Whisper judge
looped, not the synthesized audio.

**Decision:** The 8 rows are judge failures, not TTS runaways. DS7's one
known runaway is `ceb_01_mt_185`, whose round-trip WER is only 0.68 because
the judge stops transcribing long clips early. Not confirmed by listening.
The 2026-09-29 sentence is marked with a pointer here.

---

## 2026-10-11 -- Stage 7 A/B scoring

**Issue:** Comparing the backends needs the DS6/DS7 judge, and the DS6/DS7
scoring scripts exist only on JOJIE.

**Reason:** The DS2/DS5 versions of the judge scripts and the DS6/DS7 result
notebooks document the same setup, and `DS6_augmented.xlsx` stores
`reference_norm`, `hyp_norm` and `tts_rt_wer` per row to check a port
against.

**Decision:** `scripts/tts_ab.py` synthesizes the same DS6
`rag_answer_native` texts in their `final_lang` with every arm, so only
Stage 7 differs. Judge (decided 2026-10-11): `lingkodai/lingkodai-whisper-{lang}`
(not verified to be the same weights as JOJIE's `best_checkpoints/{lang}`),
base whisper-medium processor, fp16, batch 8, `max_length=440`, the same
forced decoder ids, no long-form chunking. The judge stops transcribing long
clips early, so WER counts only rows whose verbalized text has at most 75
words, the DS6/DS7 error analysis cutoff. The hypothesis is normalized as
DS6's `hyp_norm` (`scoring_normalize`, plus `eval_side_normalize` for eng),
which reproduces DS6 on 2,235 of 2,235 rows, and scored against DS6's stored
`reference_norm`, because the JOJIE scorer's reference de-lexicalization
(reversing respellings such as "sertifayd") is not in the vendored files;
`jiwer.wer` on the stored columns reproduces `tts_rt_wer` exactly. Arm C's
text differs from that reference only as listed in the splitting entry above.
UTMOS22 via `utmos-pytorch==0.1.0` `UTMOSScoreTorch`, as DS6 was scored.

---

## 2026-10-11 -- Demo notebooks

**Issue:** The Streamlit demos may not be used to show the pipeline, and
there was no lighter way to show it live: `run_staged` reloads every model
per conversation, and `app/gpu_chat_demo.py` needs Streamlit and a 40 GB GPU.

**Reason:** A notebook runs on Colab or a teammate's own machine with no
server, and can show every stage's output next to the answer.

**Decision:** Three notebooks on branch `demo-notebooks`: `demo_audio`
(spoken question in, spoken answer out), `demo_chat` (typed questions, speech
optional) and `demo_replay` (saved sessions and golden-check results, no
GPU). The live two use `src/demo.py`, a resident session that keeps each
model loaded after first use and otherwise follows `run_staged`'s stage order
and conversation rules. Differences from the evaluated pipeline, all
deliberate and shown in the output:

- Typed questions skip stages 1 and 2; GlotLID (the stage 3 text LID) picks
  the conversation's language on the first question.
- `lang=` on a first question sets the language without LID, as the
  Streamlit GPU demo's language picker does.
- Speech: qwentts.cpp in parity mode with F32 weights, arm B of the Stage 7
  A/B, chosen by HK on 2026-10-12 after the first default (segmented, Q8_0,
  for memory headroom on an L4). Arm B's tts-server used about 9.4 GB on
  the L4 in run1, so with every model loaded the notebooks need an
  estimated 21 GB (chat) and 24 GB or more (audio), past an L4's 22.5 GB
  usable: they ask for an A100 and say what to change on an L4.
- Models stay resident; greedy outputs are expected to match staged runs,
  not verified.

`scripts/build_qwentts.sh` gained `--cache-dir`: a build is cached under a key
of everything that shapes the binaries and restored only after passing the
same checks as a fresh build (tested on the CPU build: build, delete,
restore). Not yet run on a GPU.

---

## 2026-10-12 -- qwentts.cpp: trim each segment's silent lead-in

**Issue:** Listening to A/B run1, arm B's ceb answers sounded split at
every sentence, and its fil and eng answers did not. ceb is split in the
evaluated path too (`full_ceb_pipeline_segments`, pauses of 0.20, 0.45 and
0.60 s), so arm A has the same pauses, but arm B's were audibly longer.

**Reason:** Every qwentts.cpp generation starts with silence that qwen-tts
does not produce. Measured on run1's WAVs, the silence before the first word
had a median of 0.13 to 0.22 s in arm B against 0.00 to 0.04 s in arm A,
across all three languages. In fil and eng that happens once, before the
answer starts. In ceb it is added to every pause: the median pause between
ceb segments was 0.60 s in arm B against 0.46 s in arm A (90th percentile
0.88 s against 0.68 s). Rebuilding arm B's 22 ceb answers from its own
segment WAVs with each segment's lead-in cut to 20 ms gave 0.48 s and
0.67 s, matching arm A; the same rebuild without the cut reproduced arm B's
WAVs exactly. Why the server adds the silence is not verified.

**Decision:** `src/tts_cpp.py` trims every segment, in both modes, to 20 ms
before its first 10 ms frame at or above 1e-3 RMS (about -60 dBFS), before
the segments are joined. A segment with no such frame is left as is. The
trim changes the audio only, not the text path or the frame counts used for
the cap. It is a deviation from `src/tts.py`'s output, so it is a load()
option, `trim_lead_in`, on by default and recorded in the backend info.
`trim_lead_in=False` reproduces the audio of A/B run1, which predates this
change; run1's scores were not recomputed.
