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

---

## 2026-10-12 -- README "Start here" and automatic checks on GitHub

**Issue:** At the Milestone 3.5 handover, a stakeholder reading the README
asked:
- what the terminal commands are for
- whether the repo has to be cloned first
- what "the config file" is and how to make one
- what was actually tested
- how anyone can be sure the Docker images work

The meeting asked the README to cover setup, environment, data, the apps,
input, output, configuration, and adding documents to the knowledge base.

**Reason:** The README assumed a reader who already knew the project. It
never said to clone the repo, mentioned `.env` only in passing, and did not
describe the output files. Every check so far had also been run by hand.
Nothing re-checked the repo when it changed, and the GPU image had been
built only once, locally.

**Decision:**
- The README gains:
  - a "Start here" path from a fresh machine to a first answer
  - a table of every `.env` variable and where its value comes from
  - data and model tables
  - the input arguments and every output field
  - how to run both apps
  - a table of what has been verified, how and when
- "Adding documents to the knowledge base" is a placeholder for the RAG
  author, listing the loader's current constraints.
- `.github/workflows/ci.yml` runs on every push and pull request:
  - the pytest suite
  - a build of the CPU image, which must then serve its page
  - a build of the GPU image, which must then import every pipeline module
- GitHub's runners have no GPU, so a passing run shows the code and both
  images build and load, not that inference works. Running the golden check
  inside the GPU image on a GPU Docker host is still open.

---

## 2026-10-12 -- Audio LID head loaded from Hugging Face

**Issue:** The stakeholders' handover conditions ask for every model to come
from Hugging Face. The audio LID head was the one model still committed to
the repo, as `models/deeper_50chunks.pt`. The 2026-09-28 entry kept it there
at the ASR/LID author's call, with the org repo as an archive.

**Reason:** The ASR/LID author approved the move on 2026-10-12. The org copy
had already been checked identical (the 2026-09-28 entry). It was checked
again today with `scripts/compare_hf_repos.py`, and by downloading it and
loading it into `build_head()`.

**Decision:**
- `src/lid_audio.py` downloads `deeper_50chunks.pt` from
  `lingkodai/lingkodai-audio-lid-head`, pinned to commit `0ac45810`, and
  refuses to load it unless its SHA-256 is
  `101c32f02a254a1596617d44a98c9eccbc0f761862232b787ddf31e2b066dc99`. That
  is the committed file's hash, so the weights the pipeline loads have not
  changed.
- `LID_HEAD` still overrides it with a local file.
- `models/` is removed, along with its `COPY` in `docker/Dockerfile.gpu` and
  its exceptions in `.dockerignore` and `.gitignore`.
- The `HF_TOKEN` scope gains this fifth repo.
- `scripts/compare_hf_repos.py` now checks the org file against the
  recorded hash instead of the deleted local copy.
- The file stays in git history. Removing it from history would mean
  rewriting `main`'s history, and nothing requires that.

---

## 2026-10-12 -- Running a folder of your own recordings

**Issue:** The README explained the golden check and a single
`run_conversation.py --audio FILE` call. It had no simple way to run a batch
of your own pre-recorded questions and find the answers, which is what a new
user tries first.

**Reason:** Typing every file path, and then opening one folder per
conversation to read the answers, is a barrier for anyone who is not an
engineer.

**Decision:**
- `scripts/run_folder.py` runs everything in `input/` and writes to
  `output/`:
  - An audio file is a one-question conversation.
  - A subfolder is a multi-turn conversation, with its turns in filename
    order.
  - Each conversation gets the same output as `run_conversation.py`, plus
    `output/summary.csv` with one row per question.
  - Finished conversations are skipped on a rerun.
  - A failure is recorded in the summary and does not stop the others.
- It reuses `pipeline.run_staged` unchanged. Each conversation therefore
  loads the models again. Loading them once for the whole folder would
  change `src/pipeline.py`, and is left until the cost is measured on a GPU.
- `notebooks/colab_run_folder.ipynb` does the same with the Drive folders
  `lingkodai_input` and `lingkodai_output`.
- `input/` (apart from its README) and `output/` are gitignored and
  dockerignored.
- Tested with a fake pipeline (`tests/test_run_folder.py`), then run for
  real on 2026-10-12 with `notebooks/colab_run_folder.ipynb` on a Colab L4,
  from branch `stakeholder-readme-ci` at `c25f163`:
  - Input: one single question (`ta_prc_q1.ogg`) and one 2-turn subfolder
    (`2-turn/`), all Cebuano recordings made by the team.
  - All 3 questions: `final_lang` ceb, status ok, audio written. A team member
    listened and judged the audio clear.
  - 344 s for the 2-turn conversation, 422 s for the single question; most
    of it is model loading, paid once per conversation.
  - The audio LID head loaded from the `lingkodai` org repo with no error.
- Answer-quality notes from that run, for the RAG author. Neither comes
  from `run_folder.py`; `src/rag.py` passes the model's text through as the
  source notebook does:
  - "unsáon pagsketúl ug apóintment" (schedule an appointment) got the
    no-match fallback. Either the knowledge base has no matching chunk or
    retrieval missed it.
  - "unsáon pag-renew ug lisúnsya" (renew a license) got a correct answer,
    but the generator's reasoning about the candidate documents came out in
    front of it ("Ang tukma nga tubag mao: ..."), and TTS spoke all of it.

---

## 2026-10-13 -- DS6 defined in the README; name kept

**Issue:** "DS6" appears across the README, scripts and file names without
ever being defined, so a new reader meets it out of nowhere. A plainer name
such as `test_dataset` was considered.

**Reason:** Renaming touches 21 tracked files, about 130 mentions. Some of
them cannot change:
- the vendored TTS frontend, which is never edited
- the frozen `data/init_set_ds6.json`, which records its source file's name
- this file's history
The file itself is `golden/DS6_augmented.xlsx` in the private eval-assets
repo, listed in its `MANIFEST.sha256`. `scripts/fetch_eval_assets.py`
checks for that exact path, so renaming the file there would make every
existing copy of the script fail its check at once. The golden check
only runs on a GPU, so the automatic checks would not catch a reference
that was missed.

**Decision:** Keep the name. The README gains "The test dataset (DS6)",
under Data and models: what DS6 is, how it was made, what the `_augmented`
file adds, where this repo uses it, and what DS7 is. The README's DS6
mentions link to it.

---

## 2026-10-14 -- Knowledge-base section written by the RAG author

**Issue:** "Adding documents to the knowledge base" in the README was a
placeholder (2026-10-12 entry). The RAG author sent their write-up as a
Word document.

**Reason:** It is the RAG author's method (how documents were extracted,
chunked, reviewed and formatted), so it goes in as they wrote it.

**Decision:** The README section now holds the RAG author's text with
formatting changes only:
- Markdown headings and tables
- field and file names in code style
- straight quotes
- a link to the Chunk Format part
- the folder written as `data/prc_chunks/` (the document had
  `/data/prc_chunks`)

Under it, "What the code checks when it loads the files" lists the
loader's checks (see the next entry), plus:
- which fields `src/rag.py` actually reads
- how to check that a new document is retrieved (its `chunk_id` in the
  turn's `chunk_ids`)

The Word file was not committed.

---

## 2026-10-14 -- Knowledge-base loader counts the files instead of expecting 13 and 547

**Issue:** The RAG author asked whether the loader's fixed counts could be
dynamic. `src/rag.py` refused to start unless `data/prc_chunks/` held
exactly 13 files and 547 chunks, so adding a document meant editing two
numbers in the code. It also meant `--rag-data-dir`, documented as "a
different knowledge-base folder", only worked for a folder with those
exact counts.

**Reason:** The fixed counts were added during the port; the source notebook
listed its 13 files by name and only printed the total. They guarded
against one trap: a browser download renames `charter_chunks.json` to
`charter_chunks (7).json`, the `*_chunks.json` glob skips it, and a
document drops out of the knowledge base unnoticed. That trap can be
caught by name instead.

**Decision:** `load_chunks` loads every `*_chunks.json`, however many, and
stops with a clear error on:
- any other `.json` file in the folder (the download trap)
- a missing folder, or no chunk files
- a file that is not valid JSON, or not a non-empty list of objects
- a chunk missing `chunk_id`, `chunk_title`, `retrieval_text` or
  `full_entry_text`, the four fields the code always reads
- a duplicate `chunk_id` (as before)

It prints each file with its chunk count, then the total, and a note when
the totals differ from the evaluated knowledge base (13 files, 547 chunks,
kept as `EVALUATED_CHUNK_FILES` and `EVALUATED_TOTAL_CHUNKS`). A test now
runs these checks on the repo's own `data/prc_chunks/`, without pinning
counts.

What is given up: a file deleted by accident no longer stops the run. The
printed file list shows it, and git shows a deleted file in `git status`
and in a pull request.

Answers do not change. On the current 13 files the new loader returns the
same 547 chunks in the same order as the old one (compared directly,
2026-10-14), so the search index is built from identical input. No GPU
re-run is needed for this change.

---

## 2026-10-14 -- Golden check on JOJIE after the October changes

**Issue:** The last golden check on JOJIE was on 2026-09-23. Since then the
audio LID head moved to Hugging Face (2026-10-12) and the knowledge-base
loader changed (above), and neither had run on JOJIE.

**Reason:** A re-run on the same GPU as 2026-09-23 shows whether anything
changed the results.

**Decision:** Recorded as passing. Run on `main` at `f2b8de2`, RTX 2080 Ti,
torch 2.14.0+cu126, transformers 4.57.3. The eval assets on this JOJIE
account were an older copy outside `outputs/eval_assets`, so they were
fetched again with `scripts/fetch_eval_assets.py --out outputs/eval_assets`
first. Compared with the 2026-09-23 JOJIE run:

- `pytest`: 145 pass on JOJIE. The loader printed 13 files and 547 chunks,
  with no note.
- `final_lang`: 3/3 match (the script exits with an error otherwise; it
  ended cleanly).
- eng: no diffs on any field.
- fil: no transcript or English query diffs; English and native answer
  5/8 (`fil_01_mt_01`, `_03`, `_05`), the same count as 2026-09-23. The
  `fil_01_mt_05` answer differs only in the capitalization of "Letter of
  Request". The ASR drift seen on Colab for that turn did not happen here.
- ceb: English query 4/8, English and native answer 1/8, the same counts as
  2026-09-23, including the two known follow-ups (`ceb_01_mt_04` falls back
  to no-match; `ceb_01_mt_07` contradicts DS6 on the processing time).
  Transcript diffs on `ceb_01_mt_04` and `_07`, as before; the top of the
  diff list was not in the part of the log copied back, so the exact ceb
  transcript count was not recorded.
- TTS: the audio was judged good by listening, apart from the Cebuano joins
  (next entry).

---

## 2026-10-14 -- Cebuano speech: audible joins between pieces (known issue)

**Issue:** Listening to the JOJIE run above, the TTS author noticed that in
Cebuano answers, at every pause between sentences or lines, the voice
changes slightly, as if the speaker resets, and the background noise
disappears for a split second. English and Filipino answers do not do this.

**Reason:** Cebuano answers go through the vendored segmenter.
`src/tts.py`'s `_synth_ceb` generates each piece (a clause, sentence or
list item) with its own Qwen3-TTS call, and the vendored
`ceb_segmentation.concat_segments` joins them with pauses of pure digital
silence (`np.zeros`): 0.20 s after a clause, 0.45 s after a sentence,
0.60 s after a list item. Each call imitates the reference voice
separately, with its own random variation (no seed is set; see
2026-09-29), and the silent pauses have none of the faint background noise
in the generated speech. English and Filipino answers are generated in one
call. This is how the synthesis notebook's `synthesize()` made the
evaluated DS7 audio. A similar voice shift between pieces was reported in
August with the earlier VITS model; `ceb_segmentation.py`'s `voice_seed`
note records it.

**Decision:** Kept as is, and listed as a known issue in the README
(Testing). Two possible fixes, both changes from the evaluated audio and so
the TTS author's decision, not made here:
- fill the pauses with low-level background noise taken from the generated
  speech, or overlap the pieces with a short crossfade, inside
  `src/tts.py` (the vendored files are not edited)
- reset the random seed to the same value before every piece, as the
  vendored `voice_seed` option does; its own note calls it an untested
  hypothesis, written for the earlier model

Either needs a GPU run and listening before it could replace the current
behaviour.

---

## 2026-10-14 -- README: instructions for JOJIE

**Issue:** The TTS author asked whether a teammate following the README on
JOJIE would succeed. The commands were right, but steps JOJIE needs were
missing or only in the golden-check section:
- nano crashes in JOJIE's terminal, and there is no `sudo` for
  `apt install`
- `PYTHONNOUSERSITE=1` was not mentioned anywhere, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `nohup` and the
  `nvidia-smi` check only for the golden check
- "Start here" step 2 shows venv, without saying JOJIE uses the conda
  environment
- step 4 did not say to leave out the `export` the setup scripts print
- the GTX 1080 Ti was listed as if it worked, though only the RTX 2080 Ti
  has run the pipeline

**Reason:** Checked on JOJIE on 2026-10-14: `ffmpeg` is installed for
everyone, `sox` is not (it was conda-installed into `lingkod-e2e`), home
folders share one drive that was 97% full (876 GB free), and JupyterLab's
File > Open from Path... can open the hidden `.env`.

**Decision:** "Start here" gains an "On JOJIE" section with those
differences in step order (conda, `sox` through conda, disk check, editing
`.env`, what to set in every new terminal, `nohup` for long runs) and a
JOJIE row in the routes table. Elsewhere: step 4 says to drop `export`;
every new terminal sets `PYTORCH_CUDA_ALLOC_CONF` (the GPU Docker image and
the Colab notebooks already do); the system-packages notes give the conda
install for machines without `sudo`; the GTX 1080 Ti is marked untested;
the golden check gives a command that prints just its results.
