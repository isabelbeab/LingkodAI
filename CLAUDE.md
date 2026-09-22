# CLAUDE.md: LingkodAI end-to-end pipeline

Read this whole file before planning any change. It is the project context; the
code and notebooks it points to are the ground truth.

## What this repo does

LingkodAI answers spoken questions about Philippine Professional Regulation
Commission (PRC) services. A citizen speaks Cebuano (`ceb`), Filipino (`fil`) or
English (`eng`); the system replies with synthesized speech in the same language.
Scope: Scenario 2 only (language identification on), multi-turn conversations.

Stage order for every turn:

1. Audio LID: predict the spoken language from the audio.
2. ASR: transcribe with that language's fine-tuned Whisper.
3. Text LID and routing: confirm or override the language, lock it for the conversation.
4. MT in: native to English (`eng` passes through unchanged).
5. RAG: rewrite follow-ups, hybrid retrieval, LLM chunk selection, English answer.
6. MT out: English answer to the conversation language (`eng` passes through).
7. TTS: speak the native answer.

## What ships

Three artifacts, built from this one repo.

1. **The package plus CLI.** `python -m lingkodai` style entry point in
   `scripts/run_conversation.py`: audio files in, per-turn JSON and WAV out.
   This is the real system and the thing the golden check runs against.
2. **The CPU image and the Fly app.** A slim container with no torch, hosting a
   public demo: the text frontend run live, GlotLID run live, and pre-rendered
   audio from recorded end-to-end conversations. This is what gets a URL.
3. **The GPU image.** The full pipeline containerized. Built and import-checked
   locally, verified for real only on JOJIE via conda, because JOJIE is a shared
   JupyterHub and cannot run Docker. The README says exactly that. Fly retired
   GPU machines on 2026-08-01, so there is no GPU host yet; the CPU app reaches a
   GPU backend through one environment variable when one exists.

Never describe the system as "deployed" or "in production". It is a
containerized, reproducible pipeline with a hosted front end.

## Ownership and editing rules

- `src/audio.py`, `src/lid_audio.py`, `src/asr.py`, `src/lid_text.py` and
  `models/` belong to Bea, who is away this week. Do not edit them. If one
  appears to need a change, stop and ask. Workarounds go in new files.
- `pyproject.toml`, `README.md`, `.gitignore`, `docker/`, `fly.toml` are shared.
- New stage modules are ports, not redesigns: same models, prompts, constants,
  decoding settings, thresholds and fallback messages as their source of truth.
  Prompts must be byte-identical to the source.
- `src/tts_frontend/` holds HK's TTS text-frontend modules copied verbatim from
  JOJIE. Never edit them. Record each file's md5 and copy date in
  `src/tts_frontend/VENDORED.md`. They import each other by top-level name, so
  callers put `src/tts_frontend/` on `sys.path` rather than rewriting imports.
  Required files: `shared_domain_frontend.py`, `ceb_respell_lab.py`,
  `ceb_segmentation.py`, `ceb_prc_digit_verbalizer.py`, `ceb_full_pipeline.py`,
  `fil_prc_digit_verbalizer.py`, `eng_prc_digit_verbalizer.py`,
  `lingkod_tts_utils.py`, `lingkod_stt_utils.py`. All nine, or imports fail:
  `fil_prc_digit_verbalizer` calls `tts._require_stt()` at import time.

## Sources of truth

`reference/` is gitignored and must never be committed. Read these when working
on the matching stage; do not load them wholesale into context.

| Stage | Source of truth | Key facts |
|---|---|---|
| 1 Audio LID | `src/lid_audio.py` | Whisper-medium encoder fp32 (matches training) plus `models/deeper_50chunks.pt` |
| 2 ASR | `src/asr.py` | fp16; CEB uses a custom Cebuano token at id 51865 |
| 3 Text LID | `src/lid_text.py` | GlotLID v3 pinned revision, 4 candidate labels, `tgl` maps to `fil` |
| 3 Routing | OPEN, see Open items | audio vs text override rule |
| 4 MT in | `reference/bea_stage4_mt_multi_turn.ipynb` | see MT settings below |
| 5 RAG | `reference/francis_rag_pipeline.ipynb` | Francis's cleaned notebook is authoritative |
| 6 MT out | `reference/bea_stage5_backtranslate_w_lid.ipynb` | see MT settings below |
| 7 TTS | `reference/TTS files/LingkodAI_DS6_DS7_Synthesis.ipynb`, function `synthesize()` | Qwen3-TTS-12Hz-1.7B-Base, ICL voice cloning, fp32 |

Reference-only, never a source of stage logic:

- `reference/app_audio_pipeline_OLD.py`, `reference/app_text_saved_convo_OLD.py`:
  earlier Streamlit demos, UI reference only. Their stage logic is stale: the LID
  encoder runs fp16, GlotLID is unpinned, and the two disagree with each other on
  NLLB decoding (one uses 5 beams plus a repetition penalty, which is NOT the
  evaluated configuration).
- `reference/colab_streamlit_launcher.ipynb`: working Colab plus cloudflared recipe.
- `reference/bea_stage4_mt_single_turn.ipynb`: single-turn variant.
- `reference/reference/DS6_augmented.xlsx`: outputs of the evaluated
  Scenario 2 run, 2,235 rows, `status` all `ok`. The golden reference for
  tests. Arrived under this path/name, not the
  `ds6_multiturn_w_lid_backtranslated.xlsx` name used elsewhere in this file
  and in DECISIONS.md; same shape (row count, `clip_id`, `final_lang`,
  `rag_answer_native`, etc.), confirmed against `scripts/build_init_set.py`'s
  and `scripts/golden_check.py`'s expectations. It carries extra evaluation
  columns beyond those (`utmos22`, `tts_rt_wer`, `judge_hyp`,
  `roundtrip_bleu`/`chrf`, `verbalized_text`) that neither script reads --
  "augmented" per HK, no claim that the columns the pipeline does use were
  altered from the original DS6 run.

## MT settings, both directions (resolved 2026-09-19, verified against both notebooks)

Identical in and out. `facebook/nllb-200-3.3B`, fp16 on CUDA and fp32 on CPU,
6.7 GB allocated. Greedy: `generate()` receives only `forced_bos_token_id` and
`max_new_tokens=256`, nothing else. Tokenizer called with `padding=True,
truncation=True`. Batch size 8, flattening chunks across rows. Sentence-boundary
chunking at 800 characters via `re.split(r"(?<=[.!?])\s+", ...)`. `eng` rows pass
through untouched with no NLLB call. Language codes: `{"ceb": "ceb_Latn",
"fil": "tgl_Latn", "eng": "eng_Latn"}`. On a whole-batch failure, retry per
chunk. A row counts as successful only if it gets back the same number of chunks
it was split into. NLLB's generation config carries `max_length=200`;
`max_new_tokens` takes precedence and transformers warns about it on every call.
That warning is expected, not a defect.

## Execution modes

The models do not fit together on an 11 GB GPU (NLLB fp16 alone is 6.7 GB;
Qwen3-TTS-1.7B fp32 takes most of a card). Two modes:

- `staged` (default, for JOJIE's 11 GB cards): phase-major over a whole scripted
  conversation, the way the evaluated DS6 run was produced. Each phase loads its
  models once, processes every turn, then frees them (`del`, `gc.collect()`,
  `torch.cuda.empty_cache()`):
  1. audio LID, ASR, text LID and routing, all turns
  2. MT in, all turns
  3. RAG, turn by turn in order (follow-up rewriting needs earlier English answers)
  4. MT out, all turns
  5. TTS, all turns
- `resident` (40 GB or larger): everything loaded once, per-turn API for an
  interactive app. Estimated 23 GB, unmeasured. Stretch goal; build only after
  `staged` passes the golden check.

## Stage notes that are easy to get wrong

- LID runs ONCE per conversation, not per turn. Bea's Stage 4 notebook states
  `final_lang` is propagated to every row by Stage 3, and MT runs per turn
  "unlike LID which only needed to decide the conversation's language once". The
  routing module locks the language after turn 1. A wrong first-turn call
  misroutes the whole conversation by design; do not add per-turn correction.
- RAG: drop LangChain. `RunnableWithMessageHistory` only stores a message list
  (human = the English query from MT in, ai = the English answer or the no-match
  fallback); a plain list reproduces it. Keep the Harrier query-only instruction
  prefix, BM25 settings, Qdrant DBSF fusion, top 10, the three character caps,
  `NO_MATCH_SENTINEL` and `PRC_CONTACT_FALLBACK` exactly.
- RAG knowledge base: `data/prc_chunks/`, 13 files, 547 chunks. The loader
  asserts both counts and unique `chunk_id` values and fails loudly otherwise.
  Filenames must have download suffixes stripped (`charter_chunks (7).json`
  becomes `charter_chunks.json`), or the `*_chunks.json` glob silently skips them.
- TTS initialism set is FROZEN to `data/init_set_ds6.json`, generated once by
  `scripts/build_init_set.py`. Never derive it at runtime. Deriving it per answer
  is NOT equivalent: the notebook regex `\b[A-Z]{2,}[0-9]*\b` does not match
  "CPAs" at all, so the singular "CPA" only enters the set by appearing elsewhere
  in the corpus, and `spell_out_initialisms` then strips the plural and finds it.
  Per-answer derivation breaks that and silently leaves acronyms unspelled.
- TTS reference clips, locked: ENG `1904.141120.022212.0152.wav`, FIL
  `0052.110908.020820.0443.wav` (0443, not 0433), CEB
  `0233.111026.023506.0490.wav`. Each padded with 0.5 s trailing silence. Clips
  plus `ref_config.json` (the `ref_text` and `language` values copied verbatim
  from the notebook's `REF_CONFIG`) come from the private HF repo named in
  `TTS_REF_REPO`, never committed. The CEB `ref_text` contains a deliberate
  manual substitution ("siyete"); keep it as is.
- TTS generation: `x_vector_only_mode=False`, `non_streaming_mode=True`,
  `max_new_tokens=4096` with one OOM retry at 2048, `language="english"` for
  `eng` and `None` for `fil` and `ceb`.
- TTS text path: `prep_tts_text`, then `ceb` through
  `ceb_full_pipeline.full_ceb_pipeline_segments(text, initialism_set)` with
  `demo_fallback=False` (the regression-verified path, never the demo shortcut),
  per-segment generation, and `ceb_segmentation.concat_segments`; `fil` through
  `verbalize_fil_prc`; `eng` through `verbalize_eng_prc`. The CEB order inside
  the pipeline is load-bearing and stays as written: segment, `verbalize_ceb_prc`,
  `apply_loanword_fixes`, `apply_shared_frontend`.
- The synthesis notebook feeds `concat_segments` a generator whose third element
  is a float, not the timing dict its signature suggests. This works because
  `concat_segments` only collects it. Preserve as-is; do not tidy.
- The TTS frontend raises on some inputs by design (three CEB rows raised in the
  DS7 run and were excluded). A raise must not crash the turn: return the text
  answer, set audio to None, record the reason.
- Never use scoring-side normalizers (`scoring_normalize*`) in synthesis.

## Environment

- One conda env on JOJIE, `lingkod-e2e`, Python 3.12, built from this repo's
  `pyproject.toml`. Never install into any existing `lingkod-*` env.
- `qwen-tts==0.1.1` pins `transformers==4.57.3` and `accelerate==1.12.0`. Every
  stage runs on that pin. Never change it to satisfy another package; stop and
  ask. Francis's RAG ran on Colab with unpinned `transformers>=4.51`, so RAG
  under 4.57.3 is untested.
- torch and torchaudio are installed per machine (the CUDA build differs) and are
  not in `pyproject.toml`. On a new GPU, first check
  `torch.cuda.get_arch_list()` contains the card's architecture (`sm_61` for the
  GTX 1080 Ti, `sm_75` for the RTX 2080 Ti).
- Use `fasttext-numpy2`, never `fasttext` or `fasttext-wheel`: upstream
  `predict()` calls `np.array(probs, copy=False)`, which NumPy 2 rejects. The
  import name is still `fasttext`, so `src/lid_text.py` needs no change.
- The text frontend and text LID are CPU-only Python (numpy, pandas, re,
  fasttext). `lingkod_stt_utils.py` imports only pandas at module level; jiwer,
  matplotlib and soundfile are lazy. `ceb_segmentation.py` imports torch lazily
  inside `synthesize_segments`. This is what makes the CPU image possible; do not
  add a module-level torch import anywhere in that path.
- Precision per model: audio-LID encoder fp32, ASR fp16, NLLB fp16, Qwen3-TTS
  fp32 (fp16 produces NaN/inf). No bf16 on pre-Ampere GPUs. RAG's bf16 settings
  are an open item.
- JOJIE runs: `PYTHONNOUSERSITE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  launched with nohup from a terminal after checking `nvidia-smi`.
- Secrets only through environment variables (`HF_TOKEN`, `TTS_REF_REPO`,
  `APP_PASSWORD`, `GPU_BACKEND_URL`), documented in `.env.example`. Never commit
  `.env`. Never bake a token into an image layer or a Fly secret from a shared
  account; the CPU image must never need `HF_TOKEN` for weights.

## Docker and Fly

- `docker/Dockerfile.cpu`: the demo image. No torch, no CUDA, no model weights,
  target size around 200 MB. Runs `app/streamlit_app.py` on port 8080. This is
  the image Fly runs.
- `docker/Dockerfile.gpu`: the full pipeline. torch comes from the cu126 wheel
  index, not PyPI. Weights are downloaded at first boot, never baked in (roughly
  45 GB across seven models).
- Neither image baked with weights or secrets. Both build from the same
  `pyproject.toml` so the dependency set cannot drift between them.
- Fly: `fly.toml`, region `sin`, `auto_stop_machines` on so an idle demo costs
  nothing. The app is password-gated with the `APP_PASSWORD` secret because the
  synthesized voice is a cloned PLD research speaker under a CC-BY-NC
  research-only licence. The README states this, and the demo page says the voice
  is a cloned research-corpus speaker.
- Fly GPU machines were retired on 2026-08-01. Do not add a GPU section to
  `fly.toml` or suggest a Fly GPU size. A future GPU host (RunPod, Modal) is
  reached through `GPU_BACKEND_URL`.

## Where things run

- HK's laptop (Pop!_OS, 6 GB GPU): write code, run CPU-only tests with small
  fakes, build and run the CPU image. Do not download or load the real models.
- JOJIE: real GPU runs via conda, launched by HK, outputs pasted back.
- Fly: the CPU image only.

## Testing

- Golden check against DS6 on three conversations, one per language, same source
  audio. `final_lang` must match exactly. For transcript, English query, English
  answer and native answer, report exact-match rate and print every diff. Greedy
  decoding can shift slightly across GPUs and library versions (RAG originally
  ran on a roughly 40 GB Colab card), so a diff is read by HK, not auto-failed.
  TTS audio is judged by listening.
- Each stage module gets a smoke test that runs on CPU with fakes.
- The frozen initialism set gets a test asserting its count and a few known
  members, so a regenerated set cannot silently change synthesis.

## Workflow

- Work on branch `e2e-integration`; open a PR to `main` at the end. Never push to `main`.
- Plan first, then code. One stage per commit; commit messages start with the stage number.
- Every script prints resolved paths and versions (python, torch, transformers,
  GPU name, VRAM) before doing any work.
- No silent fallbacks. A missing file, token or unexpected shape raises with a
  clear message. The only designed fallbacks are the RAG no-match message and the
  TTS text-only turn.
- Never claim something works without showing the command and its output. If it
  was not run, say so.
- Docs and README: no em-dashes.

## Open items: do not guess, stop and ask

See `DECISIONS.md` (gitignored: no, it is a tracked project record) for the
reasoning behind each resolved item below and what would change it. Where
`DECISIONS.md` and this file disagree, `DECISIONS.md` is newer and wins.

Resolved (2026-09-19, see `DECISIONS.md`):

1. ~~Routing~~ Decided in `DECISIONS.md` #3, superseded 2026-09-22 per Bea's
   direct review: audio LID (turn 1 only) still picks the ASR checkpoint (that
   ordering is forced), but text LID wins on disagreement for `final_lang`;
   the disagreement is still recorded as `was_overridden` and ASR is still not
   re-run. Implemented in `src/routing.py`.
2. ~~Which `charter_chunks.json` produced DS6~~ Decided in `DECISIONS.md` #4:
   use Francis's zip copy (`charter_chunks (7).json`, -> `reference/` as
   `charter_chunks_v7.json` with a note), for provenance consistency with the
   other twelve chunk files, not a judgement of correctness. Record the
   SHA-256 of whichever file the pipeline actually loads. Caveat that survives
   this decision: no byte-exact DS6 reproduction claim for charter-derived
   answers until Francis confirms which file he ran -- a claims caveat, not a
   build blocker.
3. ~~RAG on pre-Ampere GPUs~~ Decided in `DECISIONS.md` #5: JOJIE's cards are
   Turing (compute capability 7.5, no bf16 tensor cores), so
   `bnb_4bit_compute_dtype=torch.float16` on JOJIE, not
   `torch.bfloat16` as in Francis's Colab notebook. Record this as a deviation
   from the evaluated Colab run in the output. Restore bf16 if ever run on
   Ampere or newer.
4. ~~HF access~~ Decided in `DECISIONS.md` #6: do not use Bea's shared account
   login. Use a fine-grained read-only token scoped to
   `beabucayan/lingkodai-whisper-{ceb,fil,eng}`, set as `HF_TOKEN` in `.env`;
   ask Bea for real collaborator access when she is back and revoke the
   interim token then.

Still open:

5. Whether the team accepts the password gate and the cloned-voice disclosure.
6. Encoder sharing across the three ASR checkpoints (`DECISIONS.md` #2) is a
   verification gate, not a decision: the underlying claim (all three
   fine-tunes kept the frozen `openai/whisper-medium` encoder byte-for-byte)
   is plausible but unverified. Run `scripts/verify_shared_encoder.py` (CPU
   only, safe to run while JOJIE is busy) and paste its output before writing
   any sharing code. `src/asr.py` stays untouched until then -- it is still
   Bea's file.

## Target layout

```text
CLAUDE.md  README.md  pyproject.toml  fly.toml  .env.example  .dockerignore
docker/Dockerfile.cpu  docker/Dockerfile.gpu
app/streamlit_app.py            CPU demo, no torch imports anywhere
data/prc_chunks/                13 *_chunks.json
data/init_set_ds6.json          frozen, generated by scripts/build_init_set.py
models/deeper_50chunks.pt       Bea's, do not edit
src/
  audio.py lid_audio.py asr.py lid_text.py   Bea's, do not edit
  routing.py   stage 3 decision and conversation language lock
  mt.py        stages 4 and 6
  rag.py       stage 5
  tts.py       stage 7
  tts_frontend/  vendored, do not edit
  pipeline.py    conversation state, staged and resident runners
scripts/
  run_conversation.py   CLI: audio files in, per-turn JSON and WAV out
  build_init_set.py     regenerates the frozen initialism sets
  golden_check.py       DS6 comparison report
tests/
reference/   gitignored
outputs/     gitignored
```

## CLI (target)

```text
python scripts/run_conversation.py --audio TURN1.wav TURN2.wav --out outputs/RUN_NAME
```

`TURN1.wav`, `TURN2.wav` and `RUN_NAME` are placeholders.
