# LingkodAI

LingkodAI answers spoken questions about Philippine Professional Regulation
Commission (PRC) services. A citizen speaks Cebuano (`ceb`), Filipino (`fil`)
or English (`eng`); the system replies with synthesized speech in the same
language. Scope: Scenario 2 (language identification on), multi-turn
conversations.

This is a containerized, reproducible pipeline with a hosted front end, not a
production deployment. Each stage module documents its own settings and
source of truth in its docstring (`src/mt.py`, `src/rag.py`, `src/tts.py`,
`src/routing.py`, ...); `CHANGELOG.md` records the reasoning behind
decisions that took discussion, in the order they were made.

## Pipeline stages

Every turn runs through, in order:

1. Audio LID -- predict the spoken language from the audio.
2. ASR -- transcribe with that language's fine-tuned Whisper.
3. Text LID and routing -- confirm or override the language, lock it for
   the conversation.
4. MT in -- native language to English (`eng` passes through unchanged).
5. RAG -- rewrite follow-ups, hybrid retrieval, LLM chunk selection,
   English answer.
6. MT out -- English answer to the conversation language (`eng` passes
   through).
7. TTS -- speak the native answer.

Language identification runs once per conversation, at turn 1, and is then
locked for every later turn -- not re-decided per turn.

## What ships

Three artifacts, all built from this one repo:

1. **The package plus CLI** (`scripts/run_conversation.py`): audio files in,
   per-turn JSON and WAV out. This is the real system and the thing the
   golden check runs against.
2. **The CPU image and the Fly app**: a slim container with no torch,
   hosting a public demo. It runs the text frontend and GlotLID text
   language identification live, and serves pre-rendered audio from
   recorded end-to-end conversations. This is what gets a URL.
3. **The GPU image**: the full pipeline containerized. It is built and
   import-checked locally, but verified for real only on JOJIE through the
   `lingkod-e2e` conda environment, because JOJIE is a shared JupyterHub and
   cannot run Docker.

## Execution modes

The full model set does not fit together on an 11GB GPU (NLLB-3.3B fp16
alone is 6.7GB; Qwen3-TTS-1.7B fp32 takes most of what's left). Two modes:

- **`staged`** (default, for an 11GB card, e.g. JOJIE): phase-major over a
  whole scripted conversation, the way the evaluated DS6 run was produced.
  Each phase loads its models once, processes every turn, then frees them
  (`del`, `gc.collect()`, `torch.cuda.empty_cache()`):
  1. audio LID, ASR, text LID and routing, all turns
  2. MT in, all turns
  3. RAG, turn by turn in order (follow-up rewriting needs earlier English
     answers)
  4. MT out, all turns
  5. TTS, all turns

  Implemented in `src/pipeline.py` (`run_staged`). This is the runner
  `scripts/run_conversation.py` and the golden check both use.
- **`resident`** (40GB or larger): everything loaded once, a per-turn API for
  an interactive app. A stretch goal, not implemented here -- build it only
  after `staged` passes the golden check. `app/gpu_backend_api.py` is a
  prototype of this shape for a large-VRAM host.

## Quickstart: the CLI

```bash
python scripts/run_conversation.py --audio TURN1.wav TURN2.wav --out outputs/RUN_NAME
```

`TURN1.wav`, `TURN2.wav` and `RUN_NAME` are placeholders -- pass as many
`--audio` files as the conversation has turns, in order. This needs the
`gpu` extra (see Environment below) and, for a practical runtime, an actual
GPU: NLLB and Qwen3-TTS on CPU would technically run but far too slowly to
be useful. Never run this against real models on a laptop with no GPU --
develop and test with fakes locally (see Testing below), and run for real
only on a machine that actually has one.

Output: one `manifest.json` (turn count, locked `final_lang`, whether
routing overrode text LID), one `turn_N.json` per turn (transcript, English
query, English answer, native answer, chunk ids, and whether each stage
succeeded), and one `turn_N.wav` per turn where synthesis succeeded.

## Running the CPU demo locally

```bash
docker build -f docker/Dockerfile.cpu -t lingkodai-cpu .
docker run --rm -p 8080:8080 -e APP_PASSWORD=changeme lingkodai-cpu
```

Open `http://localhost:8080`. The page is password-gated: the synthesized
audio in the "recorded conversations" tab is a cloned voice from a Philippine
Languages Database research speaker, under a CC-BY-NC, research-only
licence, so the demo is not left fully public. No `HF_TOKEN` is needed for
this image; GlotLID's model (about 1.2 GB) downloads on first use instead of
being baked in.

`demo_audio/` is gitignored and ships empty by default (only `.gitkeep`
committed). Populate it with directories, each one a
`scripts/run_conversation.py --out` output copied in verbatim, before
building the image if you want the "recorded conversations" tab to show
anything. Without it, that tab just says so; nothing errors.

## Building the GPU image

```bash
docker build -f docker/Dockerfile.gpu -t lingkodai-gpu .
```

This build needs no GPU and only import-checks the pipeline (no CUDA touched,
no weights loaded). It has not been run end to end in a container: JOJIE, the
only machine with the model weights and the right GPU, is a shared
JupyterHub and cannot run Docker. Real GPU runs happen there through the
`lingkod-e2e` conda environment instead, built from this repo's
`pyproject.toml`. Weights are not baked into the image (roughly 45 GB across
seven models) and download to `/data/hf` on first boot.

## Environment

Copy `.env.example` to `.env` and fill it in; `.env` is gitignored and must
never be committed. See that file for what each variable is for.

Three conda/pip extras, defined in `pyproject.toml`:

- base (no extra): the CPU-only, text-only core -- the TTS text frontend,
  text language identification, and the chunk loader. No torch.
- `app`: adds Streamlit, for the CPU demo.
- `gpu`: the full pipeline. torch and torchaudio are deliberately absent
  from `pyproject.toml` (the correct CUDA build differs per machine) --
  install them first, matching the target GPU, then `pip install -e ".[gpu]"`.

On JOJIE: one conda env, `lingkod-e2e`, Python 3.12, built from this repo's
`pyproject.toml`. Never install into any other `lingkod-*` env.

## Testing

```bash
# torch first (CPU wheel is enough -- the test suite never loads a real
# GPU model, only fakes), then the rest:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[app,gpu,eval,dev]"
pytest tests/
```

Every stage module has a smoke test that runs on CPU with fakes -- no real
model is downloaded or loaded by the test suite. `tests/test_init_set.py`
asserts the frozen initialism set's count and a few known members, so a
regenerated set cannot silently change synthesis without a test noticing.

The acceptance test is the golden check: three real conversations (one per
language, same source audio) run through the full pipeline on a GPU and
compared against `DS6_augmented.xlsx`, the evaluated Scenario 2 reference
run.

```bash
python scripts/golden_check.py --ds6 path/to/DS6_augmented.xlsx --out outputs/golden_check
```

`final_lang` must match exactly. For transcript, English query, English
answer and native answer, the script reports an exact-match rate and prints
every diff -- it does not fail the run on a text mismatch, since greedy
decoding can shift slightly across GPUs and library versions. A diff is
meant to be read by a person. TTS audio is judged by listening, not compared
by the script.

## Fly deployment

`fly.toml` builds `docker/Dockerfile.cpu`, region `sin`, with
`auto_stop_machines` on so an idle demo costs nothing.

```bash
fly launch --no-deploy --copy-config --name lingkodai-demo
fly secrets set APP_PASSWORD=pick-a-real-one
fly deploy
```

Fly retired GPU machines on 2026-08-01, so there is no GPU host behind this
deploy. A `GPU_BACKEND_URL`-based "live pipeline" tab and matching
`app/gpu_backend_api.py` backend were built and verified end to end against
a Colab-hosted backend, then deliberately deprecated as more moving parts
than the presentation needed -- the CPU app's job is the pre-rendered
recorded-conversations demo above, not a live GPU round trip. The code is
left in the repo (`app/gpu_backend_api.py`, the "Live pipeline" tab in
`app/streamlit_app.py`) in case a real GPU host is worth wiring up again
later, but `GPU_BACKEND_URL` is currently unset.

## Contributing

- `src/audio.py`, `src/lid_audio.py`, `src/asr.py`, `src/lid_text.py` and
  `models/` belong to Bea. Do not edit them; a workaround belongs in a new
  file.
- `src/tts_frontend/` is vendored verbatim from JOJIE and must not be
  edited either. See `src/tts_frontend/VENDORED.md`.
- New stage modules are ports, not redesigns: same models, prompts,
  constants, decoding settings, thresholds and fallback messages as their
  source notebook. Prompts must be byte-identical to the source.
- Work happens on branch `e2e-integration`; PRs to `main` only, never a
  direct push. One stage per commit; commit messages start with the stage
  number.
- No silent fallbacks: a missing file, token, or unexpected shape raises
  with a clear message rather than guessing. The only designed fallbacks
  anywhere in the pipeline are RAG's no-match message (`src/rag.py`) and
  TTS's text-only turn (`src/tts.py`'s `synthesize_turn`).

See `CHANGELOG.md` for the reasoning behind specific decisions, and each
stage module's own docstring for its settings and source of truth.
