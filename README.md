# LingkodAI

LingkodAI is a proof-of-concept multilingual voice assistant that answers
spoken questions about Philippine government services. As a start, its
knowledge base covers Professional Regulation Commission (PRC) services. A
citizen speaks Cebuano (`ceb`), Filipino (`fil`) or English (`eng`); the
system replies with synthesized speech in the same language. Scope: Scenario 2 (language identification on), multi-turn
conversations.

This is a containerized, reproducible research pipeline with a CPU demo
front end, not a production system. Each stage module documents its own settings and
source of truth in its docstring (`src/mt.py`, `src/rag.py`, `src/tts.py`,
`src/routing.py`, ...); `CHANGELOG.md` records the reasoning behind
decisions that took discussion, in the order they were made.

## Pipeline stages

Every turn runs through, in order:

1. Audio LID - predict the spoken language from the audio.
2. ASR - transcribe with that language's fine-tuned Whisper.
3. Text LID and routing - confirm or override the language, lock it for
   the conversation.
4. MT in - native language to English (`eng` passes through unchanged).
5. RAG - rewrite follow-ups, hybrid retrieval + chunk selection, English
   answer.
6. MT out - English answer to the conversation language (`eng` passes
   through).
7. TTS - speak the native answer.

Language identification runs once per conversation, at turn 1, and is then
locked for every later turn -- not re-decided per turn.

## What ships

Three artifacts, all built from this one repo:

1. **The package plus CLI** (`scripts/run_conversation.py`): audio files in,
   per-turn JSON and WAV out. This is the real system and the thing the
   golden check runs against.
2. **The CPU image**: a slim, password-gated demo container with no torch.
   It runs the text frontend and GlotLID text language identification
   live, and serves pre-rendered audio from recorded end-to-end
   conversations. It runs locally (see below); it is not hosted anywhere.
3. **The GPU image**: the full pipeline containerized. It is built and
   import-checked locally, but not run end to end in a container. The
   pipeline it packages is verified on a real GPU outside Docker instead: on
   JOJIE through the `lingkod-e2e` conda environment (JOJIE is a shared
   JupyterHub and cannot run Docker), and on Colab through
   `notebooks/colab_golden_check.ipynb`.

## Execution modes

The full model set does not fit together on an 11GB GPU (NLLB-3.3B fp16
alone is 6.7GB; Qwen3-TTS-1.7B fp32 takes most of what's left). Two modes:

- **`staged`** (default, for an 11GB card, e.g. JOJIE): phase-major over a
  whole scripted conversation, the way the evaluated DS6 run was produced.
  Each phase loads its models once, processes every turn, then frees them
  (`del`, `gc.collect()`, `torch.cuda.empty_cache()`):
  1. audio LID and text LID on the first turn, ASR on all turns, routing
  2. MT in, all turns
  3. RAG, turn by turn in order (follow-up rewriting needs earlier English
     answers)
  4. MT out, all turns
  5. TTS, all turns

  Implemented in `src/pipeline.py` (`run_staged`). This is the runner
  `scripts/run_conversation.py` and the golden check both use.
- **`resident`** (40GB or larger): everything loaded once, a per-turn API for
  an interactive app. A stretch goal, not implemented in the pipeline
  runner. `app/gpu_chat_demo.py` is a standalone live demo of this shape
  for a large-VRAM host, outside the shipped artifacts.

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
routing overrode audio LID), one `turn_N.json` per turn (transcript, English
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
no weights loaded). It has not been run end to end in a container: the GPU
machines this project has used, JOJIE (a shared JupyterHub) and Colab,
cannot run Docker. Real GPU runs happen outside Docker instead, through the
`lingkod-e2e` conda environment on JOJIE or the Colab notebook, both
installing from the same `pyproject.toml`. Weights are not baked into the image (roughly 45 GB across
seven models) and download to `/data/hf` on first boot. The two one-time
steps under Setting up the GPU pipeline below still apply inside the
container: run them with their `--out` under the mounted `/data` volume so
they persist across runs. `docker/Dockerfile.gpu`'s header shows the
commands.

## Environment

Copy `.env.example` to `.env` and fill it in; `.env` is gitignored and must
never be committed. See that file for what each variable is for. Nothing in
the code reads `.env` by itself: load it into each new shell before running
a script, with

```bash
set -a; source .env; set +a
```

(The Docker images take it with `--env-file .env` instead.)

Four pip extras, defined in `pyproject.toml`:

- base (no extra): the CPU-only, text-only core -- the TTS text frontend,
  text language identification, and the chunk loader. No torch.
- `app`: adds Streamlit, for the CPU demo.
- `gpu`: the full pipeline. torch and torchaudio are deliberately absent
  from `pyproject.toml` (the correct CUDA build differs per machine) --
  install them first, matching the target GPU, then `pip install -e ".[gpu]"`.
- `eval`: scoring libraries for the golden check.
- `dev`: pytest.

System packages: the full pipeline also needs `ffmpeg` on `PATH`
(`src/audio.py` shells out to it to resample every input to 16 kHz mono) and
`sox` (`qwen_tts` warns on import without it). On Debian or Ubuntu:
`sudo apt install ffmpeg sox`. `docker/Dockerfile.gpu` installs both.

### Setting up the GPU pipeline

On JOJIE the environment is one conda env, `lingkod-e2e`, Python 3.12, built
from this repo's `pyproject.toml`. Never install into any other `lingkod-*`
env. From the repo root:

```bash
conda create -n lingkod-e2e python=3.12 -y
conda activate lingkod-e2e
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[gpu,eval]"
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

The last line must list the card's architecture (`sm_61` for a GTX 1080 Ti,
`sm_75` for an RTX 2080 Ti). The verified JOJIE runs used torch
2.14.0+cu126 with transformers 4.57.3. On a new terminal, run
`conda activate lingkod-e2e` again and check the prompt shows it before
launching anything.

Then two one-time steps, both needing `HF_TOKEN` in the environment (see
`.env.example` for what the token must be able to read):

1. **Patch the ASR tokenizers.** The three fine-tuned Whisper checkpoints do
   not load as published under the pinned transformers 4.57.3 (see
   `CHANGELOG.md`, 2026-09-22). This script builds patched local copies
   (weights symlinked from the Hugging Face cache, not duplicated), verifies
   them, and prints three `export ASR_...=` lines. Add them to `.env`
   without the `export` (as `ASR_CEB=...` and so on), then reload `.env`.

   ```bash
   python scripts/patch_asr_tokenizers.py --out outputs/asr_patched
   ```

2. **Fetch the private eval assets.** The TTS reference clips, the DS6
   reference run and the golden conversation's source audio cannot live in
   this public repo. They are in the private Hugging Face dataset repo
   `lingkodai/lingkodai-eval-assets`, readable by `lingkodai` org members.
   This script downloads them, checks every file against the repo's
   `MANIFEST.sha256`, and prints the `TTS_REF_DIR` value for `.env` (add it,
   then reload `.env`) and the exact golden-check command.

   ```bash
   python scripts/fetch_eval_assets.py --out outputs/eval_assets
   ```

The TTS reference clips are Philippine Languages Database speaker audio
under a CC-BY-NC, research-only licence. Do not redistribute them.

The models themselves (roughly 45 GB across seven) download from the Hugging
Face Hub on first use. Apart from the three ASR checkpoints, all are public
and ungated.

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
run. It needs the GPU setup above, including both one-time steps. The audio
paths recorded in DS6 are absolute JOJIE paths, so `--audio-root-map`
remaps them onto the downloaded copy; `scripts/fetch_eval_assets.py` prints
the full command with the right paths filled in:

```bash
python scripts/golden_check.py \
    --ds6 outputs/eval_assets/golden/DS6_augmented.xlsx \
    --conversation-id CONV-MT-001 \
    --audio-root-map /home2/msds2026/ibucayan/Capstone/PRC_synthetic_qa=$PWD/outputs/eval_assets/golden/audio \
    --out outputs/golden_check
```

`final_lang` must match exactly. For transcript, English query, English
answer and native answer, the script reports an exact-match rate and prints
every diff -- it does not fail the run on a text mismatch, since greedy
decoding can shift slightly across GPUs and library versions. A diff is
meant to be read by a person. TTS audio is judged by listening, not compared
by the script.

It takes a while (every model loads once per language). On a shared machine,
launch it with `nohup ... > golden.log 2>&1 &` after checking `nvidia-smi`,
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set.

Without JOJIE, `notebooks/colab_golden_check.ipynb` runs the same golden
check on a Colab GPU (L4 preferred; a T4 is untested but has more memory
than the 11 GB cards the pipeline runs on at JOJIE): it installs, runs both
one-time steps, runs the check, saves the results to Google Drive and plays
each turn's audio. It needs an `HF_TOKEN` Colab Secret with the same access
as above.
[Open it in Colab](https://colab.research.google.com/github/isabelbeab/LingkodAI/blob/main/notebooks/colab_golden_check.ipynb).
It was run end to end on 2026-09-29 on an NVIDIA L4; the result is in
`CHANGELOG.md`.

Known issue: Qwen3-TTS occasionally fails to stop on one segment and runs
to its `max_new_tokens` cap (4096 tokens, about 5.7 minutes of garbled
audio). The turn's text answer is unaffected. It happened once in 24 turns
on Colab and was not noticed in the JOJIE runs; see `CHANGELOG.md`,
2026-09-29.

## Contributing

- `src/audio.py`, `src/lid_audio.py`, `src/asr.py`, `src/lid_text.py` and
  `models/` belong to the ASR/LID author. Do not edit them; a workaround
  belongs in a new file.
- `src/tts_frontend/` is vendored verbatim from JOJIE and must not be
  edited either. See `src/tts_frontend/VENDORED.md`.
- New stage modules are ports, not redesigns: same models, prompts,
  constants, decoding settings, thresholds and fallback messages as their
  source notebook. Prompts must be byte-identical to the source.
- Work happens on a feature branch; PRs to `main` only, never a direct
  push. One stage per commit; commit messages start with the stage
  number.
- No silent fallbacks: a missing file, token, or unexpected shape raises
  with a clear message rather than guessing. The only designed fallbacks
  anywhere in the pipeline are RAG's no-match message (`src/rag.py`) and
  TTS's text-only turn (`src/tts.py`'s `synthesize_turn`).

See `CHANGELOG.md` for the reasoning behind specific decisions, and each
stage module's own docstring for its settings and source of truth.

## Use of AI in building this prototype

This prototype was built with extensive help from Anthropic's Claude AI and
Claude Code, which we used as our main LLM throughout. We note this for
transparency given we are not software engineers. As such, most of the
packaging, containerization and pipeline engineering was learned along the
way with Claude's help. Research and evaluation decisions were ours, and all
code was checked against the source notebooks and outputs.
