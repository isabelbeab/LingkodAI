# LingkodAI

[![CI](https://github.com/isabelbeab/LingkodAI/actions/workflows/ci.yml/badge.svg)](https://github.com/isabelbeab/LingkodAI/actions/workflows/ci.yml)

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

## Contents

- [Pipeline stages](#pipeline-stages) and [This repo contains](#this-repo-contains)
- [IMPORTANT: Start here](#important-start-here), including [On JOJIE](#on-jojie)
- [Running your own recordings (`input/` and `output/`)](#running-your-own-recordings-input-and-output)
- [Configure: the `.env` file](#configure-the-env-file)
- [Data and models](#data-and-models)
- [Input and output](#input-and-output)
- [The apps (`app/`)](#the-apps-app)
- [Adding documents to the knowledge base](#adding-documents-to-the-knowledge-base)
- [Execution modes](#execution-modes)
- [Docker images](#docker-images)
- [Environment](#environment) (pip extras, system packages, GPU setup details)
- [Testing](#testing), including [what has been verified](#what-has-been-verified-and-how)
- [Contributing](#contributing)

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

## This repo contains

Three artifacts, all built from this one repo:

1. **The package plus CLI** (`scripts/run_conversation.py`): audio files in,
   per-turn JSON and WAV out. This is the real system and the thing the
   golden check runs against.
2. **The CPU image**: a slim, password-gated demo container with no torch.
   It runs the text frontend and GlotLID text language identification
   live, and serves pre-rendered audio from recorded end-to-end
   conversations. It runs locally (see below); it is not hosted anywhere.
3. **The GPU image**: the full pipeline containerized. It is built and
   import-checked, but not run end to end in a container. The
   pipeline it packages is verified on a real GPU outside Docker instead: on
   JOJIE through the `lingkod-e2e` conda environment (JOJIE is a shared
   JupyterHub and cannot run Docker), and on Colab through
   `notebooks/colab_golden_check.ipynb`.

## IMPORTANT: Start here

The grey boxes in this README are commands typed into a terminal (bash, on
Linux or macOS; WSL on Windows). There is no installer or app to download:
the Python scripts under `scripts/` are how the system is used. You give
`scripts/run_conversation.py` one audio file per question, and it writes the
answers as text (JSON) and speech (WAV). To process many recordings at once,
drop them in `input/` and run `scripts/run_folder.py`; the results appear in
`output/` (see [Running your own recordings](#running-your-own-recordings-input-and-output)).

### Options / Routes

| Route | You need | Use it to |
|---|---|---|
| **Colab: your recordings** | A Google account with a GPU runtime, and a Hugging Face token (below) | Run your own recordings with no local setup: open `notebooks/colab_run_folder.ipynb` ([link](https://colab.research.google.com/github/isabelbeab/LingkodAI/blob/main/notebooks/colab_run_folder.ipynb)), choose an L4 GPU, add the token as a Colab Secret named `HF_TOKEN`, put recordings in the Google Drive folder `lingkodai_input`, and run the cells. Results land in the Drive folder `lingkodai_output` |
| **Colab: evaluated results** | The same | Reproduce the evaluated results: open `notebooks/colab_golden_check.ipynb` ([link](https://colab.research.google.com/github/isabelbeab/LingkodAI/blob/main/notebooks/colab_golden_check.ipynb)) the same way and run the cells |
| **Your own GPU machine** | Everything in "Requirements" below | Run the pipeline on your own audio, or call it from other software |
| **JOJIE** | A JOJIE account and the Hugging Face token | The same as your own GPU machine, with the differences under [On JOJIE](#on-jojie) |
| **CPU demo only** | Docker, or Python 3.12 | Show recorded conversations and the live text frontend, with no GPU or token. See [Running the CPU demo locally](#running-the-cpu-demo-locally) |

### Requirements (What you need)

- Linux with an NVIDIA GPU of **11 GB or more** and a working driver
  (`nvidia-smi` shows the card). The pipeline was verified on an RTX 2080 Ti
  (11 GB) and an L4 (24 GB); other cards are untested.
- About **60 GB of free disk**: roughly 45 GB of model weights download on
  the first run, plus the Python environment.
- `git`, Python **3.12** (conda or venv), and the system packages `ffmpeg`
  and `sox` (`sudo apt install ffmpeg sox` on Debian or Ubuntu; without
  `sudo`, `conda install -c conda-forge ffmpeg sox` into a conda
  environment).
- **Membership in the `lingkodai` Hugging Face org** and a token from your own
  account. Create it at huggingface.co > Settings > Access Tokens >
  Fine-grained, with read access to these repos only:
  `lingkodai/lingkodai-whisper-ceb`, `lingkodai/lingkodai-whisper-fil`,
  `lingkodai/lingkodai-whisper-eng`, `lingkodai/lingkodai-audio-lid-head`
  and `lingkodai/lingkodai-eval-assets`.

### On JOJIE

JOJIE is a shared JupyterHub: no `sudo`, no Docker, a GPU and a disk shared
with other users, and terminals that can disconnect during a long run. The
steps below need these changes there. They come from the team's own JOJIE
setup, checked on 2026-10-14; a new account following them from scratch
has not been tried.

- **Step 2: use the conda environment, not venv.** Build `lingkod-e2e` as
  shown under [Setting up the GPU pipeline](#setting-up-the-gpu-pipeline).
  The pipeline has only run on JOJIE's RTX 2080 Ti (`sm_75`); the GTX 1080
  Ti (`sm_61`) is untested.
- **System packages, without `sudo`.** `ffmpeg` is already installed for
  everyone. `sox` is not: with `lingkod-e2e` active, run
  `conda install -c conda-forge sox -y`. Afterwards `which ffmpeg sox`
  should print two paths.
- **Disk.** Home folders are on one drive shared by every JOJIE user (on
  2026-10-14 it was 97% full, with 876 GB free). Check `df -h ~` before the
  first run, which downloads about 45 GB of models.
- **Step 3: edit `.env` in JupyterLab, not nano.** nano crashes in JOJIE's
  terminal. JupyterLab's file browser hides names that start with a dot, so
  use File > Open from Path... and type the path to `.env`, for example
  `LingkodAI/.env` if you cloned into the folder the file browser opens in.
  Save with Ctrl+S.
- **Every new terminal**, before running anything:

  ```bash
  conda activate lingkod-e2e
  cd LingkodAI                       # your clone
  set -a; source .env; set +a
  export PYTHONNOUSERSITE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nvidia-smi                         # GPU memory in use should be near 0 MiB
  ```

  `PYTHONNOUSERSITE=1` stops Python from also loading packages from
  `~/.local`, where an earlier `pip install --user` may have left a
  different torch or transformers than the pinned ones. If `nvidia-smi`
  shows memory already in use, another process holds the GPU (possibly
  your own earlier run) and the pipeline will likely run out of memory.
- **Start long runs with `nohup`**, so they keep going if the terminal
  disconnects: the first run (it downloads the models), `run_folder.py`
  and the golden check. For example:

  ```bash
  nohup python scripts/run_folder.py > run_folder.log 2>&1 &
  tail -f run_folder.log             # Ctrl+C stops watching, not the run
  ```

  The end of the log shows whether it finished or why it stopped.

### Step by step process

1. **Get the code.** Everything below runs from inside this folder.

   ```bash
   git clone https://github.com/isabelbeab/LingkodAI.git
   cd LingkodAI
   ```

2. **Make a Python environment and install.** torch comes first, built for
   your CUDA version (cu126 below; see pytorch.org for others), then the
   pipeline. The conda version is under
   [Setting up the GPU pipeline](#setting-up-the-gpu-pipeline).

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -e ".[gpu]"
   ```

3. **Create your configuration file, `.env`,** from the template,
   `.env.example`, and put your token on the `HF_TOKEN=` line. The template
   is in this folder but hidden (`ls -a` shows it; see
   [Configure](#configure-the-env-file), which also explains every line).

   ```bash
   cp .env.example .env
   nano .env                         # or any editor: HF_TOKEN=hf_...
   set -a; source .env; set +a       # load it into this terminal
   ```

4. **Run the two one-time setup scripts.** Each prints lines to add to
   `.env`. They are printed as `export ASR_CEB=...` and so on; write them in
   `.env` without the word `export` (as `ASR_CEB=...`), then load `.env`
   again. Why the scripts exist is explained under
   [Setting up the GPU pipeline](#setting-up-the-gpu-pipeline).

   ```bash
   python scripts/patch_asr_tokenizers.py --out outputs/asr_patched   # prints the ASR_CEB, ASR_FIL, ASR_ENG lines
   python scripts/fetch_eval_assets.py --out outputs/eval_assets      # prints the TTS_REF_DIR line
   set -a; source .env; set +a
   ```

5. **Ask a question.** Any recording of one PRC question in Cebuano,
   Filipino or English works. The downloaded eval assets include 24 (list
   them with `find outputs/eval_assets/golden/audio -type f | sort`).

   ```bash
   python scripts/run_conversation.py --audio my_question.m4a --out outputs/first_run
   ```

   The first run downloads the models and takes a while; the log prints each
   phase. When it finishes, `outputs/first_run/` holds the transcript, the
   answers and the spoken answer: see [Input and output](#input-and-output).
   For more than a few recordings, use the `input/` folder instead (next
   section).

6. **Optional: reproduce the evaluated results** with the golden check
   under [Testing](#testing).

In every new terminal, run `source .venv/bin/activate` (or
`conda activate lingkod-e2e`) and `set -a; source .env; set +a` again before
running anything. Also run `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
which lets PyTorch reuse freed GPU memory better as the stages load and free
their models (the GPU Docker image and the Colab notebooks set it already).
On JOJIE, use the full list under [On JOJIE](#on-jojie).

## Running your own recordings (`input/` and `output/`)

For your own recordings, no file paths to type: put them in the `input/`
folder and run one command. After the setup in
[Start here](#important-start-here):

1. **Put recordings in `input/`.**
   - An audio file directly in `input/` is a one-question conversation.
   - A subfolder is a multi-turn conversation. Its files are the turns, in
     filename order, so number them. Follow-ups are read in the light of
     earlier answers, as in the evaluated system.

   ```text
   input/
     renew_id.m4a            one question       -> output/renew_id/
     exam_fee.mp3            one question       -> output/exam_fee/
     ctc_request/            a conversation     -> output/ctc_request/
       01_requirements.m4a
       02_how_much.m4a
   ```

   Any format ffmpeg reads works (wav, m4a, mp3, aac, ...). Each recording
   should be one question in Cebuano, Filipino or English.

2. **Run:**

   ```bash
   python scripts/run_folder.py
   ```

   `--input DIR` and `--output DIR` choose other folders.

3. **Read the results in `output/`:**
   - `output/summary.csv`: one row per question, opens in Excel or Google
     Sheets. Columns: conversation, turn, input file, status (`ok`,
     `text only: <reason>`, or `failed`), detected language, transcript,
     English question, English answer, translated answer, knowledge-base
     chunks used, the answer's WAV file, any error, and seconds taken.
   - `output/<name>/`: the full output of that conversation, the same files
     as `run_conversation.py` writes ([Input and output](#input-and-output)),
     including `turn_N.wav`, the spoken answer.

Running it again only processes recordings that have no results yet, so you
can add files to `input/` and rerun. `--force` redoes everything. If one
recording fails, the others still run, the summary shows the error, and the
command exits with an error code.

Each conversation loads the models again (the pipeline frees the GPU
between stages so it fits on an 11 GB card), so every conversation adds
model-loading time. Questions that belong together are faster as one
subfolder.

Both folders are gitignored, apart from `input/README.md`: recordings and
results never go into git. The spoken answers use the cloned research
voice (see [Running the CPU demo locally](#running-the-cpu-demo-locally)),
so keep `output/` private.

**On Colab**, `notebooks/colab_run_folder.ipynb` does the same with Google
Drive folders: it creates `MyDrive/lingkodai_input/` and
`MyDrive/lingkodai_output/`, you upload recordings to the first through
drive.google.com, and the results are written straight to the second, so
they survive the Colab session.

To show the results on the CPU demo page, copy any `output/<name>/` folder
into `demo_audio/`.

## Configure: the `.env` file

`.env` is the repo's configuration file: one `NAME=value` per line, read by
the scripts as environment variables. Create it with
`cp .env.example .env`; the template explains every line.

The template, `.env.example`, is in the repo's top folder, next to this
README. A name that starts with a dot marks a hidden file on Linux and
macOS, so `ls` and most file browsers do not show it, and your `.env` will
be hidden the same way. `ls -a` lists them, as does turning on hidden files
in a file browser (Ctrl+H in most Linux file managers, Cmd+Shift+. in the
macOS Finder). GitHub's file list shows `.env.example` normally.

`.env` holds your token, so it is gitignored and must never be committed or
shared. Nothing in the code reads `.env` by itself: load it into each new
terminal with `set -a; source .env; set +a`. The Docker images take it with
`--env-file .env` instead.

| Variable | Needed for | Where the value comes from |
|---|---|---|
| `HF_TOKEN` | Full pipeline, setup scripts | Your fine-grained read-only Hugging Face token (see [Requirements](#requirements-what-you-need)) |
| `ASR_CEB`, `ASR_FIL`, `ASR_ENG` | Full pipeline | Printed by `scripts/patch_asr_tokenizers.py`. Leave them commented out until then: an empty value breaks loading |
| `TTS_REF_DIR` | Speech output (stage 7) | Printed by `scripts/fetch_eval_assets.py` |
| `APP_PASSWORD` | CPU demo page | Your choice. The page refuses to start without it |
| `HF_HOME` | Docker only | Already set inside both images; leave it commented out elsewhere |
| `DEMO_INPUT_AUDIO_DIR` | `app/gpu_chat_demo.py` audio mode | A folder of question recordings (not in the template) |

Optional overrides, not in the template, for when a source moves. The
defaults are what the evaluated run used:

| Variable | Default |
|---|---|
| `LID_ENCODER` | `openai/whisper-medium` |
| `LID_HEAD` | Unset: the head downloads from `lingkodai/lingkodai-audio-lid-head` at a pinned commit, checksum-verified. Set it to a local `.pt` file to load that instead |
| `GLOTLID_REPO`, `GLOTLID_REVISION` | `cis-lmu/glotlid` at a pinned revision (`src/lid_text.py`) |
| `RAG_CHUNKS_DIR` | `data/prc_chunks/` |

The models, prompts, decoding settings and thresholds are deliberately
**not** configurable: they are fixed in each `src/` module to match the
evaluated run, and each module's docstring lists them.

## Data and models

### Models

| Stage | Model | Where it lives |
|---|---|---|
| 1 Audio LID | `openai/whisper-medium` encoder + the team's classification head | Encoder: public Hub. Head: `lingkodai/lingkodai-audio-lid-head` (private, pinned commit and SHA-256) |
| 2 ASR | `lingkodai/lingkodai-whisper-{ceb,fil,eng}` | Private, `lingkodai` org |
| 3 Text LID | GlotLID v3 (`cis-lmu/glotlid`, pinned revision) | Public Hub |
| 4, 6 MT | `facebook/nllb-200-3.3B` | Public Hub |
| 5 RAG | `microsoft/harrier-oss-v1-0.6b` (embeddings), `Qwen/Qwen3-4B-Instruct-2507` (4-bit, generation) | Public Hub |
| 7 TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Public Hub |

All of them download to the Hugging Face cache (`~/.cache/huggingface`, or
`HF_HOME`) on first use. Nothing needs downloading by hand, and no model
weights are stored in this repo.

Licences: NLLB-200 is CC-BY-NC-4.0 and the TTS reference voices are CC-BY-NC
(below), so the system as built is for non-commercial research use. Check
each model card before any other use.

### The test dataset (DS6)

"DS6" appears throughout this repo, in file names, the golden check and
`CHANGELOG.md`. It is the dataset the pipeline was evaluated on. The name is
the team's internal one: it was the sixth dataset in the TTS author's
numbering. It is kept because the code, the file names and the recorded
history all use it.

How it was made:

1. **Scripts.** The team generated 96 multi-turn conversations about PRC
   services with a large language model (LLM), 7 or 8 questions each, in
   Cebuano, Filipino and English. They are synthetic: generated for
   testing, not collected from real PRC enquiries. A conversation keeps the
   same ID in all three languages (`CONV-MT-001` to `CONV-MT-096`).
2. **Recordings.** Native speakers of each language, friends of the team
   and other volunteers, read the questions aloud: 96 conversations in each
   language, 2,235 recordings in all.
3. **The run.** Every recording went through the full pipeline in Scenario 2
   (language identification on), one stage at a time across the whole
   dataset, in the stage order [`staged` mode](#execution-modes) follows.
   DS6 is the record of that run: one row per recording, with its true
   language, the reference transcript and answer, what every stage
   produced (detected language, transcript, English question, chunks
   retrieved, English answer, translated answer), timings and evaluation
   scores.

`DS6_augmented.xlsx` is that table with extra columns the TTS author added
later for the speech evaluation (speech-quality and round-trip scores, and
the text as the TTS spoke it). The columns this repo reads are the original
run's. The file is in the private eval assets (below).

DS6 is used here in two places:

- The golden check reruns one DS6 conversation, `CONV-MT-001`, in all three
  languages (24 recordings) and compares every field with DS6's rows. See
  [The golden check](#the-golden-check).
- `data/init_set_ds6.json`, the frozen list of acronyms the TTS spells out,
  was built from DS6's answers.

DS7 is the same 2,235 recordings run in Scenario 1, with language
identification off. This repo implements Scenario 2 only; DS7 appears here
only as `data/init_set_ds7.json` and in `CHANGELOG.md`.

### Data in this repo

| Path | What it is |
|---|---|
| `data/prc_chunks/` | The RAG knowledge base: 13 `*_chunks.json` files, 547 chunks as evaluated; the loader takes however many files it holds. See [Adding documents](#adding-documents-to-the-knowledge-base) |
| `data/init_set_ds6.json` | The frozen set of acronyms the TTS spells out letter by letter, built once by `scripts/build_init_set.py` from [DS6](#the-test-dataset-ds6)'s answers and never derived at run time. `init_set_ds7.json` and `init_set_live.json` are the DS7 set and the union of both |
| `demo_audio/` | Empty except `.gitkeep`: recorded conversations for the CPU demo go here |

### Private data (Hugging Face dataset `lingkodai/lingkodai-eval-assets`)

Files that cannot live in a public repo, downloaded and checksum-verified by
`scripts/fetch_eval_assets.py`:

- `tts_ref/`: the three TTS reference voices, Philippine Languages Database
  speaker audio under a CC-BY-NC, research-only licence. Do not
  redistribute them.
- `golden/DS6_augmented.xlsx`: [DS6](#the-test-dataset-ds6), the evaluated
  Scenario 2 run (2,235 rows), the reference for the golden check.
- `golden/audio/`: the golden conversation `CONV-MT-001` in all three
  languages, 24 recordings.
- `MANIFEST.sha256`: a checksum for every file.

## Input and output

### Input

`scripts/run_conversation.py --audio Q1 [Q2 ...] --out DIR`

| Argument | Meaning |
|---|---|
| `--audio` | One audio file per turn, in conversation order. One file is a single question; several make a multi-turn conversation in which follow-ups ("How much is it?") are read in the light of earlier answers. Any format `ffmpeg` reads (wav, m4a, aac, mp3, ...); every file is converted to 16 kHz mono |
| `--out` | Output folder, created if missing |
| `--init-set` | Optional. The acronym set; default `data/init_set_ds6.json` |
| `--rag-data-dir` | Optional. A different knowledge-base folder; default `data/prc_chunks/` |

Each recording should be one question in Cebuano, Filipino or English.
Audio LID looks at the first 30 seconds. The evaluated recordings are
single short questions; long recordings are untested.

### Output

```text
DIR/
  manifest.json      the conversation: turn count, locked language
  turn_0.json        everything the pipeline produced for turn 1
  turn_0.wav         the spoken answer for turn 1 (24 kHz)
  turn_1.json ...
```

`manifest.json`: `n_turns`, `final_lang` (the language locked for the
conversation), `was_overridden` (true when text LID overrode audio LID), and
`turn_files`.

`turn_N.json`, one per turn, by stage:

| Stage | Fields |
|---|---|
| Input | `turn_idx`, `audio_path` |
| 1 Audio LID (turn 1 only) | `pred_lang`, `lid_confidence` |
| 2 ASR | `transcript` |
| 3 Text LID and routing | `text_lang` and `text_lid_confidence` (turn 1 only), `final_lang`, `was_overridden` |
| 4 MT in | `english_query`, `mt_in_ok`, `mt_in_failure_reason` |
| 5 RAG | `resolved_query` (a follow-up rewritten as a standalone question), `was_dependent` (whether it was a follow-up), `english_answer`, `chunk_ids` (the knowledge-base chunks used; empty when nothing matched) |
| 6 MT out | `native_answer`, `mt_out_ok`, `mt_out_failure_reason` |
| 7 TTS | `tts_ok`, `tts_failure_reason`, `has_audio`, `audio_out_sr` |

A turn with `tts_ok: false` has its text answer but no WAV, and
`tts_failure_reason` says why. That is the designed fallback, not a crash.
When nothing in the knowledge base matches, the answer is a fixed message
pointing to PRC's contact channels, and `chunk_ids` is empty.

The same output folder is what the CPU demo shows: copy it into
`demo_audio/` and it appears as a recorded conversation.

From Python, `src.pipeline.run_staged(audio_paths, init_set)` returns the
same records as objects (load `init_set` with
`src.tts.load_init_set("data/init_set_ds6.json")`).

## The apps (`app/`)

There is no `app.py` at the top level. The two Streamlit apps live in
`app/`:

| App | What it does | Needs |
|---|---|---|
| `app/streamlit_app.py` | The CPU demo: plays recorded conversations stage by stage, and runs GlotLID and the TTS text frontend live on typed text | No GPU, no token. `APP_PASSWORD` |
| `app/gpu_chat_demo.py` | A live demo: typed questions (stages 4 to 6), or recorded and microphone questions (stages 2 and 4 to 7), with every stage's output shown | A GPU with about 40 GB (it was run on an A100), the full setup above, and `DEMO_INPUT_AUDIO_DIR` |

### Running the CPU demo locally

With Docker:

```bash
docker build -f docker/Dockerfile.cpu -t lingkodai-cpu .
docker run --rm -p 8080:8080 -e APP_PASSWORD=changeme lingkodai-cpu
```

Open `http://localhost:8080`. Without Docker, in a Python 3.12 environment:

```bash
pip install -e ".[app]"
APP_PASSWORD=changeme streamlit run app/streamlit_app.py    # opens on http://localhost:8501
```

The page is password-gated: the synthesized audio in the "recorded
conversations" tab is a cloned voice from a Philippine Languages Database
research speaker, under a CC-BY-NC, research-only licence, so the demo is
not left fully public. Use a real password outside local testing. No
`HF_TOKEN` is needed; GlotLID's model (about 1.2 GB) downloads on first use
instead of being baked in.

`demo_audio/` is gitignored and ships empty by default (only `.gitkeep`
committed), so the "recorded conversations" tab is empty on a fresh clone.
Fill it with `scripts/run_conversation.py --out` folders, copied in
verbatim (with Docker, before building the image). Nothing errors without
them; the tab just says so.

### Running the live GPU demo

After the full setup ([Start here](#important-start-here)):

```bash
pip install -e ".[gpu,app]"
export DEMO_INPUT_AUDIO_DIR=outputs/eval_assets/golden/audio   # or any folder of question recordings
streamlit run app/gpu_chat_demo.py
```

It loads each model on first use and keeps it loaded, so the first question
is slow and later ones are not. The page has **no password gate**, and its
audio mode speaks in the cloned research voice: keep it on a private
network, or share any tunnel URL privately and stop it after use.

## Adding documents to the knowledge base

The first four parts below are the RAG author's method for building the
knowledge base, as they wrote it. The last part lists what the current code
checks when it loads the files.

### Data Ingestion

Given that the documents are in different formats (i.e. text PDFs, scanned
PDFs, images), utilizing ChatGPT (GPT-5.6 Thinking) is the recommended
approach to extract structured information from the PRC documents. To ensure
data quality, the extracted information was manually reviewed and verified
against the original PRC documents before being incorporated into the
knowledge base. Below is the prompt used:

_"Please convert this to text verbatim."_

### Chunking

The documents generally follow a certain structure or pattern, and there are
instances where documents include tables. As such, a document
structure-based chunking strategy is the recommended approach to ensure that
chunks are a complete, self-contained semantic unit. For cases where
documents include tables, key-value encoding was used such that each row of
the table is encoded as a structured key-value block. Below are recommended
approaches for chunking documents:

| Document Type | Approach |
|---|---|
| FAQs | Chunk by question-answer pair |
| Operational Guidelines | Chunk by section |
| Transition Guidelines | Chunk by section |
| Tables | Chunk by key-value pairs |

Each chunk was reviewed manually to ensure correctness, then stored as a JSON
object in preparation for embedding and indexing. Refer to the
[Chunk Format](#chunk-format) for the specific fields that should be
included when building the chunk as a JSON object.

### Chunk Format

The knowledge base is stored as JSON objects with the fields described
below. It is worth noting that the `charter_chunks.json` contains additional
fields that are specific to the PRC Citizen's Charter (e.g.
`office_division`, `standard_requirements`, `steps`, `total_fee`,
`total_processing_time`) to capture the structure of its service entries.
The rest of the chunks only contain the fields described below.

| Field | Description |
|---|---|
| `chunk_id` | Unique identifier for the chunk |
| `chunk_title` | Short heading for the chunk which can be a service name, FAQ question, resolution section, matrix activity, profession or contact department |
| `retrieval_text` | Text that is embedded and used for matching with user query<br><br>Note: The `retrieval_text` for `charter_chunks.json` only contains the `chunk_title`, `chunk_description`, and `who_may_avail`. For the rest of the JSON objects, `retrieval_text` and `full_entry_text` are identical. |
| `full_entry_text` | The complete text that gets passed to the LLM as context once the chunk is selected |
| `doc_type` | Categorizes chunks as `faq`, `legal_resolution`, `cpd_matrix_entry`, `cpd_matrix_lookup`, `cpd_credit_table`, `procedural_manual`, `requirements_list`, `accreditation_form_guidance`, `contact`, `note` |
| `source_document` | Name of the PRC document that the chunk was based on |
| `source_url` | Link to the original PRC web page or PDF if available |

### Updating Knowledge Base

Once the new document is in the JSON format, add this to the
`data/prc_chunks/` folder to be included with the other existing documents.
This will be included in the embedding and indexing section of the pipeline
given that the search index is built in memory.

### What the code checks when it loads the files

`src/rag.py` checks the folder every time the RAG stage loads, and stops
with an error rather than guess:

- It reads every file named `*_chunks.json` in `data/prc_chunks/` (or
  `RAG_CHUNKS_DIR`), however many there are. Adding a document needs no
  code change.
- Any other `.json` file in the folder stops the run with an error naming
  it. A browser download often renames a file to something like
  `new_chunks (2).json`; rename it to end in `_chunks.json` (here
  `new_chunks.json`). Files that are not `.json`, such as notes, are
  ignored.
- Each file is one JSON list of chunk objects, and not empty.
- Every `chunk_id` must be unique across all the files.
- The code reads four fields, and every chunk needs all of them:
  `chunk_id`, `chunk_title`, `retrieval_text` and `full_entry_text`. It
  also reads `total_processing_time` and `total_fee` when a chunk has them
  (the Citizen's Charter entries) and adds them to the answer prompt as the
  official totals. Every other field is there for people and is not read by
  the code, so a missing or extra one does not break loading.
- It prints each file with its number of chunks, then the total. When the
  totals are not the evaluated knowledge base's (13 files, 547 chunks), it
  also prints a note that answers may differ from [DS6](#the-test-dataset-ds6).

After adding a document:

- To check that it is found, ask a question it answers and look for its
  `chunk_id` in that turn's `chunk_ids` [output field](#output).
- `pytest tests/test_rag.py` runs the same checks on `data/prc_chunks/`
  without loading any model.
- The GPU Docker image copies `data/prc_chunks/` when it is built, so
  rebuild it to include the new document.
- New documents are a change from the evaluated knowledge base: record them
  in `CHANGELOG.md`, and expect golden-check answers that cite them to
  differ from DS6.

## Execution modes

The full model set does not fit together on an 11GB GPU (NLLB-3.3B fp16
alone is 6.7GB; Qwen3-TTS-1.7B fp32 takes most of what's left). Two modes:

- **`staged`** (default, for an 11GB card, e.g. JOJIE): phase-major over a
  whole scripted conversation, the way the evaluated
  [DS6](#the-test-dataset-ds6) run was produced. Each phase loads its
  models once, processes every turn, then frees them (`del`,
  `gc.collect()`, `torch.cuda.empty_cache()`):
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

## Docker images

Neither image contains model weights or secrets, and both install from the
same `pyproject.toml`. GitHub Actions builds and checks both on every push
and pull request (see [Automatic checks](#automatic-checks-github-actions)).

- **CPU image** (`docker/Dockerfile.cpu`): see
  [Running the CPU demo locally](#running-the-cpu-demo-locally).
- **GPU image** (`docker/Dockerfile.gpu`):

  ```bash
  docker build -f docker/Dockerfile.gpu -t lingkodai-gpu .
  ```

  This build needs no GPU and only import-checks the pipeline (no CUDA
  touched, no weights loaded). It has not been run end to end in a
  container: the GPU machines this project has used, JOJIE (a shared
  JupyterHub) and Colab, cannot run Docker. Running it needs a host with an
  NVIDIA driver and the NVIDIA Container Toolkit. Weights are not baked into
  the image (roughly 45 GB across seven models) and download to `/data/hf`
  on first boot. The two one-time steps under
  [Setting up the GPU pipeline](#setting-up-the-gpu-pipeline) still apply
  inside the container: run them with their `--out` under the mounted
  `/data` volume so they persist across runs. `docker/Dockerfile.gpu`'s
  header shows the commands.

## Environment

The pip extras, defined in `pyproject.toml`:

- base (no extra): the CPU-only, text-only core -- the TTS text frontend,
  text language identification, and the chunk loader. No torch.
- `app`: adds Streamlit, for the demos.
- `gpu`: the full pipeline. torch and torchaudio are deliberately absent
  from `pyproject.toml` (the correct CUDA build differs per machine) --
  install them first, matching the target GPU, then `pip install -e ".[gpu]"`.
- `eval`: scoring libraries for the golden check.
- `dev`: pytest.

System packages: the full pipeline also needs `ffmpeg` on `PATH`
(`src/audio.py` shells out to it to resample every input to 16 kHz mono) and
`sox` (`qwen_tts` warns on import without it). On Debian or Ubuntu:
`sudo apt install ffmpeg sox`. Without `sudo`, install whichever is missing
into the conda environment, for example `conda install -c conda-forge sox`
(on JOJIE, `ffmpeg` is already installed and only `sox` is needed).
`docker/Dockerfile.gpu` installs both.

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
`sm_75` for an RTX 2080 Ti). The verified JOJIE runs used an RTX 2080 Ti
with torch 2.14.0+cu126 and transformers 4.57.3; the GTX 1080 Ti has never
run the pipeline. On a new terminal, run `conda activate lingkod-e2e` again
and check the prompt shows it before launching anything (on JOJIE, run the
full list under [On JOJIE](#on-jojie)).

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

### Automatic checks (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request, on GitHub's
free CPU runners:

1. **Tests**: the full `pytest` suite above.
2. **CPU image**: builds `docker/Dockerfile.cpu`, starts the container, and
   fails unless the page answers on port 8080 within two minutes.
3. **GPU image**: builds `docker/Dockerfile.gpu` (including its import
   check), then imports every pipeline module inside the container and runs
   the CLI's `--help`. The runners have no GPU, so this proves the image
   builds and the code loads, not that inference works.

A red cross next to a commit or pull request on GitHub means one of these
failed; the Actions tab shows which step, with its log.

### The golden check

The acceptance test is the golden check: three real conversations (one per
language, same source audio) run through the full pipeline on a GPU and
compared against `DS6_augmented.xlsx`, the evaluated Scenario 2 reference
run ([DS6](#the-test-dataset-ds6)). It needs the GPU setup above, including
both one-time steps. The audio paths recorded in DS6 are absolute JOJIE
paths, so `--audio-root-map` remaps them onto the downloaded copy;
`scripts/fetch_eval_assets.py` prints the full command with the right paths
filled in:

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

It takes a while (every model loads once per language). On a shared machine
such as JOJIE, set up the terminal as under [On JOJIE](#on-jojie), then put
`nohup` before `python scripts/golden_check.py` and `> golden.log 2>&1 &`
after the last argument. When it ends, this prints just the results, without
the diffs:

```bash
sed -n '/=== final_lang ===/,/=== diffs ===/p' golden.log
```

If `final_lang` does not match, the script ends with an error and the last
lines of the log say so.

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

Known issue, Cebuano audio only: at each pause between clauses, sentences
or list items, the voice shifts slightly and the faint background noise
drops out for a moment. Cebuano answers are split into those pieces and
each piece is generated separately (`_synth_ceb` in `src/tts.py`). Each
generation imitates the reference voice afresh, with its own random
variation, and the vendored `concat_segments` joins the pieces with pure
digital silence: 0.20 s after a clause, 0.45 s after a sentence, 0.60 s
after a list item. English and Filipino answers are generated in one piece,
so they do not have it. The evaluated DS7 audio was made the same way, so
it is kept; see `CHANGELOG.md`, 2026-10-14.

### What has been verified, and how

| What | How | Where and when | Result |
|---|---|---|---|
| Every stage's code | The `pytest` suite, CPU, fake models | Before every merge; GitHub Actions from 2026-10-12 | All pass |
| Full pipeline with the real models | Golden check against DS6 | JOJIE (conda, RTX 2080 Ti): 2026-09-22, 09-23 and 10-14. Colab (L4): 2026-09-29 | `final_lang` 3/3 from 09-23 on (09-22 led to the routing fix in `CHANGELOG.md`); text diffs read and recorded in `CHANGELOG.md`. TTS: Colab 24/24 synthesized, one runaway; JOJIE 10-14 judged good by listening, apart from the Cebuano joins (both known issues above) |
| ASR checkpoints in the `lingkodai` org | Transcripts compared with the original checkpoints, 24 clips | JOJIE, 2026-09-29 | 24/24 identical |
| Private eval assets | Every file checked against `MANIFEST.sha256` | JOJIE and Colab, 2026-09-29 | 28/28 match |
| CPU image | Built and started; page served | Locally, 2026-09-22 and 09-28; GitHub Actions from 2026-10-12 | Works |
| GPU image | Built; imports checked | Locally, 2026-09-22; GitHub Actions from 2026-10-12 | Builds. **Never run with a GPU inside Docker** |
| `scripts/run_folder.py` and `notebooks/colab_run_folder.ipynb` (added 2026-10-12) | Tests with a fake pipeline: folder rules, skipping finished work, failures, the summary | Every change, GitHub Actions. Real models: the Colab notebook on an L4, 2026-10-12, one single question and one 2-turn conversation, all Cebuano | Tests pass. Colab run: 3/3 questions answered with audio, audio judged clear by listening; 344 s and 422 s per conversation (models reload per conversation). Answer-quality notes recorded in `CHANGELOG.md` |

What would close the last gap: one golden-check run inside the GPU image on
a Docker host with an NVIDIA GPU, recorded in `CHANGELOG.md` like the Colab
run.

## Contributing

- `src/audio.py`, `src/lid_audio.py`, `src/asr.py` and `src/lid_text.py`
  belong to the ASR/LID author. Do not edit them; a workaround
  belongs in a new file.
- `src/tts_frontend/` is vendored verbatim from JOJIE and must not be
  edited either. See `src/tts_frontend/VENDORED.md`.
- New stage modules are ports, not redesigns: same models, prompts,
  constants, decoding settings, thresholds and fallback messages as their
  source notebook. Prompts must be byte-identical to the source.
- Work happens on a feature branch; PRs to `main` only, never a direct
  push. One stage per commit; commit messages start with the stage
  number. The GitHub Actions checks must pass before a merge.
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
