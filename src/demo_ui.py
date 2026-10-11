"""Display and file helpers for the demo notebooks (notebooks/demo_*.ipynb).

Torch-free on purpose: the replay notebook uses only this module, on a
runtime with no GPU and only the base dependencies. IPython and ipywidgets
are imported only when something is displayed.

A saved session is a folder in scripts/run_conversation.py's layout:
turn_N.json (pipeline.TurnRecord.to_json_dict() plus the demo fields below),
turn_N.wav when the turn has audio, and manifest.json. A demo session also
keeps the spoken question as turn_N_question.wav. scripts/golden_check.py's
output (<clip_id>.json and .wav, plus report.json) reads the same way.

Demo fields in a turn JSON, all optional:
    input_mode    "audio" or "text"
    typed_text    the question, in text mode
    lang_source   how final_lang was chosen, e.g. "GlotLID 0.97"
    chunk_titles  {chunk_id: title} for the chunks RAG used
    stage_s       {stage: seconds}
"""

from __future__ import annotations

import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = REPO_ROOT / "data" / "prc_chunks"
LANG_NAMES = {"ceb": "Cebuano", "fil": "Filipino", "eng": "English"}

# Same wording as the Streamlit demos (app/streamlit_app.py), which
# stakeholders accepted on 2026-09-28 (CHANGELOG.md).
VOICE_DISCLOSURE = (
    "Synthesized audio on this page uses a cloned voice from a Philippine "
    "Languages Database research speaker, under a CC-BY-NC, research-only "
    "licence. It is not a real PRC official."
)

# The golden conversation's audio: DS6 records JOJIE paths under this prefix,
# and the eval assets keep the layout below it under golden/audio/ (the same
# constants as scripts/fetch_eval_assets.py).
DS6_AUDIO_ROOT = "/home2/msds2026/ibucayan/Capstone/PRC_synthetic_qa"
GOLDEN_CONVERSATION = "CONV-MT-001"

TURN_FILE_RE = re.compile(r"^(turn_(?P<n>\d+)|(?P<lang>[a-z]{3})_(?P<conv>\d+)_mt_(?P<turn>\d+))\.json$")
PLAYABLE = {".wav", ".mp3", ".ogg", ".flac"}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def load_env_file(path) -> list[str]:
    """Set KEY=VALUE lines from a .env file that the environment does not
    already set. Returns the names set; values are never printed. Empty
    values are skipped: src/asr.py would read an empty ASR_* as a path."""
    path = Path(path)
    if not path.is_file():
        return []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        names.append(key)
    return names


# ---------------------------------------------------------------------------
# Knowledge base and golden clips
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _titles(chunk_dir: str) -> dict[str, str]:
    titles = {}
    for path in sorted(Path(chunk_dir).glob("*_chunks.json")):
        for chunk in json.loads(path.read_text(encoding="utf-8")):
            titles[chunk["chunk_id"]] = " ".join(str(chunk.get("chunk_title") or "").split())
    return titles


def chunk_titles(chunk_ids, chunk_dir=CHUNK_DIR) -> dict[str, str]:
    """{chunk_id: title} from the knowledge-base files; "" for an unknown id."""
    titles = _titles(str(chunk_dir))
    return {cid: titles.get(cid, "") for cid in chunk_ids}


def golden_clips(ds6_path, assets_dir) -> list[dict]:
    """The golden conversation's 24 question clips from the eval assets, in
    language and turn order, with DS6's reference transcript and English
    reference for each. Raises if a clip's audio is missing."""
    import pandas as pd

    df = pd.read_excel(ds6_path)
    rows = df[df["conversation_id"] == GOLDEN_CONVERSATION]
    audio_root = Path(assets_dir) / "golden" / "audio"
    clips, missing = [], []
    for _, r in rows.iterrows():
        raw = str(r["source_audio_path"])
        if not raw.startswith(DS6_AUDIO_ROOT):
            raise RuntimeError(f"{r['clip_id']}: audio path {raw} is not under {DS6_AUDIO_ROOT}")
        path = audio_root / raw[len(DS6_AUDIO_ROOT):].lstrip("/")
        if not path.is_file():
            missing.append(str(path))
        clips.append({
            "clip_id": r["clip_id"],
            "lang": str(r["clip_id"])[:3],
            "turn": int(str(r["turn_id"]).rsplit("T", 1)[-1]),
            "question": str(r["reference_transcript"]),
            "english": str(r["english_reference"]),
            "path": path,
        })
    if missing:
        raise FileNotFoundError(f"{len(missing)} golden clip(s) missing, e.g. {missing[:3]}; "
                                "run scripts/fetch_eval_assets.py first")
    return sorted(clips, key=lambda c: (c["lang"], c["turn"]))


# ---------------------------------------------------------------------------
# One turn as rows of text
# ---------------------------------------------------------------------------


def _conf(value) -> str:
    return f" {value:.2f}" if isinstance(value, (int, float)) else ""


def lang_source(turn: dict) -> str:
    """How the turn's language was chosen, from the demo field or, for a
    pipeline turn, from its LID fields."""
    if turn.get("lang_source"):
        return turn["lang_source"]
    parts = []
    if turn.get("pred_lang"):
        parts.append(f"audio LID {turn['pred_lang']}{_conf(turn.get('lid_confidence'))}")
    if turn.get("text_lang"):
        parts.append(f"text LID {turn['text_lang']}{_conf(turn.get('text_lid_confidence'))}")
    if turn.get("was_overridden"):
        parts.append("text LID overrode audio LID")
    return ", ".join(parts) or "locked from the first turn"


def audio_note(turn: dict) -> str:
    if turn.get("tts_ok") is None:
        return "not requested"
    if not turn["tts_ok"]:
        return f"text only: {turn.get('tts_failure_reason')}"
    note = f"{turn.get('tts_backend') or 'hf'} TTS"
    if turn.get("tts_wall_sec") is not None:
        note += f", {turn['tts_wall_sec']:.1f} s to synthesize"
    if turn.get("tts_rtf") is not None:
        note += f" (RTF {turn['tts_rtf']:.2f})"
    return note


def turn_rows(turn: dict) -> list[tuple[str, str]]:
    """(label, value) rows for one turn: question, language, the English
    pivot, sources, answers, audio and timings."""
    lang = turn.get("final_lang")
    typed = turn.get("input_mode") == "text" or (turn.get("input_mode") is None and turn.get("typed_text"))
    rows = [("You typed", turn.get("typed_text") or "") if typed else ("Heard (ASR)", turn.get("transcript") or "")]
    if lang:
        rows.append(("Language", f"{LANG_NAMES.get(lang, lang)} ({lang_source(turn)})"))
    if lang != "eng" and turn.get("english_query"):
        rows.append(("In English", turn["english_query"]))
    if turn.get("was_dependent") and turn.get("resolved_query"):
        rows.append(("Follow-up read as", turn["resolved_query"]))
    ids = turn.get("chunk_ids") or []
    titles = turn.get("chunk_titles") or chunk_titles(ids)
    sources = "; ".join(f"{cid} {titles.get(cid, '')}".strip() for cid in ids)
    rows.append(("Sources", sources or "none"))
    if lang != "eng" and turn.get("english_answer"):
        rows.append(("Answer in English", turn["english_answer"]))
    rows.append(("Answer", turn.get("native_answer") or ""))
    rows.append(("Audio", audio_note(turn)))
    if turn.get("stage_s"):
        rows.append(("Time", " | ".join(f"{k} {v:.1f} s" for k, v in turn["stage_s"].items() if v is not None)))
    return rows


def turn_html(turn: dict, title: str | None = None) -> str:
    cells = "".join(
        f"<tr><td style='vertical-align:top;padding:2px 12px 2px 0;white-space:nowrap'><b>{html.escape(k)}</b></td>"
        f"<td style='vertical-align:top;padding:2px 0'>{html.escape(str(v))}</td></tr>"
        for k, v in turn_rows(turn)
    )
    head = f"<h4 style='margin:12px 0 4px'>{html.escape(title)}</h4>" if title else ""
    return f"{head}<table style='border-collapse:collapse;text-align:left'>{cells}</table>"


def _player(audio):
    """An IPython Audio player for (array, sr) or a playable file; None for
    anything else (e.g. an .m4a path, which the browser player may not take)."""
    from IPython.display import Audio

    if audio is None:
        return None
    if isinstance(audio, (str, Path)):
        return Audio(filename=str(audio)) if Path(audio).suffix.lower() in PLAYABLE else None
    array, sr = audio
    return Audio(array, rate=sr)


def show_turn(turn: dict, audio=None, question_audio=None, title: str | None = None) -> None:
    """Display one turn: its rows, the question audio and the answer audio.
    audio and question_audio are (array, sr), a file path, or None."""
    from IPython.display import HTML, display

    display(HTML(turn_html(turn, title)))
    for label, item in (("Question audio", question_audio), ("Answer audio", audio)):
        player = _player(item)
        if player is not None:
            display(HTML(f"<i>{label}</i>"))
            display(player)


# ---------------------------------------------------------------------------
# Saved sessions
# ---------------------------------------------------------------------------


def _turn_sort_key(path: Path):
    m = TURN_FILE_RE.match(path.name)
    if m and m.group("n") is not None:
        return ("", 0, int(m.group("n")))
    return (m.group("lang"), int(m.group("conv")), int(m.group("turn")))


def find_sessions(root) -> list[Path]:
    """Folders under root (root included) holding turn JSON files, newest first."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder")
    folders = {p.parent for p in root.rglob("*.json") if TURN_FILE_RE.match(p.name)}
    return sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)


def load_session(folder) -> tuple[dict, list[dict]]:
    """(manifest or {}, turns). Each turn dict gains "_wav" and
    "_question_audio": the answer and question audio paths, or None."""
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest.get("turn_files"):
        paths = [folder / name for name in manifest["turn_files"]]
    else:
        paths = sorted((p for p in folder.glob("*.json") if TURN_FILE_RE.match(p.name)), key=_turn_sort_key)
    if not paths:
        raise FileNotFoundError(f"no turn JSON files in {folder}")
    turns = []
    for p in paths:
        turn = json.loads(p.read_text(encoding="utf-8"))
        wav, question = p.with_suffix(".wav"), p.with_name(p.stem + "_question.wav")
        turn["_wav"] = wav if wav.is_file() else None
        turn["_question_audio"] = question if question.is_file() else None
        turns.append(turn)
    return manifest, turns


def show_session(folder, limit: int | None = None) -> None:
    """Display every turn of a saved session (or a golden-check output
    folder), with a heading whenever the language changes."""
    from IPython.display import HTML, display

    manifest, turns = load_session(folder)
    head = f"{Path(folder).name}: {len(turns)} turn(s)"
    if manifest.get("tts_backend"):
        head += f", TTS {manifest['tts_backend']}"
    display(HTML(f"<h3>{html.escape(head)}</h3>"))
    last_lang = None
    for k, turn in enumerate(turns[:limit]):
        lang = turn.get("final_lang")
        if lang != last_lang:
            display(HTML(f"<h3>{html.escape(LANG_NAMES.get(lang, str(lang)))}</h3>"))
            last_lang = lang
        title = turn.get("clip_id") or f"Turn {turn.get('turn_idx', k) + 1}"
        show_turn(turn, audio=turn["_wav"], question_audio=turn["_question_audio"], title=title)


# ---------------------------------------------------------------------------
# Chat box
# ---------------------------------------------------------------------------


def bubble_html(who: str, text: str, meta: str = "") -> str:
    side = "flex-end" if who == "You" else "flex-start"
    meta_html = f"<div style='font-size:0.8em;opacity:0.7;margin-top:2px'>{html.escape(meta)}</div>" if meta else ""
    return (f"<div style='display:flex;justify-content:{side};margin:4px 0'>"
            f"<div style='max-width:80%;border:1px solid rgba(128,128,128,0.5);border-radius:8px;padding:6px 10px'>"
            f"<b>{html.escape(who)}</b><br>{html.escape(text)}{meta_html}</div></div>")


def answer_meta(turn: dict) -> str:
    """One line under the answer: language, sources and total time."""
    lang = turn.get("final_lang")
    parts = [LANG_NAMES.get(lang, str(lang))]
    if turn.get("chunk_ids"):
        parts.append("sources " + ", ".join(turn["chunk_ids"]))
    else:
        parts.append("no matching service")
    total = sum(v for v in (turn.get("stage_s") or {}).values() if v is not None)
    if total:
        parts.append(f"{total:.1f} s")
    if turn.get("tts_ok") is False:
        parts.append("no audio: " + str(turn.get("tts_failure_reason")))
    return " | ".join(parts)


class ChatBox:
    """A chat transcript with a text box, Send and New conversation buttons,
    for a session with ask_text(text, speak=...) and reset(). Errors show in
    the transcript; nothing is retried."""

    def __init__(self, session, speak: bool = True):
        import ipywidgets as w

        self.session = session
        self.log = w.Output(layout=w.Layout(border="1px solid rgba(128,128,128,0.5)", padding="6px",
                                            max_height="560px", overflow="auto"))
        self.text = w.Text(placeholder="Type a question in Cebuano, Filipino or English",
                           layout=w.Layout(width="70%"))
        self.send = w.Button(description="Send", button_style="primary")
        self.speak = w.Checkbox(value=speak, description="Speak the answer", indent=False)
        self.details = w.Checkbox(value=False, description="Show every stage", indent=False)
        self.new = w.Button(description="New conversation")
        self.status = w.HTML()
        self.send.on_click(lambda _: self.submit(self.text.value))
        self.new.on_click(lambda _: self.new_conversation())
        self.widget = w.VBox([self.log, w.HBox([self.text, self.send]),
                              w.HBox([self.speak, self.details, self.new]), self.status])

    def submit(self, text: str) -> None:
        from IPython.display import HTML, display

        text = text.strip()
        if not text:
            return
        self.send.disabled = True
        self.status.value = "<i>Working... (the first question also loads the models)</i>"
        with self.log:
            display(HTML(bubble_html("You", text)))
        try:
            turn = self.session.ask_text(text, speak=self.speak.value)
        except Exception as exc:  # shown in the transcript, not swallowed
            with self.log:
                display(HTML(bubble_html("Error", f"{type(exc).__name__}: {exc}")))
        else:
            d = turn.to_json_dict()
            with self.log:
                display(HTML(bubble_html("LingkodAI", d.get("native_answer") or "", answer_meta(d))))
                if self.details.value:
                    display(HTML(turn_html(d)))
                player = _player(turn.audio)
                if player is not None:
                    display(player)
            self.text.value = ""
        finally:
            self.send.disabled = False
            self.status.value = ""

    def new_conversation(self) -> None:
        from IPython.display import HTML, display

        self.session.reset()
        with self.log:
            display(HTML("<hr><i>New conversation: the next question sets the language again.</i>"))

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)
