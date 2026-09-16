"""Stage 3 -- Text language identification.

GlotLID v3, a fastText language identifier covering over 2,000 languages,
applied to the ASR transcript. Prediction is restricted to four labels:
Cebuano, Filipino, Tagalog, and English. Filipino and Tagalog both map to "fil".

Environment variables (all optional, defaults shown):
    GLOTLID_REPO      cis-lmu/glotlid
    GLOTLID_REVISION  85cd6716494360367b75f642b5bc78667605d0b4
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import fasttext
from huggingface_hub import hf_hub_download

GLOTLID_REPO = os.environ.get("GLOTLID_REPO", "cis-lmu/glotlid")
GLOTLID_REVISION = os.environ.get(
    "GLOTLID_REVISION", "85cd6716494360367b75f642b5bc78667605d0b4"
)

# Candidate labels in tie-breaking order; the first label with the highest
# probability is selected.
CANDIDATE_LABELS = (
    "__label__ceb_Latn",
    "__label__fil_Latn",
    "__label__tgl_Latn",
    "__label__eng_Latn",
)

LABEL_MAP = {
    "ceb_Latn": "ceb",
    "fil_Latn": "fil",
    "tgl_Latn": "fil",
    "eng_Latn": "eng",
}


@dataclass
class TextLIDResult:
    """Predicted language for a transcript."""

    lang: str | None
    raw_label: str | None
    confidence: float | None


@dataclass
class TextLID:
    """A loaded GlotLID model."""

    model: fasttext.FastText._FastText

    def predict(self, text: str) -> TextLIDResult:
        """Identify the language of a transcript.

        Line breaks are replaced with spaces. Empty text returns a result with
        all fields set to None.
        """
        clean_text = " ".join(str(text).splitlines()).strip()
        if not clean_text:
            return TextLIDResult(lang=None, raw_label=None, confidence=None)

        labels, probs = self.model.predict(clean_text, k=-1)
        label_probs = dict(zip(labels, probs))

        scores = {label: label_probs.get(label, 0.0) for label in CANDIDATE_LABELS}
        best = max(scores, key=scores.get)
        raw_label = best.replace("__label__", "")

        return TextLIDResult(
            lang=LABEL_MAP[raw_label],
            raw_label=raw_label,
            confidence=float(scores[best]),
        )


def load() -> TextLID:
    """Download (if needed) and load the pinned GlotLID v3 model."""
    path = hf_hub_download(
        repo_id=GLOTLID_REPO, filename="model_v3.bin", revision=GLOTLID_REVISION
    )
    return TextLID(fasttext.load_model(path))