"""Stage 2 -- Speech-to-text.

Three Whisper-medium checkpoints, decoder fine-tuned per language on the
Philippine Languages Database.

CEB is loaded differently from FIL and ENG. Cebuano is not one of Whisper's
languages, so training added a custom `<|cebuano|>` token at id 51865 and
resized the decoder's embedding matrix to match. The checkpoint carries its
own tokenizer. Loading fails if the tokenizer and embedding sizes disagree
or if the custom token is not at the expected id.

FIL and ENG use Whisper's native language tokens.

Environment variables (all optional, defaults shown):
    ASR_CEB   beabucayan/lingkodai-whisper-ceb
    ASR_FIL   beabucayan/lingkodai-whisper-fil
    ASR_ENG   beabucayan/lingkodai-whisper-eng
    HF_TOKEN  required while the model repos are private

System dependency: ffmpeg.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from .audio import SAMPLE_RATE, load_audio

CEB_TOKEN = "<|cebuano|>"
CEB_TOKEN_ID = 51865
MAX_LENGTH = 225

CHECKPOINTS = {
    "ceb": os.environ.get("ASR_CEB", "beabucayan/lingkodai-whisper-ceb"),
    "fil": os.environ.get("ASR_FIL", "beabucayan/lingkodai-whisper-fil"),
    "eng": os.environ.get("ASR_ENG", "beabucayan/lingkodai-whisper-eng"),
}

# Whisper's own name for each language, for the two it already supports.
WHISPER_LANG = {"fil": "tagalog", "eng": "english"}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    device = device or get_device()
    return torch.float16 if device == "cuda" else torch.float32


@dataclass
class ASRBundle:
    """A processor and model pinned to one language."""

    lang: str
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    device: str

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a single 16 kHz mono clip."""
        features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.model.dtype)

        with torch.no_grad():
            generated = self.model.generate(features, max_length=MAX_LENGTH)

        return self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    def unload(self) -> None:
        """Release the model and free GPU memory, if applicable."""
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()


def _build_ceb(repo: str, device: str, dtype: torch.dtype) -> ASRBundle:
    processor = WhisperProcessor.from_pretrained(repo, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        repo, dtype=dtype
    ).to(device).eval()

    vocab_size = len(processor.tokenizer)
    embed_size = model.get_input_embeddings().weight.shape[0]
    if vocab_size != embed_size:
        raise RuntimeError(
            f"CEB vocab/embedding mismatch: tokenizer={vocab_size}, "
            f"model={embed_size}. The repo's tokenizer does not match its weights."
        )

    token_id = processor.tokenizer.convert_tokens_to_ids(CEB_TOKEN)
    if token_id != CEB_TOKEN_ID:
        raise RuntimeError(
            f"Expected {CEB_TOKEN} at id {CEB_TOKEN_ID}, got {token_id}. "
            "The tokenizer does not match the one used in training."
        )

    transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    model.generation_config.forced_decoder_ids = [
        [1, token_id],
        [2, transcribe_id],
        [3, notimestamps_id],
    ]

    return ASRBundle("ceb", processor, model, device)


def _build_native(lang: str, repo: str, device: str, dtype: torch.dtype) -> ASRBundle:
    whisper_lang = WHISPER_LANG[lang]
    processor = WhisperProcessor.from_pretrained(
        repo, language=whisper_lang, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        repo, dtype=dtype
    ).to(device).eval()

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=whisper_lang, task="transcribe"
    )

    return ASRBundle(lang, processor, model, device)


def load(lang: str, device: str | None = None) -> ASRBundle:
    """Load the ASR bundle for one language: 'ceb', 'fil' or 'eng'."""
    if lang not in CHECKPOINTS:
        raise ValueError(f"Unknown language {lang!r}; expected one of {list(CHECKPOINTS)}")

    device = device or get_device()
    dtype = get_dtype(device)
    repo = CHECKPOINTS[lang]

    if lang == "ceb":
        return _build_ceb(repo, device, dtype)
    return _build_native(lang, repo, device, dtype)

def transcribe_file(path: str, lang: str, bundle: ASRBundle | None = None) -> str:
    """Transcribe a single audio file."""
    own = bundle is None
    bundle = bundle or load(lang)
    try:
        return bundle.transcribe(load_audio(path))
    finally:
        if own:
            bundle.unload()
