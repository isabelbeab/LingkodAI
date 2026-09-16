"""Stage 1 -- Audio language identification.

A classification head over a frozen Whisper-medium encoder. The encoder's
final hidden states (1500 frames) are averaged into 50 chunks of 30 frames.
The head scores each chunk, masks chunks that contain only padding, applies
attentive statistics pooling, and classifies the pooled vector as Cebuano,
Filipino, or English.

The chunking, the valid-chunk count, and the fp32 encoder precision match the
conditions under which the head was trained and must not be changed without
retraining the head.

Environment variables (all optional, defaults shown):
    LID_ENCODER  openai/whisper-medium
    LID_HEAD     <repository>/models/deeper_50chunks.pt
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

from .asr import SAMPLE_RATE, load_audio

LABELS = ("ceb", "fil", "eng")  # index order of the head's outputs

N_CHUNKS = 50
FRAMES_PER_CHUNK = 30
FRAMES_PER_SEC = 50
MAX_FRAMES = 1500

ENCODER_SOURCE = os.environ.get("LID_ENCODER", "openai/whisper-medium")
HEAD_PATH = os.environ.get(
    "LID_HEAD",
    str(Path(__file__).resolve().parents[1] / "models" / "deeper_50chunks.pt"),
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_head(input_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 3) -> nn.ModuleDict:
    """Construct the classification head with the layer names used in the checkpoint."""
    return nn.ModuleDict({
        "attn_fc1": nn.Linear(input_dim, 128),
        "attn_fc2": nn.Linear(128, 1),
        "clf_fc1": nn.Linear(input_dim * 2, hidden_dim),
        "clf_fc2": nn.Linear(hidden_dim, hidden_dim // 2),
        "clf_fc3": nn.Linear(hidden_dim // 2, num_classes),
    })


def valid_chunks(duration_sec: float) -> int:
    """Number of chunks containing real audio for a clip of the given duration.

    A partially filled final chunk counts as valid. The result is clipped to
    the range [1, N_CHUNKS].
    """
    chunk_size = MAX_FRAMES // N_CHUNKS
    real_frames = min(int(np.round(duration_sec * FRAMES_PER_SEC)), MAX_FRAMES)
    return int(np.clip(np.ceil(real_frames / chunk_size), 1, N_CHUNKS))


def head_forward(head: nn.ModuleDict, chunks: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    """Return logits for a batch of chunked embeddings.

    Args:
        chunks: tensor of shape (batch, N_CHUNKS, 1024).
        num_valid: tensor of shape (batch,) with the valid-chunk count per clip.
    """
    batch, n_chunks, _ = chunks.shape
    scores = head["attn_fc2"](torch.tanh(head["attn_fc1"](chunks))).squeeze(-1)

    chunk_idx = torch.arange(n_chunks, device=chunks.device).unsqueeze(0)
    scores = scores.masked_fill(chunk_idx >= num_valid.unsqueeze(1), float("-inf"))

    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
    mean = (chunks * weights).sum(dim=1)
    variance = ((chunks - mean.unsqueeze(1)) ** 2 * weights).sum(dim=1)
    std = torch.sqrt(variance.clamp(min=1e-8))
    pooled = torch.cat([mean, std], dim=1)

    h = F.relu(head["clf_fc1"](pooled))
    h = F.relu(head["clf_fc2"](h))
    return head["clf_fc3"](h)


@dataclass
class LIDResult:
    """Predicted language with per-class probabilities."""

    lang: str
    confidence: float
    probs: dict[str, float]


@dataclass
class AudioLID:
    """Feature extractor, frozen encoder, and classification head."""

    feature_extractor: WhisperFeatureExtractor
    encoder: nn.Module
    head: nn.ModuleDict
    device: str

    def encode(self, audio: np.ndarray) -> torch.Tensor:
        """Return the encoder's final hidden states, shape (1, 1500, 1024)."""
        features = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            return self.encoder(features).last_hidden_state

    def predict(self, audio: np.ndarray) -> LIDResult:
        """Identify the language of a single 16 kHz mono clip."""
        hidden = self.encode(audio)
        batch, n_frames, dim = hidden.shape
        n_chunks = n_frames // FRAMES_PER_CHUNK
        chunks = hidden[:, : n_chunks * FRAMES_PER_CHUNK, :].view(
            batch, n_chunks, FRAMES_PER_CHUNK, dim
        ).mean(dim=2)

        duration = len(audio) / SAMPLE_RATE
        num_valid = torch.tensor([valid_chunks(duration)], device=self.device)

        with torch.no_grad():
            logits = head_forward(self.head, chunks.float(), num_valid)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        best = int(probs.argmax())
        return LIDResult(
            lang=LABELS[best],
            confidence=float(probs[best]),
            probs={label: float(p) for label, p in zip(LABELS, probs)},
        )

    def unload(self) -> None:
        """Release the encoder and head and free GPU memory, if applicable."""
        del self.encoder, self.head
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load(device: str | None = None) -> AudioLID:
    """Load the feature extractor, fp32 encoder, and classification head."""
    device = device or get_device()

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

    full_model = WhisperForConditionalGeneration.from_pretrained(
        ENCODER_SOURCE, dtype=torch.float32
    )
    encoder = full_model.model.encoder.to(device).eval()
    del full_model

    head = build_head()
    head.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    head = head.to(device).eval()

    return AudioLID(feature_extractor, encoder, head, device)


def predict_file(path: str, model: AudioLID | None = None) -> LIDResult:
    """Identify the language of a single audio file.

    If no model is supplied, one is loaded and released within the call.
    """
    owns_model = model is None
    model = model or load()
    try:
        return model.predict(load_audio(path))
    finally:
        if owns_model:
            model.unload()
