"""Audio loading shared by all pipeline stages.

Audio is converted to 16 kHz mono WAV with ffmpeg and then read with librosa.

System dependency: ffmpeg.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import librosa
import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Convert an audio file to 16 kHz mono WAV with ffmpeg and load it as float32."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio_16k.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    return audio
