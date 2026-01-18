"""Speech recognition via faster-whisper."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class ASR:
    """Wrapper for faster-whisper ASR."""

    def __init__(self, *, model: str) -> None:
        self._model = WhisperModel(model, device="auto", compute_type="auto")

    def transcribe(self, audio_f32: np.ndarray, *, sample_rate: int) -> str:
        segments, _ = self._model.transcribe(audio_f32, language="en", vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()
