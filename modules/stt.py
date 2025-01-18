import numpy as np
from lightning_whisper_mlx import LightningWhisperMLX

class STTManager:
    def __init__(self):
        """Initialize speech-to-text with Whisper model"""
        self.whisper_mlx = LightningWhisperMLX(model="distil-large-v3", batch_size=12)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text using Whisper model"""
        result = self.whisper_mlx.transcribe(audio)
        return result['text'].strip()
