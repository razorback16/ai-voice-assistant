import numpy as np
import sounddevice as sd
from scipy import signal
from typing import Callable, Optional
import torch
import time
from javad.stream import Pipeline

class AudioManager:
    def __init__(self, config: dict):
        # Audio settings
        self.input_sample_rate: int = 16000
        self.output_sample_rate: int = 24000
        self.chunk_duration: float = 0.5  # seconds
        self.chunk_samples: int = int(self.input_sample_rate * self.chunk_duration)
        self.silence_duration: float = config['audio']['silence_duration']
        
        # Voice detection setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Pipeline(device=self.device, mode='instant')
        self.last_voice_time: float = 0
        self.voice_active: bool = False
        self.stream: Optional[sd.InputStream] = None

    def start_recording(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start recording audio with voice detection"""
        audio_buffer = []
        
        def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            nonlocal audio_buffer
            
            if status:
                print(f"Stream callback error: {status}")
                return

            audio = indata[:, 0]  # Take first channel if stereo
            audio_float32 = (audio * 32767).astype(np.float32)
            
            # Add audio to buffer
            audio_buffer.extend(audio.tolist())
            
            # Check for voice activity
            if self.pipeline.detect(audio_float32):
                self.last_voice_time = time.time()
                self.voice_active = True
            elif self.voice_active and (time.time() - self.last_voice_time) > self.silence_duration:
                self.voice_active = False
                # When voice becomes inactive, process the buffer
                if len(audio_buffer) > self.input_sample_rate:  # Ensure we have at least 1 second of audio
                    audio_segment = np.array(audio_buffer, dtype=np.float32)
                    callback(audio_segment)
                audio_buffer.clear()

        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.input_sample_rate,
                blocksize=self.chunk_samples,
                dtype=np.float32
            ) as stream:
                print("Listening... Press Ctrl+C to stop")
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopped listening.")
        finally:
            self.cleanup()

    def play_audio(self, audio_data: np.ndarray) -> None:
        """Play audio data through speakers"""
        if len(audio_data) == 0:
            return
            
        with sd.OutputStream(
            samplerate=self.output_sample_rate,
            channels=1,
            dtype=np.float32
        ) as out_stream:
            out_stream.write(audio_data.reshape(-1, 1))

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.stream is not None and self.stream.active:
            self.stream.stop()
        self.pipeline.reset()
