import numpy as np
import sounddevice as sd
from scipy import signal
from typing import Callable

class AudioManager:
    def __init__(self, config: dict):
        self.sample_rate = config['audio']['sample_rate']
        self.whisper_sample_rate = 16000
        self.silence_threshold = config['audio']['silence_threshold']
        self.silence_duration = config['audio']['silence_duration']

    def start_recording(self, callback: Callable[[np.ndarray], None]):
        """Start recording audio with silence detection"""
        audio_buffer = []
        silence_frames = 0
        total_frames = 0

        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_buffer, silence_frames, total_frames

            if status:
                print(status)

            audio = indata.flatten()
            level = np.abs(audio).mean()

            audio_buffer.extend(audio.tolist())
            total_frames += len(audio)

            if level < self.silence_threshold:
                silence_frames += len(audio)
            else:
                silence_frames = 0

            if silence_frames > self.silence_duration * self.sample_rate:
                audio_segment = np.array(audio_buffer, dtype=np.float32)

                if len(audio_segment) > self.sample_rate:
                    resampled_audio = signal.resample_poly(
                        audio_segment,
                        self.whisper_sample_rate,
                        self.sample_rate
                    )
                    callback(resampled_audio)

                audio_buffer.clear()
                silence_frames = 0
                total_frames = 0

        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            dtype=np.float32
        ):
            print("Recording... Press Ctrl+C to stop")
            while True:
                sd.sleep(100)

    def play_audio(self, audio_data: np.ndarray):
        """Play audio data through speakers"""
        if len(audio_data) == 0:
            return
            
        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        ) as out_stream:
            out_stream.write(audio_data.reshape(-1, 1))
