import json
from concurrent.futures import ThreadPoolExecutor
from modules.audio import AudioManager
from modules.tts import TTSManager
from modules.stt import STTManager
from modules.chat import ChatManager

class Weebo:
    def __init__(self):
        # Load configuration
        with open('settings.json', 'r') as f:
            self.config = json.load(f)

        # Initialize components
        self.audio = AudioManager(self.config)
        self.tts = TTSManager(self.config)
        self.stt = STTManager()
        self.chat = ChatManager(self.config)
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['processing']['max_threads']
        )

    def handle_audio_input(self, audio_data):
        """Process recorded audio input"""
        text = self.stt.transcribe(audio_data)
        
        # Skip empty transcriptions
        if not text.strip():
            return
            
        print(f"Transcription: {text}")
        self.chat.add_message('user', text)
        self.create_and_play_response()

    def create_and_play_response(self):
        """Generate and play response"""
        futures = []
        buffer = ""
        chunk_size = self.config['tts']['chunk_size']

        # Process response stream
        for text in self.chat.get_response():
            print(text, end='', flush=True)
            buffer += text

            # Find end of sentence to chunk at
            last_punctuation = max(
                buffer.rfind('. '),
                buffer.rfind('? '),
                buffer.rfind('! ')
            )

            if last_punctuation == -1:
                continue

            # Handle long chunks
            while last_punctuation != -1 and last_punctuation >= chunk_size:
                last_punctuation = max(
                    buffer.rfind(', ', 0, last_punctuation),
                    buffer.rfind('; ', 0, last_punctuation),
                    buffer.rfind('— ', 0, last_punctuation)
                )

            if last_punctuation == -1:
                last_punctuation = buffer.find(' ', 0, chunk_size)

            # Process chunk
            chunk_text = buffer[:last_punctuation + 1]
            ph = self.tts.phonemize(chunk_text)
            futures.append(
                self.executor.submit(
                    self.tts.generate_audio,
                    ph
                )
            )
            buffer = buffer[last_punctuation + 1:]

        # Process final chunk if any
        if buffer:
            ph = self.tts.phonemize(buffer)
            print()
            futures.append(
                self.executor.submit(
                    self.tts.generate_audio,
                    ph
                )
            )

        # Play generated audio
        for fut in futures:
            audio_data = fut.result()
            self.audio.play_audio(audio_data)

    def run(self):
        """Start the main application loop"""
        try:
            self.audio.start_recording(self.handle_audio_input)
        except KeyboardInterrupt:
            print("\nStopping...")

def main():
    weebo = Weebo()
    weebo.run()

if __name__ == "__main__":
    main()
