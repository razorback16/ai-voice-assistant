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
        """Generate and play response in a pipeline"""
        futures = []
        
        # Process response stream sentence by sentence
        for sentence in self.chat.get_response():
            print(sentence, end='', flush=True)
            
            # Process sentence
            ph = self.tts.phonemize(sentence)
            futures.append(
                self.executor.submit(
                    self.tts.generate_audio,
                    ph
                )
            )
            
            # Play completed audio chunks while next sentence is being processed
            while len(futures) > 0:  # Ensure we have at least 1 future
                audio_data = futures[0].result()
                self.audio.play_audio(audio_data)
                futures.pop(0)
        
        print()
        
        # Play any remaining audio chunks
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
