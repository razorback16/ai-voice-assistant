import json
import queue
import threading
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
        """Generate and play response with parallel processing"""
        # Create a queue for audio chunks
        audio_queue = queue.Queue()
        processing_complete = threading.Event()
        
        def generate_audio():
            """Generate audio chunks in a separate thread"""
            try:
                for sentence in self.chat.get_response():
                    print(sentence, end='', flush=True)
                    ph = self.tts.phonemize(sentence)
                    audio_data = self.tts.generate_audio(ph)
                    audio_queue.put(audio_data)
                print()
            finally:
                processing_complete.set()
        
        def play_audio():
            """Play audio chunks as they become available"""
            while not (processing_complete.is_set() and audio_queue.empty()):
                try:
                    # Wait for audio chunks with a timeout
                    audio_data = audio_queue.get(timeout=0.1)
                    self.audio.play_audio(audio_data)
                except queue.Empty:
                    continue
        
        # Start audio generation in a separate thread
        generator_thread = threading.Thread(target=generate_audio)
        generator_thread.start()
        
        # Start playback in another thread
        playback_thread = threading.Thread(target=play_audio)
        playback_thread.start()
        
        # Wait for both threads to complete
        generator_thread.join()
        playback_thread.join()

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
