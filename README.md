# Weebo

A real-time speech-to-speech chatbot powered by Whisper MLX, Ollama, and Kokoro TTS. Works natively on Apple Silicon.

Learn more [here](https://amanvir.com/weebo).

## Features

- Continuous speech recognition using Whisper MLX
- Natural language responses via Ollama (compatible with various LLM models)
- Real-time text-to-speech synthesis with Kokoro TTS
- Configurable voice settings
- Streaming response generation with sentence-by-sentence processing
- Multi-threaded audio processing pipeline

## Requirements

- Python 3.9+
- Apple Silicon Mac
- [Ollama](https://ollama.ai) installed

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models:
   - [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx) (TTS model)
   - Install your preferred LLM using Ollama (e.g., `ollama pull llama2`)

3. Configure settings:
   - Adjust `settings.json` with your preferred:
     - Ollama model name
     - System prompt
     - Voice settings
     - Processing parameters

## Usage

Run the chatbot:

```bash
python main.py
```

The program will start listening for voice input. Speak naturally and wait for a brief pause - the bot will respond with synthesized speech. Press Ctrl+C to stop.

## Project Structure

- `main.py`: Core application logic and component orchestration
- `modules/`:
  - `audio.py`: Audio input/output management
  - `chat.py`: LLM integration via Ollama
  - `stt.py`: Speech-to-text using Whisper MLX
  - `tts.py`: Text-to-speech using Kokoro
- `settings.json`: Application configuration
- `voices.json`: Voice configuration for TTS
