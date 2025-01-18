import json
import re
from typing import Dict
import numpy as np
import onnxruntime
import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import espeakng_loader

class TTSManager:
    def __init__(self, config: dict):
        self.max_phoneme_length = config['tts']['max_phoneme_length']
        self.speed = config['tts']['speed']
        self.voice = config['tts']['voice']
        
        # Initialize espeak
        self._init_espeak()
        
        # Load TTS model and voices
        self.tts_session = onnxruntime.InferenceSession(
            config['tts']['model_path'],
            providers=["CPUExecutionProvider"]
        )
        
        # Load voice profiles
        with open("voices.json") as f:
            self.voices = json.load(f)

    def _init_espeak(self):
        """Initialize espeak for phoneme generation"""
        espeak_data_path = espeakng_loader.get_data_path()
        espeak_lib_path = espeakng_loader.get_library_path()
        EspeakWrapper.set_data_path(espeak_data_path)
        EspeakWrapper.set_library(espeak_lib_path)
        self.vocab = self._create_vocab()

    def _create_vocab(self) -> Dict[str, int]:
        """Create mapping of characters/phonemes to integer tokens"""
        chars = ['$'] + list(';:,.!?¡¿—…"«»"" ') + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + \
            list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        return {c: i for i, c in enumerate(chars)}

    def phonemize(self, text: str) -> str:
        """Convert text to phonemes"""
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text).strip()
        phonemes = phonemizer.phonemize(
            text,
            "en-us",
            preserve_punctuation=True,
            with_stress=True
        )
        return "".join(p for p in phonemes.replace("r", "ɹ") if p in self.vocab).strip()

    def generate_audio(self, phonemes: str, voice: str = None, speed: float = None) -> np.ndarray:
        """Convert phonemes to audio using TTS model"""
        voice = voice or self.voice
        speed = speed or self.speed
        
        tokens = [self.vocab[p] for p in phonemes if p in self.vocab]
        if not tokens:
            return np.array([], dtype=np.float32)

        tokens = tokens[:self.max_phoneme_length]
        style = np.array(self.voices[voice], dtype=np.float32)[len(tokens)]

        audio = self.tts_session.run(
            None,
            {
                'tokens': [[0, *tokens, 0]],
                'style': style,
                'speed': np.array([speed], dtype=np.float32)
            }
        )[0]

        return audio
