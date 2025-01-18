from typing import List, Dict, Generator
import re
from ollama import chat

def remove_emojis(text: str) -> str:
    """Remove emojis from text using regex pattern"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

class ChatManager:
    def __init__(self, config: dict):
        """Initialize chat manager with configuration"""
        self.model = config['chat']['model']
        self.system_prompt = config['chat']['system_prompt']
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({
            'role': role,
            'content': content
        })

    def get_response(self) -> Generator[str, None, None]:
        """Get streaming response from the model"""
        stream = chat(
            model=self.model,
            messages=[{
                'role': 'system',
                'content': self.system_prompt
            }] + self.messages,
            stream=True,
        )

        current_response = ""
        buffer = ""
        for chunk in stream:
            text = chunk['message']['content']
            
            if len(text) == 0:
                # Handle any remaining text in buffer
                if buffer:
                    cleaned_text = remove_emojis(buffer)
                    if cleaned_text:
                        yield cleaned_text
                
                cleaned_response = remove_emojis(current_response)
                self.add_message('assistant', cleaned_response)
                current_response = ""
                buffer = ""
                continue
                
            current_response += text
            buffer += text
            
            # Split on sentence delimiters followed by space
            sentences = re.split(r'([.!?;](?=\s|$))', buffer)
            
            # Process complete sentences
            i = 0
            while i < len(sentences) - 2:  # Keep last segment if incomplete
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                cleaned_text = remove_emojis(sentence)
                if cleaned_text:
                    yield cleaned_text
                i += 2
            
            # Keep any remaining partial sentence in buffer
            buffer = ''.join(sentences[i:])
