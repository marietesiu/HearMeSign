"""tts.py — gTTS wrapper, returns MP3 bytes."""
from gtts import gTTS
import io

def text_to_speech(text: str, lang: str = "es") -> bytes:
    """Convert text to MP3 bytes using gTTS."""
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()
