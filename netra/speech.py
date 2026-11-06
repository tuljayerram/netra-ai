"""Offline text-to-speech with espeak-ng (writes a WAV file)."""
import hashlib
import re
import shutil
import subprocess

from .common import AUDIO_OUT, rel


def speak_text(text: str, voice: str = "en", words_per_minute: int = 150) -> dict:
    """Convert text to a WAV file using espeak-ng.

    Args:
        text: Sentence to speak, up to 500 characters.
        voice: espeak-ng voice such as 'en', 'hi' or 'te'.
        words_per_minute: Speaking rate from 80 to 300.
    """
    if not text.strip() or len(text) > 500:
        raise ValueError("text must contain 1 to 500 characters.")
    if not re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]+)?", voice):
        raise ValueError("Invalid voice name.")
    if not 80 <= words_per_minute <= 300:
        raise ValueError("words_per_minute must be in [80, 300].")
    exe = shutil.which("espeak-ng")
    if exe is None:
        raise RuntimeError("espeak-ng is not installed: sudo apt install espeak-ng")
    digest = hashlib.sha1(f"{voice}|{words_per_minute}|{text}".encode()).hexdigest()[:10]
    out = AUDIO_OUT / f"speech_{digest}.wav"
    subprocess.run([exe, "--stdin", "-v", voice, "-s", str(words_per_minute), "-w", str(out)],
                   input=text, text=True, check=True, timeout=30)
    return {"output": rel(out), "characters": len(text)}
