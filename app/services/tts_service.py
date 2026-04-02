"""Gemini native TTS (preview) — returns PCM as WAV for browser playback."""

from __future__ import annotations

import io
import re
import wave

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_TTS_MODEL, GEMINI_TTS_VOICE

_RATE_RE = re.compile(r"rate=(\d+)", re.IGNORECASE)


def _pcm_mime_to_rate(mime: str) -> int:
    m = _RATE_RE.search(mime or "")
    if m:
        return int(m.group(1))
    return 24_000


def _pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def synthesize_gemini_tts_wav(text: str, voice_name: str | None = None) -> bytes:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Cannot use Gemini speech.")

    voice = (voice_name or GEMINI_TTS_VOICE or "charon").strip() or "charon"
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "Read the following passage aloud in a clear, natural voice. "
        "Do not add any introduction, summary, or commentary—only speak the passage.\n\n"
        f"{text}"
    )
    config = types.GenerateContentConfig(
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=prompt,
        config=config,
    )
    candidates = response.candidates or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if not inline:
                continue
            mime = getattr(inline, "mime_type", "") or ""
            data = getattr(inline, "data", None) or b""
            if not data:
                continue
            if "L16" in mime or "pcm" in mime.lower():
                rate = _pcm_mime_to_rate(mime)
                return _pcm16_to_wav(data, rate)
            # Fallback: return raw if already a container (unlikely)
            return bytes(data)
    raise RuntimeError("Gemini TTS returned no audio data.")
