"""OpenAI speech API — returns WAV for HTMLAudioElement playback."""

from __future__ import annotations

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE

_OPENAI_VOICES = frozenset({"alloy", "echo", "fable", "onyx", "nova", "shimmer"})


def synthesize_openai_tts_wav(text: str, voice_name: str | None = None) -> bytes:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Cannot use OpenAI speech.")

    raw = (voice_name or OPENAI_TTS_VOICE or "alloy").strip().lower() or "alloy"
    voice = raw if raw in _OPENAI_VOICES else "alloy"
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=voice,  # type: ignore[arg-type]
        input=text,
        response_format="wav",
    )
    data = response.content
    if not data:
        raise RuntimeError("OpenAI TTS returned no audio data.")
    return data
