"""Speech helpers: spoken-form text for TTS and a client for the local NeMo-Speech server."""
from __future__ import annotations

import re
import time

import httpx

from atlas.config import SPEECH_URL

_MONEY = re.compile(r"(-)?\$(\d[\d,]*(?:\.\d+)?)\s*([BMK])\b")
_UNITS = {"B": "billion", "M": "million", "K": "thousand"}


def spoken(text: str) -> str:
    """Turn display text into something a TTS voice reads naturally. Numbers are unchanged in value."""
    t = _MONEY.sub(lambda m: f"{'minus ' if m.group(1) else ''}{m.group(2)} {_UNITS[m.group(3)]} dollars", text)
    t = re.sub(r"\bQ([1-4])\b", lambda m: "Q " + ["one", "two", "three", "four"][int(m.group(1)) - 1], t)
    t = re.sub(r"\bFY ?(\d{2})\b", r"F Y \1", t)
    t = re.sub(r"\bP(10|50|90)\b", r"P \1", t)
    t = re.sub(r"(\d(?:\.\d+)?)x\b", r"\1 times", t)
    t = t.replace("HHI", "H H I").replace("AUC", "A U C").replace("YoY", "year over year")
    t = t.replace("→", " to ").replace("·", ",").replace("—", ", ").replace("–", " to ")
    return t


_ES_UNITS = {"B": "mil millones", "M": "millones", "K": "mil"}
_ES_Q = {"1": "el primer trimestre", "2": "el segundo trimestre", "3": "el tercer trimestre", "4": "el cuarto trimestre"}


def spoken_es(text: str) -> str:
    """Spanish spoken form: decimal comma, 'millones de dólares', 'por ciento'. Values unchanged."""
    t = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)                       # 2,123 -> 2123
    t = _MONEY.sub(lambda m: f"{'menos ' if m.group(1) else ''}{m.group(2).replace('.', ',')} {_ES_UNITS[m.group(3)]} de dólares", t)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s?([BMK])\b", lambda m: f"{m.group(1).replace('.', ',')} {_ES_UNITS[m.group(2)]} de dólares", t)
    t = re.sub(r"(\d+)\.(\d+)\s*%", r"\1,\2 por ciento", t)
    t = re.sub(r"(\d+)\s*%", r"\1 por ciento", t)
    t = re.sub(r"(\d+)\.(\d+)x\b", r"\1,\2 veces", t)
    t = re.sub(r"(\d+)\.(\d+)", r"\1,\2", t)
    t = re.sub(r"\bQ([1-4])\b", lambda m: _ES_Q[m.group(1)], t)
    t = re.sub(r"\bFY ?(\d{2})\b", r"año fiscal \1", t)
    t = re.sub(r"\bP(10|50|90)\b", r"P \1", t)
    t = re.sub(r"\bde el\b", "del", t)
    return t.replace("→", " a ").replace("·", ",")


class Speech:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=SPEECH_URL, timeout=httpx.Timeout(20.0, connect=0.5))
        self.ok = False
        self.last_tts_ms = None
        self.last_asr_ms = None

    async def health(self) -> bool:
        try:
            r = await self.client.get("/health", timeout=1.0)
            self.ok = r.status_code == 200
        except Exception:
            self.ok = False
        return self.ok

    async def tts(self, text: str, voice: str = "Sofia", language: str = "en-US") -> tuple[bytes, float]:
        t = time.perf_counter()
        say = spoken_es(text) if language.startswith("es") else spoken(text)
        r = await self.client.post("/v1/audio/speech", json={"input": say, "voice": voice, "language": language})
        r.raise_for_status()
        ms = (time.perf_counter() - t) * 1000
        self.last_tts_ms = ms
        return r.content, ms

    async def transcribe(self, wav: bytes) -> tuple[str, float]:
        t = time.perf_counter()
        r = await self.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")})
        r.raise_for_status()
        ms = (time.perf_counter() - t) * 1000
        self.last_asr_ms = ms
        return r.json().get("text", ""), ms
