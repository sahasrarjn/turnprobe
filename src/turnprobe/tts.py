"""Stimulus synthesis with an on-disk cache.

Split utterances ("A <pause> B") are synthesized as ONE sentence with a pause marker,
then cut at the pause. That keeps continuation prosody on A, which a human would hear
as "not finished" — the realistic case for a mid-sentence pause.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import (
    from_pcm16,
    load_mono,
    normalize_rms,
    resample,
    silence_near,
    trim_silence,
    write_wav,
)

CACHE_DIR = Path(".cache/tts")


class TTS:
    name = "tts"
    voice = ""

    def pause_text(self, a: str, b: str) -> str:
        raise NotImplementedError

    def synth(self, text: str, sr: int) -> np.ndarray:
        raise NotImplementedError


class SayTTS(TTS):
    """macOS `say`. Good enough for harness work; use OpenAI TTS for published runs."""

    name = "say"

    def __init__(self, voice: str = "Samantha"):
        self.voice = voice

    def pause_text(self, a: str, b: str) -> str:
        return f"{a} [[slnc 900]] {b}"

    def synth(self, text: str, sr: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            subprocess.run(
                ["say", "-v", self.voice, "-o", str(out), f"--data-format=LEI16@{sr}", text],
                check=True,
                capture_output=True,
            )
            return load_mono(out, sr)


class OpenAITTS(TTS):
    name = "openai"

    def __init__(self, voice: str = "coral", model: str = "gpt-4o-mini-tts"):
        self.voice = voice
        self.model = model

    def pause_text(self, a: str, b: str) -> str:
        return f"{a} ... {b}"

    def synth(self, text: str, sr: int) -> np.ndarray:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or use --tts say")
        body = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",
            "instructions": "Speak naturally, like a person talking to a voice assistant. "
            "At '...' pause as if mid-thought, keeping the sentence unfinished.",
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            pcm = resp.read()
        return resample(from_pcm16(pcm), 24000, sr)


def make_tts(name: str, voice: str | None = None) -> TTS:
    if name == "say":
        return SayTTS(voice or "Samantha")
    if name == "openai":
        return OpenAITTS(voice or "coral")
    raise ValueError(f"unknown TTS provider: {name}")


@dataclass
class SplitClip:
    a: np.ndarray
    b: np.ndarray
    method: str


def _cache_path(tts: TTS, text: str, sr: int, suffix: str) -> Path:
    h = hashlib.sha1(f"{tts.name}|{tts.voice}|{sr}|{text}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{tts.name}-{h}{suffix}.wav"


def synth_clip(tts: TTS, text: str, sr: int) -> np.ndarray:
    path = _cache_path(tts, text, sr, "")
    if path.exists():
        return load_mono(path, sr)
    x = normalize_rms(trim_silence(tts.synth(text, sr), sr))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, x, sr)
    return x


def transcribe_openai(x: np.ndarray, sr: int, model: str = "gpt-4o-mini-transcribe") -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    b = uuid.uuid4().hex
    body = (
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={b}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["text"]


_DIGITS = "zero one two three four five six seven eight nine".split()


def spoken_tokens(text: str) -> list[str]:
    """Lowercase word tokens with digits spelled out ("415" -> four one five, "oh" -> zero)."""
    out = []
    for tok in re.findall(r"[a-z']+|\d", text.lower()):
        out.append(_DIGITS[int(tok)] if tok.isdigit() else ("zero" if tok == "oh" else tok))
    return out


def split_matches(heard_a: str, heard_b: str, a: str, b: str) -> bool:
    ha, hb, ea, eb = map(spoken_tokens, (heard_a, heard_b, a, b))
    return (
        bool(ha) and bool(hb)
        and ha[-1] == ea[-1] and hb[0] == eb[0]
        and abs(len(ha) - len(ea)) <= 1 and abs(len(hb) - len(eb)) <= 1
    )


def synth_split(tts: TTS, a: str, b: str, sr: int, verify: bool | None = None, tries: int = 4) -> SplitClip:
    """Cut A and B from one continuous utterance at the pause nearest the expected split.

    With `verify` (default: on for OpenAI TTS), both halves are transcribed and the take is
    rejected unless the cut lands on the expected word boundary.
    """
    text = tts.pause_text(a, b)
    pa, pb = _cache_path(tts, text, sr, "-a"), _cache_path(tts, text, sr, "-b")
    if pa.exists() and pb.exists():
        return SplitClip(load_mono(pa, sr), load_mono(pb, sr), "continuous")
    verify = (tts.name == "openai") if verify is None else verify
    ratio = len(spoken_tokens(a)) / max(1, len(spoken_tokens(a)) + len(spoken_tokens(b)))
    for attempt in range(tries):
        whole = trim_silence(tts.synth(text, sr), sr)
        gap = silence_near(whole, sr, ratio)
        if gap is None:
            continue
        clip_a, clip_b = trim_silence(whole[: gap[0]], sr), trim_silence(whole[gap[1] :], sr)
        if verify:
            ta, tb = transcribe_openai(clip_a, sr), transcribe_openai(clip_b, sr)
            if not split_matches(ta, tb, a, b):
                continue
        joint = normalize_rms(np.concatenate([clip_a, clip_b]))
        clip_a, clip_b = joint[: len(clip_a)], joint[len(clip_a) :]
        pa.parent.mkdir(parents=True, exist_ok=True)
        write_wav(pa, clip_a, sr)
        write_wav(pb, clip_b, sr)
        return SplitClip(clip_a, clip_b, "continuous")
    raise RuntimeError(f"could not get a clean split for {a!r} | {b!r} after {tries} takes")
