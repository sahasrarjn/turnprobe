"""Adapter contract.

Normalized event types (data keys in parentheses):
  audio               (item_id, response_id, pcm: bytes)       model audio chunk, pcm16 mono
  user_speech_started (audio_ms, item_id?)                     system's own VAD says user began
  user_speech_stopped (audio_ms, item_id?)                     system's own VAD says user ended
  response_created    (response_id)
  response_done       (response_id, status)
  transcript          (response_id?, delta)
  error               (message, raw)
  native              (name, raw)                              anything else, kept for the log

`audio_ms` is on the system's input-audio clock (ms of audio appended since session start),
which lines up with the trial timeline because the harness streams continuously from t0.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class Event:
    t_ns: int
    type: str
    data: dict = field(default_factory=dict)


class Adapter:
    name = "adapter"
    sample_rate = 24000
    #: whether the client should flush playout when the system reports user speech
    flush_on_user_speech = True

    def __init__(self) -> None:
        self._q: asyncio.Queue[Event | None] = asyncio.Queue()

    def emit(self, type: str, **data) -> None:
        self._q.put_nowait(Event(time.monotonic_ns(), type, data))

    async def events(self) -> AsyncIterator[Event]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def connect(self) -> None:
        raise NotImplementedError

    async def send_audio(self, pcm: bytes) -> None:
        raise NotImplementedError

    async def truncate(self, item_id: str, audio_end_ms: int) -> None:
        """Tell the system how much of an interrupted item the user actually heard."""

    async def close(self) -> None:
        self._q.put_nowait(None)

    def describe(self) -> dict:
        return {"system": self.name}
