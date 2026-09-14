"""Calibration bot: a fake realtime system with exactly known timing.

It runs a plain energy VAD on the incoming stream, fires `user_speech_stopped` after
`silence_duration_ms` of silence, waits `response_delay_ms`, then streams a voiced
babble response faster than realtime (like real servers do). Because its behavior is
deterministic, any difference between configured and measured timing is harness error.
"""

from __future__ import annotations

import asyncio
import itertools

from ..audio import from_pcm16, rms_db, speechlike, to_pcm16
from .base import Adapter

_ids = itertools.count(1)


class MockBot(Adapter):
    name = "mock"

    def __init__(
        self,
        silence_duration_ms: float = 500,
        response_delay_ms: float = 300,
        threshold_db: float = -45,
        min_speech_ms: float = 40,
        response_ms: float = 2400,
        chunk_ms: float = 100,
        speedup: float = 4.0,
        interrupt_response: bool = True,
    ):
        super().__init__()
        self.silence_ms = silence_duration_ms
        self.delay_ms = response_delay_ms
        self.threshold_db = threshold_db
        self.min_speech_ms = min_speech_ms
        self.response_ms = response_ms
        self.chunk_ms = chunk_ms
        self.speedup = speedup
        self.flush_on_user_speech = interrupt_response
        self._recv = 0
        self._in_speech = False
        self._speech_run = 0.0
        self._silence_run = 0.0
        self._task: asyncio.Task | None = None

    async def connect(self) -> None:
        self.emit("native", name="session.created", raw=self.describe())

    async def send_audio(self, pcm: bytes) -> None:
        x = from_pcm16(pcm)
        frame_ms = len(x) * 1000.0 / self.sample_rate
        self._recv += len(x)
        end_ms = self._recv * 1000.0 / self.sample_rate
        if rms_db(x) > self.threshold_db:
            self._silence_run = 0.0
            if not self._in_speech:
                self._speech_run += frame_ms
                if self._speech_run >= self.min_speech_ms:
                    self._in_speech = True
                    self.emit("user_speech_started", audio_ms=end_ms - self._speech_run)
                    if self._task and not self._task.done() and self.flush_on_user_speech:
                        self._task.cancel()
        else:
            self._speech_run = 0.0
            if self._in_speech:
                self._silence_run += frame_ms
                if self._silence_run >= self.silence_ms:
                    self._in_speech = False
                    self.emit("user_speech_stopped", audio_ms=end_ms - self._silence_run)
                    self._task = asyncio.create_task(self._respond())

    async def _respond(self) -> None:
        await asyncio.sleep(self.delay_ms / 1000)
        n = next(_ids)
        rid, item = f"resp_{n}", f"item_{n}"
        self.emit("response_created", response_id=rid)
        clip = speechlike(self.response_ms, self.sample_rate, seed=n)
        step = int(self.sample_rate * self.chunk_ms / 1000)
        try:
            for i in range(0, len(clip), step):
                self.emit("audio", item_id=item, response_id=rid, pcm=to_pcm16(clip[i : i + step]))
                await asyncio.sleep(self.chunk_ms / 1000 / self.speedup)
            self.emit("response_done", response_id=rid, status="completed")
        except asyncio.CancelledError:
            self.emit("response_done", response_id=rid, status="cancelled")
            raise

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        await super().close()

    def describe(self) -> dict:
        return {
            "system": self.name,
            "silence_duration_ms": self.silence_ms,
            "response_delay_ms": self.delay_ms,
            "threshold_db": self.threshold_db,
        }
