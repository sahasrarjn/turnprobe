"""Trial runner: one clock, realtime pacing, reactive fixture scripts.

Timeline convention: t = 0 ms is the start of the first mic frame. A mic frame covering
[i*20, (i+1)*20) ms is sent at t = (i+1)*20 ms, when a real microphone would have it.
Model audio is placed on the same timeline by the playout emulator.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field

import numpy as np

from .adapters.base import Adapter
from .audio import db_to_gain, from_pcm16, to_pcm16
from .playout import PlayoutEmulator

FRAME_MS = 20
LEAD_MS = 60  # never schedule user audio closer than this to "now"


@dataclass
class EchoConfig:
    delay_ms: float = 120.0
    gain_db: float = -18.0


@dataclass
class TrialResult:
    sr: int
    user: np.ndarray  # clean user speech as scheduled
    mic: np.ndarray  # what was actually sent (user + echo)
    model: np.ndarray  # model audio at the playout head
    events: list[dict]
    marks: dict[str, float]
    send_lateness_ms: dict[str, float]
    adapter: dict = field(default_factory=dict)


class MicTimeline:
    def __init__(self, sr: int, seconds: float = 60):
        self.sr = sr
        self.buf = np.zeros(int(sr * seconds), np.float32)

    def place(self, clip: np.ndarray, start: int) -> None:
        end = start + len(clip)
        if end > len(self.buf):
            grown = np.zeros(max(end, 2 * len(self.buf)), np.float32)
            grown[: len(self.buf)] = self.buf
            self.buf = grown
        self.buf[start:end] += clip

    def read(self, start: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        end = min(start + n, len(self.buf))
        if end > start:
            out[: end - start] = self.buf[start:end]
        return out


class TrialContext:
    """What a fixture script sees. Times are ms on the trial timeline."""

    def __init__(self, trial: Trial):
        self._t = trial
        self.cursor_ms = 0.0
        self.marks: dict[str, float] = {}

    def now_ms(self) -> float:
        return (time.monotonic_ns() - self._t.t0_ns) / 1e6

    def _earliest(self) -> float:
        return max(self.cursor_ms, self.now_ms() + LEAD_MS)

    def silence(self, ms: float) -> None:
        self.cursor_ms = self._earliest() + ms

    def play(self, clip: np.ndarray, label: str, at_ms: float | None = None) -> tuple[float, float]:
        sr = self._t.sr
        start_ms = self._earliest() if at_ms is None else max(at_ms, self.now_ms() + LEAD_MS)
        s = int(round(start_ms * sr / 1000))
        self._t.mic.place(clip, s)
        start, end = s * 1000 / sr, (s + len(clip)) * 1000 / sr
        self.marks[f"{label}.start"], self.marks[f"{label}.end"] = start, end
        self.cursor_ms = end
        return start, end

    def mark(self, name: str, ms: float) -> None:
        self.marks[name] = ms

    async def wait_until(self, ms: float) -> None:
        delay = ms - self.now_ms()
        if delay > 0:
            await asyncio.sleep(delay / 1000)

    def model_onsets_ms(self) -> list[float]:
        return [(t - self._t.t0_ns) / 1e6 for t in self._t.playout.audible_onsets_ns]

    async def wait_model_onset(self, after_ms: float, timeout_ms: float) -> float | None:
        deadline = self.now_ms() + timeout_ms
        while self.now_ms() < deadline:
            hits = [t for t in self.model_onsets_ms() if t >= after_ms]
            if hits:
                return hits[0]
            await asyncio.sleep(0.01)
        return None

    async def wait_settled(
        self, after_ms: float, quiet_ms: float = 1500, give_up_ms: float = 9000, hard_cap_ms: float = 45000
    ) -> None:
        """Return when no response is in flight, playout has been quiet for `quiet_ms`, and
        either a response finished after `after_ms` or `give_up_ms` has passed since it."""
        t = self._t
        while True:
            now = self.now_ms()
            audible_end = t.playout.last_audible_end_ns
            audible_end_ms = -1e9 if audible_end is None else (audible_end - t.t0_ns) / 1e6
            idle = t.active_responses == 0 and now >= max(audible_end_ms, after_ms) + quiet_ms
            done = t.last_response_done_ms
            spoke_after = any(after_ms <= o <= now for o in self.model_onsets_ms())  # heard, not merely queued
            responded = (done is not None and done >= after_ms) or spoke_after
            if idle and (responded or now >= after_ms + give_up_ms):
                return
            if now >= after_ms + hard_cap_ms:
                return
            await asyncio.sleep(0.05)


class Trial:
    def __init__(
        self,
        adapter: Adapter,
        sr: int = 24000,
        echo: EchoConfig | None = None,
        jitter_buffer_ms: float = 0.0,
    ):
        self.adapter = adapter
        self.sr = sr
        self.echo = echo
        self.mic = MicTimeline(sr)
        self.playout = PlayoutEmulator(sr, jitter_buffer_ms=jitter_buffer_ms)
        self.t0_ns = 0
        self.events: list[dict] = []
        self.active_responses = 0
        self.last_response_done_ms: float | None = None
        self._cancelled_items: set[str] = set()
        self._sent: list[np.ndarray] = []
        self._lateness: list[float] = []
        self._send_ms: list[float] = []
        self._stop = False

    def _ms(self, t_ns: int) -> float:
        return round((t_ns - self.t0_ns) / 1e6, 2)

    async def run(self, script: Callable[[TrialContext], Awaitable[None]]) -> TrialResult:
        self.t0_ns = time.monotonic_ns() + 100_000_000
        ctx = TrialContext(self)
        sender = asyncio.create_task(self._sender())
        receiver = asyncio.create_task(self._receiver())
        try:
            await script(ctx)
        finally:
            self._stop = True
            with suppress(Exception):
                await sender
            receiver.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await receiver
        mic = np.concatenate(self._sent) if self._sent else np.zeros(0, np.float32)
        n = len(mic)
        late = np.array(self._lateness) if self._lateness else np.zeros(1)
        return TrialResult(
            sr=self.sr,
            user=self.mic.read(0, n),
            mic=mic,
            model=self.playout.render(self.t0_ns, n),
            events=self.events,
            marks=ctx.marks,
            send_lateness_ms={
                "p50": float(np.percentile(late, 50)),
                "p99": float(np.percentile(late, 99)),
                "max": float(late.max()),
                # where the sender fell behind (frame time, ms late) and how long the slowest sends blocked
                "spikes": [(i * FRAME_MS, round(x, 1)) for i, x in enumerate(self._lateness) if x > 50][:40],
                "send_call_max_ms": round(max(self._send_ms, default=0.0), 1),
            },
            adapter=self.adapter.describe(),
        )

    async def _sender(self) -> None:
        n = self.sr * FRAME_MS // 1000
        frame_ns = FRAME_MS * 1_000_000
        echo_gain = db_to_gain(self.echo.gain_db) if self.echo else 0.0
        echo_ns = int(self.echo.delay_ms * 1e6) if self.echo else 0
        i = 0
        while not self._stop:
            deadline = self.t0_ns + (i + 1) * frame_ns
            delay = deadline - time.monotonic_ns()
            if delay > 0:
                await asyncio.sleep(delay / 1e9)
            self._lateness.append((time.monotonic_ns() - deadline) / 1e6)
            frame = self.mic.read(i * n, n)
            if self.echo:
                frame = frame + echo_gain * self.playout.render(self.t0_ns + i * frame_ns - echo_ns, n)
            self._sent.append(frame)
            s0 = time.monotonic_ns()
            await self.adapter.send_audio(to_pcm16(frame))
            self._send_ms.append((time.monotonic_ns() - s0) / 1e6)
            i += 1

    async def _receiver(self) -> None:
        async for ev in self.adapter.events():
            rec = {"t_ms": self._ms(ev.t_ns), "type": ev.type}
            rec.update({k: v for k, v in ev.data.items() if k != "pcm"})
            if ev.type == "audio":
                item = ev.data.get("item_id", "")
                pcm = ev.data["pcm"]
                rec["samples"] = len(pcm) // 2
                if item in self._cancelled_items:
                    rec["ignored"] = True
                else:
                    start = self.playout.push(item, from_pcm16(pcm), ev.t_ns)
                    rec["play_ms"] = self._ms(start)
            elif ev.type == "user_speech_started" and self.adapter.flush_on_user_speech:
                cut = self.playout.flush(ev.t_ns)
                self.events.append(rec)
                for item, played_ms in cut.items():
                    self._cancelled_items.add(item)
                    await self.adapter.truncate(item, int(played_ms))
                    self.events.append(
                        {"t_ms": self._ms(time.monotonic_ns()), "type": "client_truncate",
                         "item_id": item, "audio_end_ms": round(played_ms, 1)}
                    )
                continue
            elif ev.type == "response_created":
                self.active_responses += 1
            elif ev.type == "response_done":
                self.active_responses = max(0, self.active_responses - 1)
                self.last_response_done_ms = rec["t_ms"]
            self.events.append(rec)
