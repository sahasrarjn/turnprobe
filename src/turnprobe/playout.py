"""Emulated client speaker.

Realtime servers stream audio faster than realtime, so "when did the user hear it" is
not "when did bytes arrive". This schedules every received chunk on a playout timeline
(start = max(arrival + jitter buffer, end of queued audio)), supports flushing on
interruption, and reports exactly how much of each item was played — the number a
correct client sends back as `audio_end_ms` when truncating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NS = 1_000_000_000


@dataclass
class Chunk:
    item_id: str
    start_ns: int
    samples: np.ndarray


class PlayoutEmulator:
    def __init__(
        self,
        sr: int,
        jitter_buffer_ms: float = 0.0,
        onset_threshold_db: float = -40.0,
        new_burst_gap_ms: float = 300.0,
    ):
        self.sr = sr
        self.jb_ns = int(jitter_buffer_ms * 1e6)
        self.chunks: list[Chunk] = []
        self.end_ns = 0
        self.audible_onsets_ns: list[int] = []
        self._thr = 10 ** (onset_threshold_db / 20)
        self._gap_ns = int(new_burst_gap_ms * 1e6)
        self._need_onset = True
        self.last_audible_end_ns: int | None = None

    def _dur_ns(self, n: int) -> int:
        return int(round(n * NS / self.sr))

    def push(self, item_id: str, samples: np.ndarray, arrival_ns: int) -> int:
        start = arrival_ns + self.jb_ns if arrival_ns >= self.end_ns else self.end_ns
        self.chunks.append(Chunk(item_id, start, samples))
        self.end_ns = start + self._dur_ns(len(samples))
        # An onset is loud audio after >= new_burst_gap_ms of quiet — whether the quiet was
        # no audio at all (turn-based APIs) or silent samples in a continuous stream (full duplex).
        win = max(1, int(self.sr * 0.005))
        usable = len(samples) // win * win
        if usable:
            frames = samples[:usable].reshape(-1, win)
            for k in np.flatnonzero(np.sqrt(np.mean(frames * frames, axis=1)) > self._thr):
                t = start + self._dur_ns(int(k) * win)
                if self._need_onset or self.last_audible_end_ns is None or t - self.last_audible_end_ns >= self._gap_ns:
                    self.audible_onsets_ns.append(t)
                    self._need_onset = False
                self.last_audible_end_ns = t + self._dur_ns(win)
        return start

    def flush(self, at_ns: int) -> dict[str, float]:
        """Drop everything scheduled after at_ns. Returns {item_id: played_ms} for cut items."""
        kept: list[Chunk] = []
        cut: set[str] = set()
        for c in self.chunks:
            n = len(c.samples)
            if c.start_ns + self._dur_ns(n) <= at_ns:
                kept.append(c)
                continue
            keep = max(0, int((at_ns - c.start_ns) * self.sr / NS))
            if keep < n:
                cut.add(c.item_id)
            if keep > 0:
                kept.append(Chunk(c.item_id, c.start_ns, c.samples[:keep]))
        self.chunks = kept
        self.end_ns = max((c.start_ns + self._dur_ns(len(c.samples)) for c in kept), default=0)
        # Onsets inside flushed audio were never heard.
        self.audible_onsets_ns = [t for t in self.audible_onsets_ns if t < at_ns]
        if cut:
            self._need_onset = True
            if self.last_audible_end_ns is not None:
                self.last_audible_end_ns = min(self.last_audible_end_ns, at_ns)
        return {
            item: sum(len(c.samples) for c in kept if c.item_id == item) * 1000.0 / self.sr
            for item in cut
        }

    def is_playing(self, at_ns: int) -> bool:
        return self.end_ns > at_ns and any(c.start_ns <= at_ns for c in self.chunks)

    def render(self, start_ns: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        end_ns = start_ns + self._dur_ns(n)
        for c in self.chunks:
            c_end = c.start_ns + self._dur_ns(len(c.samples))
            if c_end <= start_ns or c.start_ns >= end_ns:
                continue
            off = int(round((c.start_ns - start_ns) * self.sr / NS))
            a, b = max(0, off), min(n, off + len(c.samples))
            if b > a:
                out[a:b] += c.samples[a - off : b - off]
        return out
