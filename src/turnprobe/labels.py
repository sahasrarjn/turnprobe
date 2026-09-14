"""Offline labeling of the as-heard recording."""

from __future__ import annotations

import numpy as np

from .audio import frame_db


def speech_segments(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -42.0,
    frame_ms: float = 10.0,
    min_speech_ms: float = 60.0,
    merge_gap_ms: float = 300.0,
) -> list[tuple[float, float]]:
    """(onset_ms, offset_ms) speech regions. Model audio is clean digital TTS, so an
    energy detector with gap merging is reliable here; user-side times come from fixtures."""
    if len(x) == 0:
        return []
    loud = frame_db(x, sr, frame_ms) > threshold_db
    segs: list[list[float]] = []
    start = None
    for i, on in enumerate(np.append(loud, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segs.append([start * frame_ms, i * frame_ms])
            start = None
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap_ms:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_speech_ms]
