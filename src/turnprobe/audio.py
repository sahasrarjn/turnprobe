"""Small audio helpers. All internal audio is mono float32 in [-1, 1]."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    return resample(x, file_sr, sr)


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    g = gcd(src_sr, dst_sr)
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def from_pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(r, 1e-6))


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def frame_db(x: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    """Per-frame RMS level in dBFS."""
    n = max(1, int(sr * frame_ms / 1000))
    usable = len(x) // n * n
    if usable == 0:
        return np.array([rms_db(x)])
    frames = x[:usable].reshape(-1, n).astype(np.float64)
    r = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(r, 1e-6))


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -50.0) -> np.ndarray:
    """Cut leading/trailing audio below threshold so clip edges are speech edges."""
    n = int(sr * 0.005)
    levels = frame_db(x, sr, 5.0)
    loud = np.flatnonzero(levels > threshold_db)
    if len(loud) == 0:
        return x[:0]
    return x[loud[0] * n : (loud[-1] + 1) * n]


def normalize_rms(x: np.ndarray, target_db: float = -24.0) -> np.ndarray:
    """Scale speech to a typical close-mic level, measured over voiced frames only."""
    levels = frame_db(x, 24000 if len(x) > 0 else 1, 10.0)
    voiced = levels[levels > levels.max() - 30] if len(levels) else levels
    if len(voiced) == 0:
        return x
    current = 10 * np.log10(np.mean(10 ** (voiced / 10)))
    y = x * db_to_gain(target_db - current)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / peak * 0.89 if peak > 0.89 else y).astype(np.float32)


def internal_silences(
    x: np.ndarray, sr: int, threshold_db: float = -48.0, min_ms: float = 80.0, margin: float = 0.08
) -> list[tuple[int, int]]:
    """Sample ranges of silent runs inside a clip (ignoring the outer `margin` fraction)."""
    frame = int(sr * 0.005)
    levels = frame_db(x, sr, 5.0)
    silent = np.append(levels < threshold_db, False)
    first, last = int(len(levels) * margin), int(len(levels) * (1 - margin))
    runs, start = [], None
    for i in range(first, last + 1):
        if silent[i] and i < last and start is None:
            start = i
        elif (not silent[i] or i == last) and start is not None:
            if (i - start) * 5 >= min_ms:
                runs.append((start * frame, i * frame))
            start = None
    return runs


def silence_near(x: np.ndarray, sr: int, expected_ratio: float) -> tuple[int, int] | None:
    """The internal silence whose center is closest to expected_ratio of the clip."""
    runs = internal_silences(x, sr)
    if not runs:
        return None
    target = expected_ratio * len(x)
    return min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - target))


def speechlike(duration_ms: float, sr: int, seed: int = 0, level: float = 0.3) -> np.ndarray:
    """Deterministic voiced 'babble' with syllable rhythm, for the calibration bot."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_ms / 1000)
    t = np.arange(n) / sr
    f0 = 140 + 12 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 1.5, n).cumsum() / np.sqrt(n)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    carrier = sum((0.6 ** k) * np.sin((k + 1) * phase) for k in range(6))
    syl = 0.18
    env = 0.35 + 0.65 * np.abs(np.sin(np.pi * t / syl))
    fade = np.minimum(1.0, np.minimum(t / 0.01, (duration_ms / 1000 - t) / 0.01))
    y = carrier * env * np.clip(fade, 0, 1)
    return (level * y / np.max(np.abs(y))).astype(np.float32)
