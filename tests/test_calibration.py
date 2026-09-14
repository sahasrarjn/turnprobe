"""End-to-end harness check against the deterministic mock bot (runs in real time, ~15 s)."""

import asyncio

from turnprobe.audio import speechlike
from turnprobe.experiments.pause_sweep import analyze, run_trial
from turnprobe.tts import SplitClip

SR = 24000
CLIP = SplitClip(a=speechlike(1200, SR, seed=1), b=speechlike(900, SR, seed=2), method="synthetic")
SETTING = "mock:silence_duration_ms=500,response_delay_ms=300"


def run(pause_ms: int) -> dict:
    return analyze(asyncio.run(run_trial("mock", SETTING, None, CLIP, pause_ms)), {})


def test_short_pause_gives_one_clean_response_with_exact_gap():
    a = run(300)
    assert a["outcome"] == "clean"
    assert not a["split_in_pause"]
    assert 800 - 25 <= a["final_gap_ms"] <= 800 + 25


def test_long_pause_is_an_audible_cut_in_then_interrupted_and_truncated():
    a = run(1400)
    assert a["outcome"] == "cut_in"
    assert a["split_in_pause"]
    # The bot finished *sending* before the user resumed (servers stream faster than realtime),
    # so nothing is left to cancel server-side; the client must truncate what was never played.
    assert a["truncations"] >= 1
    assert 800 - 25 <= a["final_gap_ms"] <= 800 + 25
