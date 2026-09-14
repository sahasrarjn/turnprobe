"""Overlap experiment against the mock bot: any user sound stops it (like a plain VAD system)."""

import asyncio

from turnprobe.audio import speechlike
from turnprobe.experiments.overlap import OverlapConfig, analyze, run_trial

SR = 24000


def test_backchannel_stops_a_vad_only_system_and_it_replies():
    cfg = OverlapConfig(system="mock", settings=[])
    g = {"setting": "mock:silence_duration_ms=500,response_delay_ms=200,response_ms=6000", "clip": "mmhm", "offset_ms": 1500, "rep": 0}
    clips = {"mmhm": speechlike(400, SR, seed=3)}
    a = analyze(asyncio.run(run_trial(g, cfg, speechlike(1500, SR, seed=4), clips)), g)
    assert a["outcome"] == "stopped_then_replied"
    assert a["correct"] is False
    # mock needs 40 ms of speech to detect, then the client flushes immediately
    assert 30 <= a["stop_latency_ms"] <= 90
    assert 30 <= a["talk_over_ms"] <= 90
