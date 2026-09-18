"""Hard-case scripts against the mock bot: the harness plays every part and the analysis runs."""

import asyncio

from turnprobe.audio import speechlike
from turnprobe.experiments.hard_cases import CASES, HardCasesConfig, analyze, run_trial

SR = 24000
MOCK = "mock:silence_duration_ms=500,response_delay_ms=200,response_ms=3000"


def clips():
    s = lambda ms, seed: speechlike(ms, SR, seed=seed)
    return {
        ("echo", None, None): {"question": s(1500, 1)}, ("echo_ctrl", None, None): {"question": s(1500, 1)},
        ("side_talk", None, None): {"opener": s(1200, 2), "hold": s(600, 3), "aside": s(1500, 4) * 0.3, "back": s(1200, 5)},
        ("correction", None, "day"): {"a": s(1500, 6), "b": s(1000, 7)},
        ("um_hold", None, None): {"a": s(900, 8), "um": s(400, 9), "hang_on": s(900, 10), "b": s(1500, 11)},
    }


def run(case):
    cfg = HardCasesConfig(system="mock", settings=[MOCK], cases=[case], voice=None)
    g = {"setting": MOCK, "case": case, "rep": 0}
    return asyncio.run(run_trial(g, cfg, clips())), g


def test_echo_makes_a_vad_system_interrupt_itself():
    result, g = run("echo")
    a = analyze(result, g)
    assert a["outcome"] in ("restart_loop", "stalled")
    assert a["truncations"] > 0  # its own echo triggered the speech detector and the client flushed


def test_no_echo_reference_is_clean():
    result, g = run("echo_ctrl")
    assert analyze(result, g)["outcome"] == "clean"


def test_scripted_cases_place_every_part():
    for case, parts in (("side_talk", "OHTF"), ("correction", "AB"), ("um_hold", "AUNB")):
        result, g = run(case)
        starts = [result.marks[f"{p}.start"] for p in parts]
        assert starts == sorted(starts), case
        assert "outcome" in analyze(result, g)
    assert set(CASES) == {"echo", "echo_ctrl", "side_talk", "correction", "um_hold"}
