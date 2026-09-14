import numpy as np

from turnprobe.playout import NS, PlayoutEmulator

SR = 24000


def tone(ms: float) -> np.ndarray:
    n = int(SR * ms / 1000)
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)


def test_chunks_queue_back_to_back_even_when_they_arrive_early():
    p = PlayoutEmulator(SR)
    s1 = p.push("a", tone(100), arrival_ns=1 * NS)
    s2 = p.push("a", tone(100), arrival_ns=1 * NS + 10_000_000)  # arrives 10 ms later
    assert s1 == 1 * NS
    assert s2 == 1 * NS + 100_000_000
    assert p.audible_onsets_ns == [1 * NS]


def test_flush_reports_only_audio_actually_played():
    p = PlayoutEmulator(SR)
    for i in range(5):
        p.push("item", tone(100), arrival_ns=NS + i * 1_000_000)  # 500 ms queued almost at once
    cut = p.flush(NS + 250_000_000)
    assert set(cut) == {"item"}
    assert abs(cut["item"] - 250.0) < 1.0
    rendered = p.render(NS, int(SR * 0.6))
    assert np.abs(rendered[int(SR * 0.26):]).max() == 0.0


def test_new_burst_after_gap_registers_new_onset():
    p = PlayoutEmulator(SR)
    p.push("a", tone(100), arrival_ns=NS)
    p.push("b", tone(100), arrival_ns=2 * NS)
    assert p.audible_onsets_ns == [NS, 2 * NS]


def test_flush_forgets_onsets_that_were_never_played():
    p = PlayoutEmulator(SR)
    p.push("a", tone(200), arrival_ns=NS)
    p.push("a", np.zeros(int(SR * 0.5), np.float32), arrival_ns=NS)  # queued silence
    p.push("a", tone(200), arrival_ns=NS)  # queued second sentence, onset at +700 ms
    assert len(p.audible_onsets_ns) == 2
    p.flush(NS + 300_000_000)
    assert p.audible_onsets_ns == [NS]
