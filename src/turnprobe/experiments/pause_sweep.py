"""Experiment 1: how long can you pause mid-sentence before the system takes the turn?

Each trial: 500 ms lead-in, clip A, a silence of exactly `pause_ms`, clip B, then wait for
the system to settle. A and B are cut from one continuous TTS utterance, so A ends with
"unfinished" prosody. Every trial is a fresh session, so no conversation history leaks
between trials.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..adapters import make_adapter
from ..audio import write_wav
from ..labels import speech_segments
from ..session import Trial, TrialResult
from ..tts import SplitClip, make_tts, synth_split
from .runner import new_run_dir, run_grid, write_run_meta

SR = 24000

STIMULI: dict[str, dict] = {
    "booking": {"kind": "incomplete", "a": "I'd like to book a table for", "b": "four people at seven tonight."},
    "phone": {"kind": "dictation", "a": "Sure, my phone number is four one five,", "b": "five five five, zero one nine two."},
    "capital": {"kind": "incomplete", "a": "Can you tell me what the capital of", "b": "Australia is?"},
    "reservation": {"kind": "complete", "a": "I need to change my reservation.", "b": "It's under the name Jordan."},
    "weather": {"kind": "incomplete", "a": "What's the weather going to be like in", "b": "Seattle this weekend?"},
    "remind": {"kind": "incomplete", "a": "Can you remind me to call my mom when I get", "b": "home from work?"},
    "egg": {"kind": "incomplete", "a": "How long should I boil an egg if I want the yolk", "b": "still a little runny?"},
    "router": {"kind": "incomplete", "a": "So what I'm trying to figure out is", "b": "whether I need a new router."},
    "address": {"kind": "dictation", "a": "The delivery address is fourteen twenty eight Elm Street,", "b": "Springfield, zip code six two seven oh four."},
    "gate": {"kind": "dictation", "a": "The gate code is four seven seven one,", "b": "then the pound key."},
    "order": {"kind": "complete", "a": "My order hasn't arrived yet.", "b": "It was supposed to come on Tuesday."},
    "cancel": {"kind": "complete", "a": "I want to cancel my subscription.", "b": "The premium one, not the basic plan."},
}
ORIGINAL_STIMULI = ["booking", "phone", "capital", "reservation"]


@dataclass
class SweepConfig:
    system: str
    settings: list[str]
    pauses: list[int]
    reps: int = 3
    stimuli: list[str] = field(default_factory=lambda: list(ORIGINAL_STIMULI))
    tts: str = "say"
    voice: str | None = None
    voices: list[str] | None = None  # several TTS voices; each (stimulus, voice) is its own clip
    model: str | None = None
    concurrency: int = 1
    out: str = "runs"
    seed: int = 7
    budget_usd: float | None = None


def analyze(result: TrialResult, g: dict | None = None) -> dict:
    m = result.marks
    a_end, b_start, b_end = m["A.end"], m["B.start"], m["B.end"]
    all_segs = speech_segments(result.model, result.sr)
    # The outcome is system-neutral: ANY model speech before you finish counts. Short sounds
    # (< 900 ms, ending before you finish) are also tallied as descriptors, because full-duplex
    # models make them in two flavors: an ack inside the pause ("mhm", "got it") and a false
    # start that overlaps your resumed speech and backs off. Transcripts tell them apart.
    segs = all_segs
    short = [s for s in all_segs if s[1] - s[0] < 900 and s[1] <= b_end + 100]
    short_in_pause = [s for s in short if s[1] <= b_start + 50]
    short_over_user = [s for s in short if s[1] > b_start + 50]
    first = segs[0][0] if segs else None

    if first is None:
        outcome = "no_response"
    elif first < b_start:
        outcome = "cut_in"  # model audible during the pause
    elif first < b_end:
        outcome = "talk_over"  # response to the pause landed while the user was talking again
    else:
        outcome = "clean"

    stops = [e for e in result.events if e["type"] == "user_speech_stopped" and e.get("audio_ms") is not None]
    # audio_ms is the system's commit point on the input clock (for OpenAI: speech end + silence timer)
    split = any(a_end - 250 <= e["audio_ms"] < b_start + 150 for e in stops)
    final_stop = next((e for e in stops if e["audio_ms"] >= b_start + 150), None)

    after_b = [s for s in segs if s[0] >= b_end]
    final_onset = after_b[0][0] if after_b else None
    gap_ms = None if final_onset is None else round(final_onset - b_end, 1)

    endpoint_lag = gen_lag = vad_offset = None
    if final_stop is not None:
        endpoint_lag = round(final_stop["t_ms"] - b_end, 1)
        vad_offset = round(final_stop["audio_ms"] - b_end, 1)
        first_audio = next(
            (e for e in result.events
             if e["type"] == "audio" and not e.get("ignored") and e["t_ms"] >= final_stop["t_ms"]),
            None,
        )
        if first_audio is not None:
            gen_lag = round(first_audio["t_ms"] - final_stop["t_ms"], 1)

    responses = [e for e in result.events if e["type"] == "response_created"]
    cancelled = [e for e in result.events if e["type"] == "response_done" and e.get("status") == "cancelled"]
    errors = [e for e in result.events if e["type"] == "error"]
    user_spans = [(m["A.start"], a_end), (b_start, b_end)]
    talk_over_ms = sum(max(0.0, min(s1, u1) - max(s0, u0)) for s0, s1 in segs for u0, u1 in user_spans)
    return {
        "outcome": outcome,
        "first_onset_ms": first,
        "split_in_pause": split,
        "final_gap_ms": gap_ms,
        "endpoint_lag_ms": endpoint_lag,
        "vad_end_offset_ms": vad_offset,
        "generation_lag_ms": gen_lag,
        "responses": len(responses),
        "cancelled": len(cancelled),
        "talk_over_ms": round(talk_over_ms, 1),
        "truncations": sum(1 for e in result.events if e["type"] == "client_truncate"),
        "api_errors": [e.get("message") for e in errors],
        "short_in_pause": len(short_in_pause),
        "short_over_user": len(short_over_user),
        "model_segments": all_segs,
    }


async def run_trial(system: str, setting: str, model: str | None, clip: SplitClip, pause_ms: int) -> TrialResult:
    adapter = make_adapter(system, setting, model)
    await adapter.connect()
    try:
        trial = Trial(adapter, sr=SR)

        async def script(ctx):
            ctx.silence(500)
            ctx.play(clip.a, "A")
            ctx.silence(pause_ms)
            ctx.play(clip.b, "B")
            await ctx.wait_settled(after_ms=ctx.marks["B.end"])

        return await trial.run(script)
    finally:
        await adapter.close()


async def run_sweep(cfg: SweepConfig) -> Path:
    run_dir = new_run_dir(cfg.out, "pause-sweep", cfg.system)
    (run_dir / "stimuli").mkdir(exist_ok=True)
    voices = cfg.voices or [cfg.voice]
    clips: dict[tuple[str, str | None], SplitClip] = {}
    for voice in voices:
        tts = make_tts(cfg.tts, voice)
        for sid in cfg.stimuli:
            s = STIMULI[sid]
            c = clips[(sid, voice)] = await asyncio.to_thread(synth_split, tts, s["a"], s["b"], SR)
            tag = f"{sid}-{tts.voice}"
            write_wav(run_dir / "stimuli" / f"{tag}-A.wav", c.a, SR)
            write_wav(run_dir / "stimuli" / f"{tag}-B.wav", c.b, SR)
            print(f"stimulus {tag}: A {len(c.a) / SR:.2f}s, B {len(c.b) / SR:.2f}s", flush=True)

    grid = [
        {"setting": st, "stimulus": sid, "kind": STIMULI[sid]["kind"], "voice": v, "pause_ms": p, "rep": r}
        for st in cfg.settings for sid in cfg.stimuli for v in voices for p in cfg.pauses for r in range(cfg.reps)
    ]
    write_run_meta(run_dir, "pause_sweep", config=asdict(cfg), tts={"provider": cfg.tts, "voice": ",".join(str(v) for v in voices)},
                   stimuli={sid: STIMULI[sid] for sid in cfg.stimuli},
                   trials_planned=len(grid))
    await run_grid(
        run_dir, grid,
        run_one=lambda g: run_trial(cfg.system, g["setting"], cfg.model, clips[(g["stimulus"], g["voice"])], g["pause_ms"]),
        analyze=analyze,
        concurrency=cfg.concurrency, budget_usd=cfg.budget_usd, seed=cfg.seed,
        describe=lambda g: f"{g['setting']:<36} {g['stimulus']:<11} {str(g['voice']):<6} pause={g['pause_ms']:>5}ms",
    )
    return run_dir
