"""Experiment 2: talking over the model — "mm-hm" should not stop it, "wait, stop" should.

Each trial asks a question that draws a long spoken answer, waits for the model's audio to be
audible at the playout head, then plays an overlap clip `offset_ms` later (anchored to the
model's own speech, so every system is interrupted at the same point in its answer). We record
whether the model stopped, how fast, how long it kept talking over the user, and what it said next.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..adapters import make_adapter
from ..audio import write_wav
from ..labels import speech_segments
from ..session import Trial, TrialResult
from ..tts import make_tts, synth_clip
from .runner import new_run_dir, run_grid, write_run_meta

SR = 24000

QUESTION = "My wifi keeps dropping every few minutes. Can you walk me through fixing it, step by step?"
INSTRUCTIONS = (
    "You are a patient tech-support voice assistant. When asked for steps, give one continuous spoken "
    "walkthrough of about six steps, roughly thirty seconds long. Do not stop to ask for confirmation."
)

CLIPS: dict[str, dict] = {
    "mmhm": {"kind": "backchannel", "text": "Mm-hmm."},
    "yeah": {"kind": "backchannel", "text": "Yeah."},
    "okay": {"kind": "backchannel", "text": "Okay."},
    "right": {"kind": "backchannel", "text": "Right, right."},
    "stop": {"kind": "interrupt", "text": "Wait, stop."},
    "repeat": {"kind": "interrupt", "text": "Sorry, what was the first step again?"},
    "yeah_but": {"kind": "interrupt", "text": "Yeah, but what if I don't know the router password?"},
}


@dataclass
class OverlapConfig:
    system: str
    settings: list[str]
    clips: list[str] = field(default_factory=lambda: list(CLIPS))
    offsets: list[int] = field(default_factory=lambda: [2500])
    reps: int = 3
    tts: str = "say"
    voice: str | None = None
    model: str | None = None
    concurrency: int = 1
    out: str = "runs"
    budget_usd: float | None = None
    seed: int = 11


def analyze(result: TrialResult, g: dict) -> dict:
    m, ev = result.marks, result.events
    segs = speech_segments(result.model, result.sr)
    transcripts: dict[str, str] = {}
    for e in ev:
        if e["type"] == "transcript" and e.get("response_id"):
            transcripts[e["response_id"]] = transcripts.get(e["response_id"], "") + e.get("delta", "")
    base = {"model_segments": segs, "kind": CLIPS[g["clip"]]["kind"]}
    if "X.start" not in m:
        return {**base, "outcome": "no_answer"}
    x0, x1 = m["X.start"], m["X.end"]
    # "Model was talking" includes landing in a short gap between its words (< 400 ms).
    covering = next((s for s in segs if s[0] <= x0 + 50 and s[1] > x0 - 400), None)
    if covering is None:
        return {**base, "outcome": "model_silent_at_overlap"}

    detected = next((e for e in ev if e["type"] == "user_speech_started" and e["t_ms"] >= x0 - 100), None)
    trunc = next((e for e in ev if e["type"] == "client_truncate" and e["t_ms"] >= x0 - 100), None)
    if trunc is not None:
        stop_ms = trunc["t_ms"]  # client flushed playback on the system's speech event
    else:
        # No client flush (e.g. full-duplex models that stop on their own): the model stopped if
        # a speech segment ends between the overlap onset and 2 s after it, followed by >= 900 ms
        # of silence. A trailing word before going quiet counts toward stop latency.
        stop_ms = None
        after = [s for s in segs if s[1] > x0]
        for i, s in enumerate(after):
            if s[1] > x1 + 2000:
                break
            nxt = after[i + 1][0] if i + 1 < len(after) else None
            if nxt is None or nxt - s[1] >= 900:
                stop_ms = s[1]
                break

    responses_after = [e for e in ev if e["type"] == "response_created" and e["t_ms"] >= x0]
    reply_seg = next((s for s in segs if stop_ms is not None and s[0] > stop_ms + 50), None)
    if stop_ms is None:
        outcome = "kept_talking"
    elif reply_seg is not None:
        outcome = "stopped_then_replied"
    else:
        outcome = "stopped_silent"

    kind = CLIPS[g["clip"]]["kind"]
    correct = outcome == "kept_talking" if kind == "backchannel" else outcome != "kept_talking"
    done_ids = [e.get("response_id") for e in ev if e["type"] == "response_done"]
    return {
        **base,
        "outcome": outcome,
        "correct": correct,
        "detect_ms": None if detected is None else round(detected["t_ms"] - x0, 1),
        "stop_latency_ms": None if stop_ms is None else round(stop_ms - x0, 1),
        "talk_over_ms": round(max(0.0, min(covering[1], x1) - x0), 1),
        "reply_gap_ms": None if reply_seg is None else round(reply_seg[0] - x1, 1),
        "responses_after_overlap": len(responses_after),
        "answer_text": transcripts.get(done_ids[0], "") if done_ids else "",
        "reply_text": " | ".join(transcripts.get(e.get("response_id"), "") for e in responses_after),
    }


async def run_trial(g: dict, cfg: OverlapConfig, question, clips) -> TrialResult:
    adapter = make_adapter(cfg.system, g["setting"], cfg.model, instructions=INSTRUCTIONS)
    await adapter.connect()
    try:
        trial = Trial(adapter, sr=SR)

        async def script(ctx):
            ctx.silence(500)
            _, q_end = ctx.play(question, "Q")
            onset = await ctx.wait_model_onset(after_ms=q_end, timeout_ms=15000)
            if onset is None:
                return
            ctx.mark("model.onset", onset)
            ctx.play(clips[g["clip"]], "X", at_ms=onset + g["offset_ms"])
            await ctx.wait_settled(after_ms=ctx.marks["X.end"], quiet_ms=2000, give_up_ms=8000)

        return await trial.run(script)
    finally:
        await adapter.close()


async def run_overlap(cfg: OverlapConfig) -> Path:
    run_dir = new_run_dir(cfg.out, "overlap", cfg.system)
    (run_dir / "stimuli").mkdir(exist_ok=True)
    tts = make_tts(cfg.tts, cfg.voice)
    question = await asyncio.to_thread(synth_clip, tts, QUESTION, SR)
    write_wav(run_dir / "stimuli" / "question.wav", question, SR)
    clips = {}
    for cid in cfg.clips:
        clips[cid] = await asyncio.to_thread(synth_clip, tts, CLIPS[cid]["text"], SR)
        write_wav(run_dir / "stimuli" / f"{cid}.wav", clips[cid], SR)
    grid = [
        {"setting": s, "clip": c, "offset_ms": o, "rep": r}
        for s in cfg.settings for c in cfg.clips for o in cfg.offsets for r in range(cfg.reps)
    ]
    write_run_meta(run_dir, "overlap", config=asdict(cfg), tts={"provider": tts.name, "voice": tts.voice},
                   question=QUESTION, instructions=INSTRUCTIONS, clips={c: CLIPS[c] for c in cfg.clips},
                   trials_planned=len(grid))
    await run_grid(
        run_dir, grid,
        run_one=lambda g: run_trial(g, cfg, question, clips),
        analyze=analyze,
        concurrency=cfg.concurrency, budget_usd=cfg.budget_usd, seed=cfg.seed,
        describe=lambda g: f"{g['setting']:<36} {g['clip']:<9} +{g['offset_ms']}ms",
    )
    return run_dir
