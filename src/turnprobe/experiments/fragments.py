"""Control: does semantic VAD judge the whole sentence, or only what came after the last pause?

For each sentence and voice, three utterances are played to a fresh session:
  whole      the full sentence, synthesized in one take with no pause
  b_alone    only the second half (the exact clip used after the pause in the sweep)
  split      first half, a `pause_ms` silence, second half (same clips as the sweep)
If the classifier scores only the last fragment, `split` should wait like `b_alone`, not like `whole`.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..adapters import make_adapter
from ..audio import write_wav
from ..labels import speech_segments
from ..session import Trial, TrialResult
from ..tts import make_tts, synth_clip, synth_split
from .pause_sweep import STIMULI
from .runner import new_run_dir, run_grid, write_run_meta

SR = 24000
CONDITIONS = ["whole", "b_alone", "split"]


@dataclass
class FragmentsConfig:
    system: str
    settings: list[str]
    stimuli: list[str] = field(default_factory=lambda: ["capital", "weather", "booking", "reservation"])
    voices: list[str] = field(default_factory=lambda: ["coral", "ash"])
    pause_ms: int = 1500
    reps: int = 1
    tts: str = "openai"
    model: str | None = None
    concurrency: int = 1
    out: str = "runs"
    budget_usd: float | None = None
    seed: int = 13


def analyze(result: TrialResult, g: dict) -> dict:
    end = result.marks["U.end"]
    segs = speech_segments(result.model, result.sr)
    early = [s for s in segs if s[0] < end]
    after = [s for s in segs if s[0] >= end]
    stops = [e for e in result.events if e["type"] == "user_speech_stopped" and e.get("audio_ms") is not None]
    final = next((e for e in stops if e["audio_ms"] >= end - 150), None)
    return {
        "outcome": "early_speech" if early else ("clean" if after else "no_response"),
        "final_gap_ms": None if not after else round(after[0][0] - end, 1),
        "commit_after_end_ms": None if final is None else round(final["audio_ms"] - end, 1),
        "decide_after_commit_ms": None if final is None else round(final["t_ms"] - final["audio_ms"], 1),
        "turns_ended": len(stops),
        "model_segments": segs,
    }


async def run_trial(g: dict, cfg: FragmentsConfig, clips: dict) -> TrialResult:
    adapter = make_adapter(cfg.system, g["setting"], cfg.model)
    await adapter.connect()
    try:
        trial = Trial(adapter, sr=SR)
        whole, split = clips[(g["stimulus"], g["voice"])]

        async def script(ctx):
            ctx.silence(500)
            if g["condition"] == "whole":
                ctx.play(whole, "U")
            elif g["condition"] == "b_alone":
                ctx.play(split.b, "U")
            else:
                ctx.play(split.a, "A")
                ctx.silence(cfg.pause_ms)
                ctx.play(split.b, "U")
            await ctx.wait_settled(after_ms=ctx.marks["U.end"], quiet_ms=1500, give_up_ms=11000)

        return await trial.run(script)
    finally:
        await adapter.close()


async def run_fragments(cfg: FragmentsConfig) -> Path:
    run_dir = new_run_dir(cfg.out, "fragments", cfg.system)
    (run_dir / "stimuli").mkdir(exist_ok=True)
    clips = {}
    for voice in cfg.voices:
        tts = make_tts(cfg.tts, voice)
        for sid in cfg.stimuli:
            s = STIMULI[sid]
            whole = await asyncio.to_thread(synth_clip, tts, f"{s['a']} {s['b']}", SR)
            split = await asyncio.to_thread(synth_split, tts, s["a"], s["b"], SR)
            clips[(sid, voice)] = (whole, split)
            write_wav(run_dir / "stimuli" / f"{sid}-{voice}-whole.wav", whole, SR)
    grid = [
        {"setting": st, "stimulus": sid, "voice": v, "condition": c, "rep": r}
        for st in cfg.settings for sid in cfg.stimuli for v in cfg.voices for c in CONDITIONS for r in range(cfg.reps)
    ]
    write_run_meta(run_dir, "fragments", config=asdict(cfg), stimuli={s: STIMULI[s] for s in cfg.stimuli},
                   trials_planned=len(grid))
    await run_grid(
        run_dir, grid,
        run_one=lambda g: run_trial(g, cfg, clips),
        analyze=analyze,
        concurrency=cfg.concurrency, budget_usd=cfg.budget_usd, seed=cfg.seed,
        describe=lambda g: f"{g['setting']:<30} {g['stimulus']:<11} {g['voice']:<6} {g['condition']:<8}",
    )
    return run_dir
