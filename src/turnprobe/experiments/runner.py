"""Shared machinery for experiments: run a shuffled grid of trials with a concurrency limit and a
spend cap, save each trial's audio/events, and append one summary row per trial."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from ..audio import write_wav
from ..pricing import openai_usage_cost
from ..session import TrialResult


def new_run_dir(out: str, experiment: str, system: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out) / f"{stamp}-{experiment}-{system}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_meta(run_dir: Path, experiment: str, **fields) -> None:
    meta = {
        "experiment": experiment,
        "turnprobe_version": __version__,
        "started": datetime.now().astimezone().isoformat(),
        **fields,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))


def trial_cost(result: TrialResult) -> float:
    if result.adapter.get("system") == "openai_live":
        from ..adapters.openai_live import OpenAILive

        secs = result.adapter.get("usage_seconds") or (len(result.mic) / result.sr + 2.0)
        return secs / 60.0 * OpenAILive.PRICE_PER_MINUTE
    if result.adapter.get("system") != "openai":
        return 0.0
    model = result.adapter.get("model", "")
    return sum(openai_usage_cost(model, e.get("usage")) for e in result.events if e["type"] == "response_done")


def save_trial(trial_dir: Path, result: TrialResult, meta: dict, analysis: dict) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_wav(trial_dir / "audio.wav", np.stack([result.mic, result.model], axis=1), result.sr)
    with open(trial_dir / "events.jsonl", "w") as f:
        for e in result.events:
            f.write(json.dumps(e, default=str) + "\n")
    (trial_dir / "trial.json").write_text(
        json.dumps({**meta, "marks": result.marks, "analysis": analysis, "adapter": result.adapter}, indent=2, default=str)
    )


async def run_grid(
    run_dir: Path,
    grid: list[dict],
    run_one: Callable[[dict], Awaitable[TrialResult]],
    analyze: Callable[[TrialResult, dict], dict],
    concurrency: int = 1,
    budget_usd: float | None = None,
    seed: int = 7,
    describe: Callable[[dict], str] = lambda g: json.dumps(g),
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> None:
    grid = list(grid)
    random.Random(seed).shuffle(grid)
    summary_path = run_dir / "summary.jsonl"
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    state = {"done": 0, "spent": 0.0}

    async def one(i: int, g: dict) -> None:
        trial_id = f"t{i:04d}"
        meta = {"trial_id": trial_id, **g, "started": datetime.now().astimezone().isoformat()}
        async with sem:
            t = time.monotonic()
            if budget_usd is not None and state["spent"] >= budget_usd:
                # Spend is counted when trials finish, so overshoot is at most `concurrency` trials.
                analysis = {"outcome": "skipped_budget"}
            else:
                try:
                    result = await run_one(g)
                    analysis = {**analyze(result, g), "cost_usd": round(trial_cost(result), 5),
                                "send_lateness_ms": result.send_lateness_ms}
                    save_trial(run_dir / "trials" / trial_id, result, meta, analysis)
                except Exception as exc:  # keep going; record the failure
                    analysis = {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
        row = {**meta, **{k: v for k, v in analysis.items() if k != "model_segments"}}
        async with lock:
            state["spent"] += analysis.get("cost_usd", 0.0)
            state["done"] += 1
            with open(summary_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            log(f"[{state['done']}/{len(grid)}] {describe(g)} → {analysis['outcome']} "
                f"spent=${state['spent']:.2f} ({time.monotonic() - t:.1f}s)")

    await asyncio.gather(*(one(i, g) for i, g in enumerate(grid)))
