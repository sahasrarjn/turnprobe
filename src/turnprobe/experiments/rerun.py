"""Re-run selected trials of an existing run in place, with the current code.

Used when a harness bug affected a subset of trials: the replaced trials keep their grid
parameters and trial ids, and their summary rows are overwritten.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

import soundfile as sf

from ..tts import make_tts, synth_clip, synth_split
from .runner import save_trial, trial_cost

GRID_KEYS = ("setting", "stimulus", "kind", "voice", "pause_ms", "rep", "clip", "offset_ms", "condition")


def ended_early(run_dir: Path) -> list[str]:
    """Trials where a playback flush happened and the recording stops < 2.6 s after the user's last
    audio with no model speech after it: the symptom of the flushed-onset bug."""
    run = json.loads((run_dir / "run.json").read_text())
    end_key = {"overlap": "X.end", "fragments": "U.end"}.get(run["experiment"], "B.end")
    out = []
    for tj in sorted((run_dir / "trials").glob("*/trial.json")):
        d = json.loads(tj.read_text())
        end = d["marks"].get(end_key)
        if end is None:
            continue
        events = [json.loads(l) for l in (tj.parent / "events.jsonl").read_text().splitlines() if l.strip()]
        if not any(e["type"] == "client_truncate" for e in events):
            continue
        dur_ms = sf.info(tj.parent / "audio.wav").duration * 1000
        spoke_after = any(s[0] >= end for s in d["analysis"].get("model_segments", []))
        if dur_ms - end < 2600 and not spoke_after:
            out.append(d["trial_id"])
    return out


async def rerun(run_dir: Path, trial_ids: list[str], concurrency: int = 2, budget_usd: float | None = None) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    cfg, exp = run["config"], run["experiment"]
    summary = {json.loads(l)["trial_id"]: json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines() if l.strip()}

    def meta_for(tid: str) -> dict:
        tj = run_dir / "trials" / tid / "trial.json"
        return json.loads(tj.read_text()) if tj.exists() else summary[tid]  # errored trials have no trial dir

    metas = {tid: meta_for(tid) for tid in trial_ids}
    trial_ids = list(trial_ids)
    random.Random(17).shuffle(trial_ids)  # if the budget runs out, the gaps land at random

    if exp == "overlap":
        from . import overlap as mod
        ocfg = mod.OverlapConfig(**cfg)
        tts = make_tts(ocfg.tts, ocfg.voice)
        question = synth_clip(tts, mod.QUESTION, mod.SR)
        clips = {c: synth_clip(tts, mod.CLIPS[c]["text"], mod.SR) for c in ocfg.clips}
        run_one = lambda g: mod.run_trial(g, ocfg, question, clips)
    elif exp == "fragments":
        from . import fragments as mod
        fcfg = mod.FragmentsConfig(**cfg)
        clips = {}
        for v in fcfg.voices:
            tts = make_tts(fcfg.tts, v)
            for sid in fcfg.stimuli:
                s = mod.STIMULI[sid]
                clips[(sid, v)] = (synth_clip(tts, f"{s['a']} {s['b']}", mod.SR), synth_split(tts, s["a"], s["b"], mod.SR))
        run_one = lambda g: mod.run_trial(g, fcfg, clips)
    else:
        from . import pause_sweep as mod
        pcfg = mod.SweepConfig(**cfg)
        cache = {}

        def clip_for(g):
            key = (g["stimulus"], g.get("voice"))
            if key not in cache:
                s = mod.STIMULI[g["stimulus"]]
                cache[key] = synth_split(make_tts(pcfg.tts, g.get("voice") or pcfg.voice), s["a"], s["b"], mod.SR)
            return cache[key]
        run_one = lambda g: mod.run_trial(pcfg.system, g["setting"], pcfg.model, clip_for(g), g["pause_ms"])

    sem = asyncio.Semaphore(concurrency)
    new_rows: dict[str, dict] = {}
    spent = 0.0

    async def one(tid: str):
        nonlocal spent
        d = metas[tid]
        g = {k: d[k] for k in GRID_KEYS if k in d}
        meta = {"trial_id": tid, **g, "started": datetime.now().astimezone().isoformat(), "rerun": True}
        async with sem:
            if budget_usd is not None and spent >= budget_usd:
                print(f"skipped {tid}: re-run budget reached", flush=True)
                return
            try:
                result = await run_one(g)
            except Exception as exc:
                print(f"re-run {tid} failed again: {type(exc).__name__}: {exc}", flush=True)
                return
        analysis = {**mod.analyze(result, g), "cost_usd": round(trial_cost(result), 5), "send_lateness_ms": result.send_lateness_ms}
        save_trial(run_dir / "trials" / tid, result, meta, analysis)
        new_rows[tid] = {**meta, **{k: v for k, v in analysis.items() if k != "model_segments"}}
        spent += analysis["cost_usd"]
        print(f"re-ran {tid} {g} → {analysis['outcome']} (${analysis['cost_usd']:.3f})", flush=True)

    await asyncio.gather(*(one(t) for t in trial_ids))
    rows = [json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines() if l.strip()]
    with open(run_dir / "summary.jsonl", "w") as f:
        for r in rows:
            if r["trial_id"] in new_rows:
                # keep the original spend on record too: the first attempt was also billed
                r = {**new_rows[r["trial_id"]], "cost_usd": new_rows[r["trial_id"]]["cost_usd"] + r.get("cost_usd", 0.0)}
            f.write(json.dumps(r, default=str) + "\n")
    return spent
