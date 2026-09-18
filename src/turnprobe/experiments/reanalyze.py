"""Re-score saved trials with the current analysis code, without re-running any sessions."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from ..session import TrialResult


def reanalyze(run_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    if run["experiment"] == "overlap":
        from .overlap import analyze
    elif run["experiment"] == "hard_cases":
        from .hard_cases import analyze
    else:
        from .pause_sweep import analyze
    old = {json.loads(l)["trial_id"]: json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines() if l.strip()}
    rows = []
    for tid, row in sorted(old.items()):
        tdir = run_dir / "trials" / tid
        if not (tdir / "trial.json").exists():
            rows.append(row)
            continue
        trial = json.loads((tdir / "trial.json").read_text())
        audio, sr = sf.read(tdir / "audio.wav", dtype="float32")
        events = [json.loads(l) for l in (tdir / "events.jsonl").read_text().splitlines() if l.strip()]
        result = TrialResult(sr=sr, user=audio[:, 0], mic=audio[:, 0], model=audio[:, 1], events=events,
                             marks=trial["marks"], send_lateness_ms=trial["analysis"].get("send_lateness_ms", {}),
                             adapter=trial["adapter"])
        g = {k: trial[k] for k in trial if k not in ("marks", "analysis", "adapter")}
        analysis = {**trial["analysis"], **analyze(result, g)}
        trial["analysis"] = analysis
        (tdir / "trial.json").write_text(json.dumps(trial, indent=2, default=str))
        rows.append({**row, **{k: v for k, v in analysis.items() if k != "model_segments"}})
    with open(run_dir / "summary.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return len(rows)
