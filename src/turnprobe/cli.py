"""turnprobe command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .experiments.overlap import CLIPS, OverlapConfig, run_overlap
from .experiments.pause_sweep import STIMULI, SweepConfig, run_sweep
from .report import build_report


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="turnprobe")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("pause-sweep", help="mid-sentence pause tolerance")
    ps.add_argument("--system", required=True, choices=["mock", "openai", "openai_live"])
    ps.add_argument("--setting", action="append", required=True,
                    help="e.g. server_vad:silence_duration_ms=500  semantic_vad:eagerness=low  (repeatable)")
    ps.add_argument("--pauses", type=_ints, default=_ints("300,600,900,1200,1600,2000,2500"))
    ps.add_argument("--reps", type=int, default=3)
    ps.add_argument("--stimuli", default="all", help=f"comma list from: {', '.join(STIMULI)}")
    ps.add_argument("--tts", default="say", choices=["say", "openai"])
    ps.add_argument("--voice")
    ps.add_argument("--voices", help="comma list of TTS voices; overrides --voice")
    ps.add_argument("--model")
    ps.add_argument("--concurrency", type=int, default=1)
    ps.add_argument("--out", default="runs")
    ps.add_argument("--budget-usd", type=float, default=None, help="stop starting trials once estimated spend reaches this")

    ov = sub.add_parser("overlap", help='"mm-hm" vs "wait, stop" while the model is talking')
    ov.add_argument("--system", required=True, choices=["mock", "openai", "openai_live"])
    ov.add_argument("--setting", action="append", required=True)
    ov.add_argument("--clips", default="all", help=f"comma list from: {', '.join(CLIPS)}")
    ov.add_argument("--offsets", type=_ints, default=_ints("2500"), help="ms after model audio onset")
    ov.add_argument("--reps", type=int, default=3)
    ov.add_argument("--tts", default="say", choices=["say", "openai"])
    ov.add_argument("--voice")
    ov.add_argument("--model")
    ov.add_argument("--concurrency", type=int, default=1)
    ov.add_argument("--out", default="runs")
    ov.add_argument("--budget-usd", type=float, default=None)

    fr = sub.add_parser("fragments", help="control: whole sentence vs second half alone vs split with a pause")
    fr.add_argument("--system", required=True, choices=["mock", "openai", "openai_live"])
    fr.add_argument("--setting", action="append", required=True)
    fr.add_argument("--stimuli", default="capital,weather,booking,reservation")
    fr.add_argument("--voices", default="coral,ash")
    fr.add_argument("--pause-ms", type=int, default=1500)
    fr.add_argument("--model")
    fr.add_argument("--concurrency", type=int, default=1)
    fr.add_argument("--out", default="runs")
    fr.add_argument("--budget-usd", type=float, default=None)

    rr = sub.add_parser("rerun", help="re-run selected trials in place (e.g. after a harness fix)")
    rr.add_argument("run_dir")
    rr.add_argument("--trials", help="comma list of trial ids; default: auto-detect trials that ended early")
    rr.add_argument("--dry-run", action="store_true")
    rr.add_argument("--concurrency", type=int, default=2)
    rr.add_argument("--budget-usd", type=float, default=None)

    rp = sub.add_parser("report", help="build report.html for a run directory")
    rp.add_argument("run_dir")

    ra = sub.add_parser("reanalyze", help="re-score saved trials with current analysis code, then rebuild the report")
    ra.add_argument("run_dir")

    cal = sub.add_parser("calibrate", help="measure harness timing error against the mock bot")
    cal.add_argument("--reps", type=int, default=2)

    args = p.parse_args(argv)

    if args.cmd == "rerun":
        from .experiments.rerun import ended_early, rerun

        run_dir = Path(args.run_dir)
        ids = args.trials.split(",") if args.trials else ended_early(run_dir)
        print(f"{len(ids)} trials to re-run: {', '.join(ids)}")
        if ids and not args.dry_run:
            spent = asyncio.run(rerun(run_dir, ids, concurrency=args.concurrency, budget_usd=args.budget_usd))
            print(f"re-run spend ${spent:.2f}")
            print(build_report(run_dir))
        return 0

    if args.cmd == "reanalyze":
        from .experiments.reanalyze import reanalyze

        print(f"re-scored {reanalyze(args.run_dir)} trials")
        print(build_report(args.run_dir))
        return 0

    if args.cmd == "report":
        print(build_report(args.run_dir))
        return 0

    if args.cmd == "calibrate":
        settings = ["mock:silence_duration_ms=500,response_delay_ms=300",
                    "mock:silence_duration_ms=800,response_delay_ms=150"]
        cfg = SweepConfig(system="mock", settings=settings, pauses=[300, 1400], reps=args.reps,
                          stimuli=["booking"], out="runs/calibration")
        run_dir = asyncio.run(run_sweep(cfg))
        rows = [json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines()]
        # The bot reports where *its* detector saw speech end (vad_end_offset_ms, relative to the
        # gold clip end). Expected onset = that point + silence timer + response delay. Whatever
        # remains is harness error: pacing, event timing, playout placement, offline labeling.
        print("\nsetting                                              pause  outcome   gap    detector  harness err")
        errors = []
        for r in sorted(rows, key=lambda r: (r["setting"], r["pause_ms"])):
            params = dict(kv.split("=") for kv in r["setting"].split(":")[1].split(","))
            timer = float(params["silence_duration_ms"]) + float(params["response_delay_ms"])
            gap, det = r.get("final_gap_ms"), r.get("vad_end_offset_ms")
            err = None if gap is None or det is None else round(gap - (det + timer), 1)
            if err is not None:
                errors.append(err)
            print(f"{r['setting']:<52} {r['pause_ms']:>5}  {r['outcome']:<8} {gap!s:>6} {det!s:>9}  {err!s:>8}")
        if errors:
            print(f"\nharness error: min {min(errors):.1f} ms, max {max(errors):.1f} ms, "
                  f"spread {max(errors) - min(errors):.1f} ms")
        print(build_report(run_dir))
        return 0

    if args.cmd == "fragments":
        from .experiments.fragments import FragmentsConfig, run_fragments

        fcfg = FragmentsConfig(system=args.system, settings=args.setting, stimuli=args.stimuli.split(","),
                               voices=args.voices.split(","), pause_ms=args.pause_ms, model=args.model,
                               concurrency=args.concurrency, out=args.out, budget_usd=args.budget_usd)
        run_dir = asyncio.run(run_fragments(fcfg))
        print(run_dir)
        return 0

    if args.cmd == "overlap":
        clips = list(CLIPS) if args.clips == "all" else args.clips.split(",")
        ocfg = OverlapConfig(system=args.system, settings=args.setting, clips=clips, offsets=args.offsets,
                             reps=args.reps, tts=args.tts, voice=args.voice, model=args.model,
                             concurrency=args.concurrency, out=args.out, budget_usd=args.budget_usd)
        n = len(ocfg.settings) * len(clips) * len(ocfg.offsets) * ocfg.reps
        print(f"{n} trials planned", file=sys.stderr)
        print(build_report(asyncio.run(run_overlap(ocfg))))
        return 0

    stimuli = list(STIMULI) if args.stimuli == "all" else args.stimuli.split(",")
    voices = args.voices.split(",") if args.voices else None
    cfg = SweepConfig(system=args.system, settings=args.setting, pauses=args.pauses, reps=args.reps,
                      stimuli=stimuli, tts=args.tts, voice=args.voice, voices=voices, model=args.model,
                      concurrency=args.concurrency, out=args.out, budget_usd=args.budget_usd)
    trials = len(cfg.settings) * len(stimuli) * len(voices or [None]) * len(cfg.pauses) * cfg.reps
    print(f"{trials} trials planned", file=sys.stderr)
    run_dir = asyncio.run(run_sweep(cfg))
    print(build_report(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
