"""Build data.js + audio clips for blog post 1 (gpt-realtime-2.1 turn detection).

Sources:
  pause sweep v2   runs/20260913-225604-pause-sweep-openai   (12 sentences x 2 voices x 5 pauses x 4 settings)
  overlap          runs/20260913-190825-overlap-openai (coral) + runs/20260913-225604-overlap-openai (ash)
  fragments        runs/20260913-225604-fragments-openai
Prints every number quoted in the post so the prose can be checked against it.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PAUSE = ROOT / "runs/20260913-225604-pause-sweep-openai"
PAUSE_V1 = ROOT / "runs/20260913-190315-pause-sweep-openai"  # source of the hero phone-number trial
OVERLAP = [ROOT / "runs/20260913-190825-overlap-openai", ROOT / "runs/20260913-225604-overlap-openai"]
FRAG = ROOT / "runs/20260913-225604-fragments-openai"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/post1"

SET = {"server_vad:silence_duration_ms=500": "server", "semantic_vad:eagerness=high": "sem_high",
       "semantic_vad:eagerness=auto": "sem_auto", "semantic_vad:eagerness=low": "sem_low"}
VALID = ("clean", "cut_in", "talk_over", "no_response")


MAX_SEND_LATENESS_P99_MS = 20.0  # trials where the pacer fell behind (network trouble) are excluded


def rows_of(run: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run / "summary.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if (r.get("send_lateness_ms") or {}).get("p99", 0.0) <= MAX_SEND_LATENESS_P99_MS]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0, 0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def pct(v, q):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.percentile(v, q)), 1)


def envelope(x, sr, step_ms=20):
    n = int(sr * step_ms / 1000); m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    db = 20 * np.log10(np.maximum(np.sqrt((fr ** 2).mean(axis=1)), 1e-5))
    return [round(float(max(0.0, min(1.0, (d + 60) / 50))), 3) for d in db]


def example(run: Path, tid: str, name: str) -> dict:
    tdir = run / "trials" / tid
    d = json.loads((tdir / "trial.json").read_text())
    ev = [json.loads(l) for l in (tdir / "events.jsonl").read_text().splitlines() if l.strip()]
    x, sr = sf.read(tdir / "audio.wav", dtype="float32")
    texts, status, keep, first_audio = {}, {}, [], {}
    for e in ev:
        if e["type"] == "transcript":
            texts[e.get("response_id")] = texts.get(e.get("response_id"), "") + e.get("delta", "")
        if e["type"] == "response_done":
            status[e.get("response_id")] = e.get("status")
        if e["type"] in ("user_speech_started", "user_speech_stopped", "response_created", "response_done", "client_truncate"):
            keep.append({k: e.get(k) for k in ("t_ms", "type", "audio_ms", "status")})
        if e["type"] == "audio" and not e.get("ignored") and e.get("response_id") not in first_audio:
            first_audio[e.get("response_id")] = e["t_ms"]
    keep += [{"t_ms": t, "type": "first_audio"} for t in first_audio.values()]
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tdir / "audio.wav"), "-codec:a", "libmp3lame",
                    "-b:a", "64k", str(OUT / "audio" / f"{name}.mp3")], check=True)
    return {"name": name, "trial": f"{run.name}/{tid}", "setting": SET[d["setting"]],
            "meta": {k: d.get(k) for k in ("stimulus", "voice", "pause_ms", "clip", "offset_ms")},
            "marks": d["marks"], "analysis": {k: d["analysis"].get(k) for k in ("outcome", "final_gap_ms", "reply_gap_ms", "stop_latency_ms")},
            "user": envelope(x[:, 0], sr), "model": envelope(x[:, 1], sr), "duration_ms": round(len(x) / sr * 1000),
            "events": sorted(keep, key=lambda e: e["t_ms"]),
            "responses": [{"text": texts.get(rid, ""), "status": status.get(rid), "heard": rid in first_audio} for rid in status]}


def trials(run: Path):
    for tj in sorted((run / "trials").glob("*/trial.json")):
        yield json.loads(tj.read_text())


def main():
    data = {"pause": {}, "gaps": {}, "anatomy": {}, "overlap": [], "meta": {}, "examples": []}
    pr = [r for r in rows_of(PAUSE) if r["outcome"] in VALID]
    print(f"pause sweep: {len(pr)} valid of {len(rows_of(PAUSE))} rows; outcomes {Counter(r['outcome'] for r in rows_of(PAUSE))}")
    pauses = sorted({r["pause_ms"] for r in pr})
    for s, key in SET.items():
        rs = [r for r in pr if r["setting"] == s]
        pts = []
        for p in pauses:
            g = [r for r in rs if r["pause_ms"] == p]
            ks, ka = sum(bool(r["split_in_pause"]) for r in g), sum(r["outcome"] in ("cut_in", "talk_over") for r in g)
            pts.append({"pause": p, "n": len(g), "split": ks, "audible": ka, "split_ci": wilson(ks, len(g)), "audible_ci": wilson(ka, len(g))})
        data["pause"][key] = pts
        gaps = [r["final_gap_ms"] for r in rs if r.get("final_gap_ms") is not None]
        data["gaps"][key] = {"values": sorted(round(v) for v in gaps), "p10": pct(gaps, 10), "p50": pct(gaps, 50), "p90": pct(gaps, 90), "max": max(gaps)}
        ev, gen = [r.get("endpoint_lag_ms") for r in rs], [r.get("generation_lag_ms") for r in rs]
        resid = [r["final_gap_ms"] - r["endpoint_lag_ms"] - r["generation_lag_ms"] for r in rs
                 if None not in (r.get("final_gap_ms"), r.get("endpoint_lag_ms"), r.get("generation_lag_ms"))]
        commit = [r.get("vad_end_offset_ms") for r in rs]
        data["anatomy"][key] = {"decision": pct(ev, 50), "generation": pct(gen, 50), "playout": pct(resid, 50), "gap": pct(gaps, 50), "commit_offset": pct(commit, 50)}
        by_kind = {k: [sum(r["outcome"] in ("cut_in", "talk_over") for r in rs if r["kind"] == k), sum(1 for r in rs if r["kind"] == k)]
                   for k in ("incomplete", "dictation", "complete")}
        long = [r for r in rs if r["pause_ms"] >= 600]
        data["meta"][key] = {"n": len(rs), "ended_long_pause": [sum(bool(r["split_in_pause"]) for r in long), len(long)],
                             "audible_total": [sum(r["outcome"] in ("cut_in", "talk_over") for r in rs), len(rs)], "audible_by_kind": by_kind,
                             "no_response": sum(r["outcome"] == "no_response" for r in rs)}
        print(f"  {key:9} n={len(rs)} ended(>=600ms)={data['meta'][key]['ended_long_pause']} heard={data['meta'][key]['audible_total']} by kind {by_kind} "
              f"gap p50/p90={data['gaps'][key]['p50']}/{data['gaps'][key]['p90']} anatomy={data['anatomy'][key]} no_resp={data['meta'][key]['no_response']}")
        print("            by pause: " + " ".join(f"{p['pause']}:{p['split']}/{p['audible']}/{p['n']}" for p in pts))

    for s, key in SET.items():
        rs = [r for r in pr if r["setting"] == s]
        print(f"  {key:9} heard by kind " + str({k: f"{v[0]}/{v[1]}" for k, v in data['meta'][key]['audible_by_kind'].items()})
              + " | gen p50 " + str(data['anatomy'][key]['generation']) + " | commit p50 " + str(data['anatomy'][key]['commit_offset']))
    data["meta"]["pause_trials"] = [len(pr), len([r for r in json.loads("[" + ",".join((PAUSE / "summary.jsonl").read_text().split("\n")[:-1]) + "]")])]

    # classifier ceilings: decision time minus commit, max per setting
    kept = {r["trial_id"] for r in pr}
    for s, key in SET.items():
        waits = []
        for d in trials(PAUSE):
            if d["setting"] != s or d["trial_id"] not in kept:
                continue
            a = d["analysis"]
            if a.get("endpoint_lag_ms") is not None and a.get("vad_end_offset_ms") is not None:
                waits.append(a["endpoint_lag_ms"] - a["vad_end_offset_ms"])
        data["meta"][key]["classifier_wait_max"] = round(max(waits)) if waits else None
        data["meta"][key]["classifier_wait_p95"] = pct(waits, 95)
        data["meta"][key]["classifier_wait_p99"] = pct(waits, 99)
        print(f"  {key:9} classifier wait after commit: p95 {pct(waits, 95)} p99 {pct(waits, 99)} max {max(waits) if waits else None}")

    ov = []
    ov_rows_all = []
    for run in OVERLAP:
        ov_rows_all += rows_of(run)
        voice = json.loads((run / "run.json").read_text())["config"].get("voice") or "coral"
        for r in rows_of(run):
            if r["outcome"] in ("kept_talking", "stopped_then_replied", "stopped_silent"):
                ov.append({"setting": SET[r["setting"]], "clip": r["clip"], "kind": r["kind"], "voice": voice, "outcome": r["outcome"],
                           "stop_latency_ms": r.get("stop_latency_ms"), "reply_gap_ms": r.get("reply_gap_ms"), "talk_over_ms": r.get("talk_over_ms")})
    data["overlap"] = ov
    stops = [r["stop_latency_ms"] for r in ov if r["stop_latency_ms"] is not None]
    data["meta"]["overlap_stopped"] = [sum(r["outcome"] != "kept_talking" for r in ov), len(ov)]
    data["meta"]["overlap_stop_p50"] = pct(stops, 50)
    data["meta"]["overlap_stop_range"] = [min(stops), max(stops)]
    print(f"overlap: stopped {data['meta']['overlap_stopped']} stop p50 {pct(stops, 50)} range {data['meta']['overlap_stop_range']} outcomes {Counter(r['outcome'] for r in ov)}")
    for k in ("server", "sem_auto"):
        for kind in ("backchannel", "interrupt"):
            g = [r["reply_gap_ms"] for r in ov if r["setting"] == k and r["kind"] == kind and r["reply_gap_ms"] is not None]
            print(f"  {k} {kind}: reply gap p50 {pct(g, 50)} range {min(g) if g else None}-{max(g) if g else None}")
    for c in ("yeah", "okay", "right", "mmhm"):
        g = [r["reply_gap_ms"] for r in ov if r["setting"] == "sem_auto" and r["clip"] == c]
        print(f"  sem_auto after '{c}': {sorted(round(x) for x in g if x)}")

    fr = rows_of(FRAG)
    data["meta"]["fragments"] = [{k: r.get(k) for k in ("setting", "stimulus", "voice", "condition", "final_gap_ms", "decide_after_commit_ms", "outcome")} for r in fr]
    ocfg = json.loads((OVERLAP[0] / "run.json").read_text())
    data["meta"]["clips"], data["meta"]["question"] = ocfg["clips"], ocfg["question"]
    data["meta"]["stimuli"] = json.loads((PAUSE / "run.json").read_text())["stimuli"]
    allrows = pr + ov_rows_all + fr
    data["meta"]["sessions"] = sum(1 for r in allrows if r["outcome"] not in ("error", "skipped_budget"))
    print(f"analyzed sessions: pause {len(pr)} + overlap {len(ov)} + fragments {len(fr)}")
    data["meta"]["cost_usd"] = round(sum(r.get("cost_usd", 0) for r in allrows), 2)
    lat = [r["send_lateness_ms"]["p99"] for r in allrows if r.get("send_lateness_ms")]
    data["meta"]["harness_p99_max"] = round(max(lat), 2)
    print(f"sessions {data['meta']['sessions']} cost ${data['meta']['cost_usd']} harness p99 max {data['meta']['harness_p99_max']} ms")

    # examples
    def pick(run, pred, prefer=None):
        cands = [d for d in trials(run) if pred(d)]
        if prefer:
            cands.sort(key=prefer)
        return cands[0]["trial_id"] if cands else None

    t = pick(PAUSE, lambda d: SET[d["setting"]] == "server" and d["pause_ms"] == 900 and d["analysis"]["outcome"] == "clean"
             and d["analysis"].get("split_in_pause") and d["stimulus"] in ("booking", "remind", "weather"))
    data["examples"].append(example(PAUSE, t, "hidden-split"))

    def cut_quote(d):
        ev = [json.loads(l) for l in (PAUSE / "trials" / d["trial_id"] / "events.jsonl").read_text().splitlines() if l.strip()]
        text = "".join(e.get("delta", "") for e in ev if e["type"] == "transcript")
        return -("finish" in text.lower() or "go ahead" in text.lower())
    def v1_quote(d):
        ev = [json.loads(l) for l in (PAUSE_V1 / "trials" / d["trial_id"] / "events.jsonl").read_text().splitlines() if l.strip()]
        return "still reading it out" in "".join(e.get("delta", "") for e in ev if e["type"] == "transcript")
    t = pick(PAUSE_V1, lambda d: SET[d["setting"]] == "server" and d["stimulus"] == "phone" and d["analysis"]["outcome"] == "cut_in" and v1_quote(d))
    data["examples"].append(example(PAUSE_V1, t, "audible-cut-in"))
    t = pick(PAUSE, lambda d: SET[d["setting"]] == "sem_low" and d["stimulus"] == "capital" and (d["analysis"].get("final_gap_ms") or 0) > 7000,
             prefer=lambda d: d["pause_ms"])
    data["examples"].append(example(PAUSE, t, "patient-wait"))
    t = pick(OVERLAP[0], lambda d: SET[d["setting"]] == "sem_auto" and d["clip"] == "yeah" and (d["analysis"].get("reply_gap_ms") or 0) > 4000)
    data["examples"].append(example(OVERLAP[0], t, "yeah-stops-it"))
    for e in data["examples"]:
        print(f"example {e['name']}: {e['trial']} {e['meta']} {e['analysis']} | " + " || ".join(f"[{r['status']}] {r['text'][:90]}" for r in e["responses"]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.js").write_text("window.DATA = " + json.dumps(data, default=float, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT / 'data.js'} ({(OUT / 'data.js').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
