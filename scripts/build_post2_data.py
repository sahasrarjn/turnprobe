"""Build data.js + audio clips for blog post 2: GPT-Live-1 vs gpt-realtime-2.1 on the same tests.

Sources:
  gpt-realtime-2.1  pause sweep v2    runs/20260913-225604-pause-sweep-openai   (4 settings)
                    overlap           runs/20260913-190825-overlap-openai (coral), runs/20260913-225604-overlap-openai (ash)
                    fragments         runs/20260913-225604-fragments-openai
  gpt-live-1        pause sweep v2    newest runs/*-pause-sweep-openai_live with voices coral,ash
                    overlap           runs/20260913-223544-overlap-openai_live (coral), newest ash overlap run
                    fragments         newest runs/*-fragments-openai_live
Prints every number quoted in the post.

Usage: uv run python scripts/build_post2_data.py <post dir>
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1])
RUNS = ROOT / "runs"
MAX_LATE = 20.0
VALID = ("clean", "cut_in", "talk_over", "no_response")
SYS = {"server_vad:silence_duration_ms=500": "server", "semantic_vad:eagerness=high": "sem_high",
       "semantic_vad:eagerness=auto": "sem_auto", "semantic_vad:eagerness=low": "sem_low", "default:": "live"}
ORDER = ["server", "sem_high", "sem_auto", "sem_low", "live"]
ACK = set("mm-hmm mhm mm hmm hm uh-huh got it okay ok go on ahead sure alright right yeah yes".split())


def latest(pattern: str, want=None) -> Path:
    runs = sorted(RUNS.glob(pattern))
    if want:
        runs = [r for r in runs if want(json.loads((r / "run.json").read_text()))]
    return runs[-1]


PAUSE_RT = RUNS / "20260913-225604-pause-sweep-openai"
PAUSE_LIVE = latest("*-pause-sweep-openai_live", lambda r: r["config"].get("voices") == ["coral", "ash"])
OVER_RT = [RUNS / "20260913-190825-overlap-openai", RUNS / "20260913-225604-overlap-openai"]
OVER_LIVE = [RUNS / "20260913-223544-overlap-openai_live", latest("*-overlap-openai_live", lambda r: r["config"].get("voice") == "ash")]
FRAG_RT = RUNS / "20260913-225604-fragments-openai"
FRAG_LIVE = latest("*-fragments-openai_live")


def rows(run: Path) -> list[dict]:
    out = [json.loads(l) for l in (run / "summary.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in out if (r.get("send_lateness_ms") or {}).get("p99", 0.0) <= MAX_LATE]


def trial(run: Path, tid: str):
    d = json.loads((run / "trials" / tid / "trial.json").read_text())
    ev = [json.loads(l) for l in (run / "trials" / tid / "events.jsonl").read_text().splitlines() if l.strip()]
    return d, ev


def pct(v, q):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.percentile(v, q)), 1)


def classify(text: str) -> str:
    toks = re.findall(r"[a-z][a-z'\-]*|\d+", text.lower())
    return "ack" if toks and all(t in ACK for t in toks) else "reply"


def live_words(d, ev):
    """Model transcript words placed on the playout timeline: the transcript clock runs ahead of
    playback by a roughly constant lag, so shift it to line up the first word with the first audio."""
    words = [e for e in ev if e["type"] == "transcript" and e.get("start_ms") is not None]
    segs = d["analysis"].get("model_segments") or []
    if not words or not segs:
        return []
    shift = segs[0][0] - words[0]["start_ms"]
    return [(w["start_ms"] + shift, w["delta"]) for w in words]


def early_text(sysname, d, ev):
    end = d["marks"]["B.end"]
    if sysname == "live":
        return "".join(w for t, w in live_words(d, ev) if t < end).strip()
    heard = {e.get("response_id") for e in ev if e["type"] == "audio" and not e.get("ignored") and (e.get("play_ms") or 1e12) < end}
    return "".join(e.get("delta", "") for e in ev if e["type"] == "transcript" and e.get("response_id") in heard).strip()


def envelope(x, sr, step_ms=20):
    n = int(sr * step_ms / 1000); m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    db = 20 * np.log10(np.maximum(np.sqrt((fr ** 2).mean(axis=1)), 1e-5))
    return [round(float(max(0.0, min(1.0, (v + 60) / 50))), 3) for v in db]


def pick(run: Path, clip: str, outcome: str) -> str:
    """Trial for a clip/outcome whose stop latency is closest to that group's median (a typical case)."""
    cands = [r for r in rows(run) if r["clip"] == clip and r["outcome"] == outcome]
    if not cands:
        raise RuntimeError(f"no {clip}/{outcome} trial in {run}")
    vals = [r.get("stop_latency_ms") or 0 for r in cands]
    med = float(np.median(vals))
    return min(cands, key=lambda r: abs((r.get("stop_latency_ms") or 0) - med))["trial_id"]


def rt_words(ev):
    """gpt-realtime words placed on the playout timeline. Transcripts come per output item without word
    timings, so each item's words are spread evenly over that item's audio, and words past the point where
    the client truncated playback are dropped (the listener never heard them). Used only for quotes."""
    text = {e["raw"]["item_id"]: e["raw"]["transcript"] for e in ev
            if e["type"] == "native" and e.get("name") == "response.output_audio_transcript.done"}
    audio = {}
    for e in ev:
        if e["type"] == "audio" and not e.get("ignored") and e.get("play_ms") is not None:
            a = audio.setdefault(e["item_id"], [e["play_ms"], 0.0])
            a[1] += e["samples"] / 24.0
    cut = {e["item_id"]: e["audio_end_ms"] for e in ev if e["type"] == "client_truncate"}
    out = []
    for item, (start, total) in audio.items():
        toks = re.findall(r"\S+\s*", text.get(item, ""))
        heard = min(total, cut.get(item, total))
        for i, tok in enumerate(toks):
            rel = total * i / max(1, len(toks))
            if rel < heard:
                out.append((start + rel, tok))
    return sorted(out)


def quote(words, t0, t1, limit=170):
    ws = [w for t, w in words if t0 <= t < t1]
    txt = "".join(ws).strip()
    if not txt:
        return ""
    lead = "…" if any(t < t0 for t, _ in words) and t0 > 0 else ""
    more = any(t >= t1 for t, _ in words)
    if len(txt) > limit:
        txt = txt[:limit].rsplit(" ", 1)[0]; more = True
    tail = "" if not more else (" …" if txt[-1] in ".!?" else "…")
    return f"“{lead}{txt}{tail}”"


def cut_audio(tdir, name, w0, w1):
    fade = 0.25
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{w0 / 1000:.3f}", "-t", f"{(w1 - w0) / 1000:.3f}", "-i", str(tdir / "audio.wav"),
                    "-af", f"afade=t=out:st={(w1 - w0) / 1000 - fade:.3f}:d={fade}", "-codec:a", "libmp3lame", "-b:a", "64k",
                    str(OUT / "audio" / f"{name}.mp3")], check=True)


def clip(run, tid, name, window, result):
    """One recorded trial cut to `window` (absolute ms, or a callable of the trial's marks/analysis).
    Everything exported is relative to the window start, so the player clock matches the trace."""
    assert tid in {r["trial_id"] for r in rows(run)}, f"{run.name}/{tid} excluded by lateness filter"
    d, ev = trial(run, tid)
    cfg = json.loads((run / "run.json").read_text())
    live = d["setting"] == "default:"
    m, a = d["marks"], d["analysis"]
    words = live_words(d, ev) if live else rt_words(ev)
    tdir = run / "trials" / tid
    x, sr = sf.read(tdir / "audio.wav", dtype="float32")
    dur = round(len(x) / sr * 1000)
    w0, w1 = window(m, a) if callable(window) else window
    w0, w1 = max(0, round(w0)), min(dur, round(w1))
    cut_audio(tdir, name, w0, w1)
    i0, i1 = w0 // 20, w1 // 20
    marks, lines = [], []
    onsets = [s0 for s0, _ in a.get("model_segments") or []]
    if "X.start" in m:  # overlap trial
        said = cfg["clips"][d["clip"]]["text"]
        marks.append((m["X.start"], "u", f"“{said.rstrip('.')}”"))
        if a["outcome"] != "kept_talking":
            marks.append((m["X.start"] + a["stop_latency_ms"], "m", f"stops +{a['stop_latency_ms'] / 1000:.2f} s"))
            if a.get("reply_gap_ms") is not None and a["reply_gap_ms"] > 0:
                marks.append((m["X.end"] + a["reply_gap_ms"], "m", "resumes" if a["kind"] == "backchannel" else "replies"))
        lines = [("model", quote(words, w0, m["X.start"])), ("caller", f"“{said}”"),
                 ("model", quote(words, (m["X.end"] + a["reply_gap_ms"] - 500) if a["outcome"] != "kept_talking" else m["X.start"], w1))]
    else:  # pause trial
        stim = cfg["stimuli"][d["stimulus"]]
        marks.append((m["A.end"], "u", f"{d['pause_ms'] / 1000:g} s pause"))
        marks.append((m["B.end"], "u", "caller done"))
        early = [o for o in onsets if m["A.end"] < o < m["B.end"]]
        late = [o for o in onsets if o >= m["B.end"]]
        if early:
            marks.append((early[0], "m", "speaks early"))
        if late and late[0] < w1:
            marks.append((late[0], "m", f"reply +{(late[0] - m['B.end']) / 1000:.1f} s"))
        # Group words by the audio segment nearest to them, so a sentence that runs past the caller's
        # restart stays in one quote; segments that start before the caller finishes are the early speech.
        segs = a.get("model_segments") or []
        def seg_of(t):
            return min(range(len(segs)), key=lambda i: 0 if segs[i][0] <= t <= segs[i][1] else min(abs(t - segs[i][0]), abs(t - segs[i][1])))
        grouped = {}
        for t, w in words:
            if segs:
                grouped.setdefault(seg_of(t), []).append(w)
        def text(pred, limit):
            txt = "".join(w for i in sorted(grouped) if pred(segs[i][0]) for w in grouped[i]).strip()
            if len(txt) > limit:
                txt = txt[:limit].rsplit(" ", 1)[0] + "…"
            return f"“{txt}”" if txt else ""
        lines = [("caller", f"“{stim['a']}”")]
        if early:
            lines.append(("model", text(lambda o: m["A.end"] < o < m["B.end"], 170)))
        lines.append(("caller", f"“{stim['b']}”"))
        if late and late[0] < w1:
            lines.append(("model", text(lambda o: o >= m["B.end"], 120)))
    return {"name": name, "trial": f"{run.name}/{tid}", "system": "live" if live else SYS[d["setting"]], "voice": d.get("voice") or cfg["config"].get("voice"),
            "window": [w0, w1], "result": result,
            "user": envelope(x[:, 0], sr)[i0:i1], "model": envelope(x[:, 1], sr)[i0:i1],
            "marks": [[round(t - w0), k, lab] for t, k, lab in sorted(marks) if w0 <= t <= w1],
            "lines": [[who, q] for who, q in lines if q]}


def main():
    data = {"order": ORDER, "early": {}, "gaps": {}, "overlap": [], "fragments": {}, "meta": {}, "pairs": {}, "gallery": []}
    print("runs:", PAUSE_LIVE.name, [r.name for r in OVER_LIVE], FRAG_LIVE.name)

    # ------------------------------------------------------------------ pause sweep
    early_examples = {}
    for run in (PAUSE_RT, PAUSE_LIVE):
        for r in rows(run):
            if r["outcome"] not in VALID:
                continue
            key = SYS[r["setting"]]
            e = data["early"].setdefault(key, {k: {"n": 0, "none": 0, "ack": 0, "reply": 0} for k in ("incomplete", "dictation", "complete", "all")})
            cat = "none"
            if r["outcome"] in ("cut_in", "talk_over"):
                d, ev = trial(run, r["trial_id"])
                txt = early_text(key, d, ev)
                cat = classify(txt)
                if key == "live":
                    early_examples.setdefault((r["kind"], cat), []).append((r["trial_id"], r["stimulus"], r["voice"], r["pause_ms"], txt))
            for k in (r["kind"], "all", f"pause_{r['pause_ms']}"):
                e.setdefault(k, {"n": 0, "none": 0, "ack": 0, "reply": 0})
                e[k]["n"] += 1; e[k][cat] += 1
            g = data["gaps"].setdefault(key, {"values": []})
            if r.get("final_gap_ms") is not None:
                g["values"].append(round(r["final_gap_ms"]))
    for key in ORDER:
        v = data["gaps"][key]["values"]; v.sort()
        data["gaps"][key].update(p10=pct(v, 10), p50=pct(v, 50), p90=pct(v, 90), max=max(v), n=len(v))
        e = data["early"][key]
        print(f"  {key:9} early speech " + "  ".join(f"{k}: none {e[k]['none']} ack {e[k]['ack']} reply {e[k]['reply']} / {e[k]['n']}" for k in e)
              + f" | gap p50 {data['gaps'][key]['p50']} p90 {data['gaps'][key]['p90']} max {data['gaps'][key]['max']} n {len(v)}")
    for (kind, cat), xs in sorted(early_examples.items()):
        print(f"  live {kind}/{cat}:", [(t, s, v, p, txt[:50]) for t, s, v, p, txt in xs])

    # ------------------------------------------------------------------ overlap
    for run in OVER_RT + OVER_LIVE:
        voice = json.loads((run / "run.json").read_text())["config"]["voice"]
        for r in rows(run):
            key = SYS[r["setting"]]
            data["overlap"].append({"system": key, "voice": voice, "clip": r["clip"], "kind": r["kind"], "outcome": r["outcome"],
                                    "stop_latency_ms": r.get("stop_latency_ms"), "reply_gap_ms": r.get("reply_gap_ms"), "run": run.name, "trial": r["trial_id"]})
    for key in ("server", "sem_auto", "live"):
        rs = [r for r in data["overlap"] if r["system"] == key]
        bc = [r for r in rs if r["kind"] == "backchannel"]; it = [r for r in rs if r["kind"] == "interrupt"]
        stops = sorted(r["stop_latency_ms"] for r in it if r["stop_latency_ms"] is not None)
        bstops = sorted(r["stop_latency_ms"] for r in bc if r["stop_latency_ms"] is not None)
        data["meta"][f"overlap_{key}"] = {"bc_kept": sum(r["outcome"] == "kept_talking" for r in bc), "bc_n": len(bc),
                                          "int_stopped": sum(r["outcome"] != "kept_talking" for r in it), "int_n": len(it),
                                          "int_stop_p50": pct(stops, 50), "int_stop_min": stops[0] if stops else None, "int_stop_max": stops[-1] if stops else None}
        print(f"  overlap {key:9} backchannel kept talking {data['meta'][f'overlap_{key}']['bc_kept']}/{len(bc)} | interrupts stopped "
              f"{data['meta'][f'overlap_{key}']['int_stopped']}/{len(it)} stop p50 {pct(stops, 50)} range {stops[:1]}..{stops[-1:]} | backchannel stops {bstops}")
        print("     ", Counter((r["clip"], r["outcome"]) for r in rs))

    # ------------------------------------------------------------------ fragments
    for run in (FRAG_RT, FRAG_LIVE):
        for r in rows(run):
            key = SYS[r["setting"]]
            if r["outcome"] == "error":
                continue
            data["fragments"].setdefault(key, {}).setdefault(f"{r['stimulus']}/{r['voice']}", {})[r["condition"]] = r.get("final_gap_ms")
    for key, tab in data["fragments"].items():
        print(f"  fragments {key}:", {k: {c: (None if v is None else round(v / 1000, 1)) for c, v in cond.items()} for k, cond in sorted(tab.items())})

    # ------------------------------------------------------------------ meta
    stim = json.loads((PAUSE_RT / "run.json").read_text())["stimuli"]
    data["meta"].update(stimuli=stim, clips=json.loads((OVER_RT[0] / "run.json").read_text())["clips"],
                        question=json.loads((OVER_RT[0] / "run.json").read_text())["question"],
                        live_cost=round(sum(r.get("cost_usd", 0) for run in [PAUSE_LIVE, *OVER_LIVE, FRAG_LIVE]
                                            for r in (json.loads(l) for l in (run / "summary.jsonl").read_text().splitlines() if l.strip())), 2),
                        pause_n={k: data["early"][k]["all"]["n"] for k in ORDER})
    print("meta:", {k: data["meta"][k] for k in ("live_cost", "pause_n")})

    # ------------------------------------------------------------------ audio: paired trials and a listening gallery
    over = lambda pre, post: (lambda m, a: (m["X.start"] - pre, m["X.start"] + post))
    def pause_win(m, a, tail=2400, cap=14000):
        late = [s0 for s0, _ in a.get("model_segments") or [] if s0 >= m["B.end"]]
        end = (late[0] + tail) if late else m["B.end"] + 2000
        return m["A.start"] - 400, min(end, m["A.start"] - 400 + cap)
    OR, OL = OVER_RT[1], OVER_LIVE[1]  # both ash
    data["pairs"] = {
        "yeah": [clip(OR, "t0003", "pair-yeah-rt", over(2600, 4600), "Stops within 0.1 s of “yeah”, then starts the walkthrough again."),
                 clip(OL, "t0007", "pair-yeah-live", over(2600, 4600), "Keeps talking.")],
        "stop": [clip(OR, "t0005", "pair-stop-rt", over(2600, 4600), "Stops 0.14 s after the caller starts."),
                 clip(OL, "t0012", "pair-stop-live", over(2600, 4600), "Talks over the caller for 1.5 s, then stops.")],
        "egg": [clip(PAUSE_RT, "t0153", "pair-egg-rt", (100, 18300), "Waits for the whole question, then 8.3 s more."),
                clip(PAUSE_LIVE, "t0077", "pair-egg-live", (100, 18300), "Answers during the pause, before the question is finished.")],
    }
    OL0 = OVER_LIVE[0]  # coral
    data["gallery"] = [
        clip(PAUSE_LIVE, "t0017", "g-phone-live", pause_win, "Mid phone number: says “okay” and tells the caller to go on."),
        clip(PAUSE_RT, "t0031", "g-phone-rt", pause_win, "Same sentence and voice: says almost the same thing."),
        clip(PAUSE_LIVE, "t0076", "g-gate-live", pause_win, "Reads the gate code back before the caller has finished it."),
        clip(PAUSE_LIVE, "t0097", "g-remind-live", pause_win, "Agrees to a reminder before the caller says when."),
        clip(PAUSE_RT, "t0135", "g-remind-rt", pause_win, "Same sentence: asks “When you get where?”"),
        clip(PAUSE_LIVE, "t0070", "g-capital-live", pause_win, "Asks which country before the caller says it."),
        clip(OL, "t0003", "g-mmhm-live", over(2400, 3600), "Talks straight through “mm-hmm”."),
        clip(OL0, "t0015", "g-okay-live", over(2400, 4200), "The one backchannel it stopped for."),
        clip(OL0, "t0020", "g-stop-live", over(2400, 4600), "The one “wait, stop” it talked through."),
        clip(OL0, "t0017", "g-password-live", over(2400, 7400), "Stops, answers the question, carries on."),
    ]
    for c in [*[c for p in data["pairs"].values() for c in p], *data["gallery"]]:
        print(f"  {c['name']:16} {c['trial']} {c['system']}/{c['voice']} window {c['window']} ({(c['window'][1] - c['window'][0]) / 1000:.1f} s)")
        print("       marks:", c["marks"])
        for who, q in c["lines"]:
            print(f"       {who:6} {q}")

    (OUT / "data.js").write_text("window.DATA = " + json.dumps(data, default=float) + ";\n")
    print(f"wrote {OUT / 'data.js'}")
    return data, early_examples


if __name__ == "__main__":
    main()
