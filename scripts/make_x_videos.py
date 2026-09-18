"""Render the blog post's audio traces as captioned 1080p videos for an X thread.

Each video is the trace window from the post (same clip audio), drawn as a static card per
event state, with a moving playhead and the current event as a large caption.

Usage: uv run python scripts/make_x_videos.py <blog post dir> <out dir>
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

POST = Path(sys.argv[1])
OUT = Path(sys.argv[2])
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H, SCALE = 1280, 720, 1.5  # CSS px; rendered at 1920x1080
PX0, PX1 = 96, 1224  # plot x range (CSS px)
LANE_U, LANE_M = (176, 286), (338, 448)
AXIS_Y = 470

INK, INK2, INK3, PAPER, RULE = "#1B1D21", "#4A4E58", "#747984", "#FBFAF7", "#E3E0D8"
USER, MODEL, SERVER = "#0087a0", "#B8740F", "#7a5bc9"


def load():
    s = (POST / "data.js").read_text()
    return json.loads(s[s.index("{"): s.rindex("}") + 1])


def ev(e, type_, n=0):
    return [x for x in e["events"] if x["type"] == type_][n]["t_ms"]


def secs(ms, d=1):
    return f"{ms / 1000:.{d}f} s"


def specs(data):
    ex = {e["name"]: e for e in data["examples"]}
    stim = data["meta"]["stimuli"]
    out = []

    e = ex["audible-cut-in"]; s = stim[e["meta"]["stimulus"]]
    out.append(dict(name="1-cut-off-mid-number", e=e,
        title="Cut off halfway through a phone number",
        sub="gpt-realtime-2.1 · server_vad, 500 ms · recorded trial",
        intro=("Caller", f"“{s['a']} … {s['b']}”"),
        events=[
            (ev(e, "user_speech_stopped"), "s", "Caller pauses. 0.7 s in, the server decides the turn is over", None),
            (ev(e, "first_audio"), "m", "The model starts talking mid-number", ("Model", "“I’m here and listening. If you’re still reading it out, go ahead and finish…”")),
            (ev(e, "client_truncate"), "u", "Caller keeps reading; playback is cut", ("Caller", f"“{s['b']}”")),
            (ev(e, "first_audio", 1), "m", "The actual answer", ("Model", "“Got it, 415-555-0192. If you want, I can read it back again to double-check.”")),
        ]))

    e = ex["hidden-split"]; s = stim[e["meta"]["stimulus"]]
    out.append(dict(name="2-silent-split", e=e,
        title="The turn ended. Nobody heard it.",
        sub="gpt-realtime-2.1 · server_vad, 500 ms · recorded trial",
        intro=("Caller", f"“{s['a']} … {s['b']}”"),
        events=[
            (ev(e, "user_speech_stopped"), "s", "0.9 s pause: the server ends the turn and starts a reply", None),
            (ev(e, "user_speech_started", 1), "u", "Caller resumes. The reply is cancelled before any audio", ("Caller", f"“{s['b']}”")),
            (ev(e, "user_speech_stopped", 1), "s", "The real end of the turn. The session now holds two turns", None),
            (ev(e, "first_audio"), "m", "The reply you hear", ("Model", "“Got it, four people at 7 tonight. Which restaurant is this for?”")),
        ]))

    e = ex["patient-wait"]; s = stim[e["meta"]["stimulus"]]
    gap = e["analysis"]["final_gap_ms"]
    out.append(dict(name="3-nine-seconds-for-canberra", e=e,
        title=f"{round(gap / 1000)} seconds for Canberra",
        sub="gpt-realtime-2.1 · semantic_vad, eagerness low · recorded trial",
        intro=("Caller", f"“{s['a']} … {s['b']}”"),
        events=[
            (e["marks"]["B.end"], "u", "The question is finished. Silence.", None),
            (ev(e, "user_speech_stopped"), "s", f"{secs(ev(e, 'user_speech_stopped') - e['marks']['B.end'])} later, semantic VAD decides the caller is done", None),
            (ev(e, "first_audio"), "m", f"First audio, {secs(gap)} after the question ended", ("Model", "“The capital of Australia is Canberra.”")),
        ]))

    e = ex["yeah-stops-it"]
    out.append(dict(name="4-yeah-stops-it", e=e,
        title="One “yeah” costs four seconds",
        sub="gpt-realtime-2.1 · semantic_vad, eagerness auto · recorded trial",
        intro=("Caller asked", "“Can you walk me through fixing my wifi, step by step?”"),
        events=[
            (e["marks"]["model.onset"], "m", "The model starts its walkthrough", ("Model", "“Alright, let’s walk through a few quick fixes…”")),
            (e["marks"]["X.start"], "u", "Caller says “yeah”, meaning keep going", ("Caller", "“Yeah.”")),
            (ev(e, "client_truncate"), "s", "Playback stops 0.12 s later", None),
            (ev(e, "user_speech_stopped", 1), "s", f"Semantic VAD waits {secs(ev(e, 'user_speech_stopped', 1) - e['marks']['X.end'])} to see if “yeah” was the start of something", None),
            (ev(e, "first_audio", 1), "m", f"The model resumes after {secs(e['analysis']['reply_gap_ms'])} of silence", ("Model", "“First, bring your router and modem into a simple known-good state…”")),
        ]))
    return out


def envelope_path(arr, lane, w0, w1):
    mid, half = (lane[0] + lane[1]) / 2, (lane[1] - lane[0]) / 2
    x = lambda t: PX0 + (t - w0) / (w1 - w0) * (PX1 - PX0)
    i0, i1 = max(0, w0 // 20), min(len(arr) - 1, -(-w1 // 20))
    top = [f"{x(i * 20):.1f},{mid - max(0.02, arr[i]) * half:.1f}" for i in range(i0, i1 + 1)]
    bot = [f"{x(i * 20):.1f},{mid + max(0.02, arr[i]) * half:.1f}" for i in range(i1, i0 - 1, -1)]
    return "M" + "L".join(top + bot) + "Z"


def card(spec, k):
    e = spec["e"]; w0, w1 = e["window"]
    x = lambda t: PX0 + (t - w0) / (w1 - w0) * (PX1 - PX0)
    colors = {"u": USER, "m": MODEL, "s": SERVER}
    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="position:absolute;inset:0">']
    for label, lane in (("CALLER", LANE_U), ("MODEL", LANE_M)):
        mid = (lane[0] + lane[1]) / 2
        svg.append(f'<line x1="{PX0}" x2="{PX1}" y1="{mid}" y2="{mid}" stroke="{RULE}"/>')
        svg.append(f'<text x="40" y="{mid + 4}" font-family="Menlo" font-size="12" letter-spacing="1" fill="{INK3}">{label}</text>')
    svg.append(f'<path d="{envelope_path(e["user"], LANE_U, w0, w1)}" fill="{USER}" opacity="0.9"/>')
    svg.append(f'<path d="{envelope_path(e["model"], LANE_M, w0, w1)}" fill="{MODEL}" opacity="0.9"/>')
    svg.append(f'<line x1="{PX0}" x2="{PX1}" y1="{AXIS_Y}" y2="{AXIS_Y}" stroke="#C4C3BC"/>')
    for t in range(w0, w1 + 1, 1000):
        svg.append(f'<text x="{x(t):.1f}" y="{AXIS_Y + 18}" text-anchor="middle" font-family="Menlo" font-size="12" fill="{INK3}">{(t - w0) // 1000} s</text>')
    prev = -1e9
    for i, (t, who, _, _) in enumerate(spec["events"]):
        dx = max(x(t), prev + 28)  # keep badges from overlapping when events are close together
        prev = dx
        if i >= k:
            continue
        c = colors[who]
        svg.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{LANE_U[0]}" y2="{LANE_M[1]}" stroke="{INK3}" stroke-width="1" opacity="0.6"/>')
        if dx != x(t):
            svg.append(f'<line x1="{x(t):.1f}" x2="{dx:.1f}" y1="312" y2="312" stroke="{INK3}" stroke-width="1"/>')
        svg.append(f'<circle cx="{dx:.1f}" cy="312" r="12" fill="{c}" stroke="{PAPER}" stroke-width="3"/>')
        svg.append(f'<text x="{dx:.1f}" y="316.5" text-anchor="middle" font-family="Menlo" font-size="12" font-weight="700" fill="#fff">{i + 1}</text>')
    svg.append("</svg>")

    if k == 0:
        num, color, caption = "", INK3, "Listen"
        quote = spec["intro"]
    else:
        t, who, caption, quote = spec["events"][k - 1]
        num, color = str(k), colors[who]
    badge = (f'<span style="display:inline-flex;width:40px;height:40px;border-radius:50%;background:{color};color:#fff;'
             f'font:700 20px Menlo;align-items:center;justify-content:center;flex:none">{num}</span>') if num else ""
    q = ""
    if quote:
        q = (f'<div style="margin-top:14px;font:17px/1.45 Georgia,serif;color:{INK2}">'
             f'<span style="font:12px Menlo;letter-spacing:1px;color:{INK3};text-transform:uppercase;margin-right:10px">{html.escape(quote[0])}</span>'
             f'{html.escape(quote[1])}</div>')
    return f"""<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;width:{W}px;height:{H}px;background:{PAPER};position:relative;overflow:hidden;font-family:-apple-system,Helvetica,Arial,sans-serif">
<div style="position:absolute;left:40px;top:34px;right:40px">
  <div style="font:600 40px/1.1 Georgia,serif;color:{INK};letter-spacing:-0.5px">{html.escape(spec['title'])}</div>
  <div style="margin-top:10px;font:14px Menlo;color:{INK3}">{html.escape(spec['sub'])}</div>
</div>
{''.join(svg)}
<div style="position:absolute;left:40px;right:40px;top:520px;border-top:1px solid {RULE};padding-top:22px">
  <div style="display:flex;gap:16px;align-items:center;font:600 28px/1.25 -apple-system,Helvetica,sans-serif;color:{INK}">{badge}<span>{html.escape(caption)}</span></div>
  {q}
</div>
<div style="position:absolute;left:40px;right:40px;bottom:22px;display:flex;justify-content:space-between;font:13px Menlo;color:{INK3}">
  <span>sahasrarjn.com/blog/realtime-turn-detection</span><span>audio: real API session, synthetic caller voice</span>
</div>
</body></html>"""


def render(html_text, png, tmp):
    page = tmp / (png.stem + ".html")
    page.write_text(html_text)
    # headless Chrome sometimes writes the screenshot and then never exits: wait for the file, then stop it
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp / 'chrome'}",
                             f"--force-device-scale-factor={SCALE}", f"--window-size={W},{H}", f"--screenshot={png}", page.as_uri()],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline, last = time.monotonic() + 60, -1
    while time.monotonic() < deadline:
        size = png.stat().st_size if png.exists() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(0.5)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if not png.exists() or png.stat().st_size == 0:
        raise RuntimeError(f"screenshot failed: {png}")


def main():
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in specs(data):
        e = spec["e"]; w0, w1 = e["window"]
        span = (w1 - w0) / 1000
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            times = [0.0] + [(t - w0) / 1000 for t, *_ in spec["events"]] + [span]
            lines = []
            for k in range(len(spec["events"]) + 1):
                png = tmp / f"s{k}.png"
                render(card(spec, k), png, tmp)
                lines += [f"file '{png}'", f"duration {max(0.04, times[k + 1] - times[k]):.3f}"]
            lines.append(f"file '{tmp / f's{len(spec['events'])}.png'}'")
            (tmp / "list.txt").write_text("\n".join(lines) + "\n")
            s = SCALE
            # playhead: a thin bar overlaid with x driven by the frame timestamp (drawbox has no time variable)
            bar_h = round((LANE_M[1] - LANE_U[0]) * s + 20)
            out = OUT / f"{spec['name']}.mp4"
            graph = (f"[0:v]fps=30,format=yuv420p[bg];"
                     f"[bg][2:v]overlay=x='{PX0 * s}+t/{span}*{(PX1 - PX0) * s}':y={round(LANE_U[0] * s - 10)}:eval=frame:shortest=0[v]")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                            "-i", str(POST / "audio" / f"{e['name']}.mp3"),
                            "-f", "lavfi", "-i", f"color=c=0x1B1D21:s=4x{bar_h}:r=30",
                            "-filter_complex", graph, "-map", "[v]", "-map", "1:a",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-t", f"{span:.3f}",
                            "-movflags", "+faststart", str(out)], check=True)
            print(f"wrote {out} ({span:.1f} s)")


if __name__ == "__main__":
    main()
