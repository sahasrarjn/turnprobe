"""Render post 2's paired clips as captioned 1080p videos for an X thread.

One video per pair: the same card shows both models' waveforms, gpt-realtime plays first, then
GPT-Live, with a playhead over whichever is playing and the events as captions. Audio is the two
clips from the post, joined with a short gap.

Usage: uv run python scripts/make_x_videos2.py <blog post dir> <out dir> [pair ...]
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
ONLY = sys.argv[3:]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H, SCALE = 1280, 720, 1.5  # CSS px, rendered at 1920x1080
PX0, PX1 = 96, 1224
BLOCKS = [  # one per system: header y, caller lane, model lane, axis y (mark labels sit between them)
    dict(head=152, u=(204, 254), m=(266, 316), axis=332),
    dict(head=368, u=(420, 470), m=(482, 532), axis=548),
]
GAP_S = 0.6

INK, INK2, INK3, PAPER, RULE = "#1B1D21", "#4A4E58", "#747984", "#FBFAF7", "#E3E0D8"
USER, MODEL, LIVE, RT = "#0087a0", "#B8740F", "#c2417f", "#4750b8"

TITLES = {
    "yeah": ("A “yeah” restarts one model and not the other",
             "“My wifi keeps dropping. Can you walk me through fixing it?” The caller says “yeah” 2.6 s in."),
    "stop": ("“Wait, stop”: 0.14 s vs 1.5 s",
             "Same walkthrough, same caller voice. The caller cuts in 2.6 s after the model starts."),
    "egg": ("One waits 8 seconds, the other answers early",
            "“How long should I boil an egg if I want the yolk … still a little runny?” with a 2.5 s pause."),
}
SYS = {"live": "GPT-Live-1", "server": "gpt-realtime-2.1 · server_vad",
       "sem_low": "gpt-realtime-2.1 · semantic_vad, low", "sem_auto": "gpt-realtime-2.1 · semantic_vad, auto",
       "sem_high": "gpt-realtime-2.1 · semantic_vad, high"}


def load():
    s = (POST / "data.js").read_text()
    return json.loads(s[s.index("{"): s.rindex("}") + 1])


def envelope_path(arr, lane, span):
    mid, half = (lane[0] + lane[1]) / 2, (lane[1] - lane[0]) / 2
    x = lambda t: PX0 + t / span * (PX1 - PX0)
    top = [f"{x(i * 20):.1f},{mid - max(0.02, arr[i]) * half:.1f}" for i in range(len(arr))]
    bot = [f"{x(i * 20):.1f},{mid + max(0.02, arr[i]) * half:.1f}" for i in range(len(arr) - 1, -1, -1)]
    return "M" + "L".join(top + bot) + "Z"


def block_svg(clip, blk, span, active, shown):
    """One system's two lanes, its marks up to `shown`, dimmed when it isn't the one playing."""
    x = lambda t: PX0 + t / span * (PX1 - PX0)
    op = "1" if active else "0.35"
    accent = LIVE if clip["system"] == "live" else RT
    out = [f'<g opacity="{op}">']
    out.append(f'<text x="40" y="{blk["head"]}" font-family="Menlo" font-size="15" font-weight="700" fill="{accent}">'
               f'{html.escape(SYS[clip["system"]])}</text>')
    out.append(f'<text x="{PX1}" y="{blk["head"]}" text-anchor="end" font-family="-apple-system,Helvetica,sans-serif" '
               f'font-size="17" fill="{INK2}">{html.escape(clip["result"])}</text>')
    for label, lane, color, env in (("CALLER", blk["u"], USER, clip["user"]), ("MODEL", blk["m"], MODEL, clip["model"])):
        mid = (lane[0] + lane[1]) / 2
        out.append(f'<line x1="{PX0}" x2="{PX1}" y1="{mid}" y2="{mid}" stroke="{RULE}"/>')
        out.append(f'<text x="40" y="{mid + 4}" font-family="Menlo" font-size="11" letter-spacing="1" fill="{INK3}">{label}</text>')
        out.append(f'<path d="{envelope_path(env, lane, span)}" fill="{color}" opacity="0.9"/>')
    out.append(f'<line x1="{PX0}" x2="{PX1}" y1="{blk["axis"]}" y2="{blk["axis"]}" stroke="#C4C3BC"/>')
    for t in range(0, int(span) + 1, 1000):
        out.append(f'<text x="{x(t):.1f}" y="{blk["axis"] + 16}" text-anchor="middle" font-family="Menlo" '
                   f'font-size="11" fill="{INK3}">{t // 1000} s</text>')
    ends = [-1e9, -1e9]  # two label rows, so labels on close marks don't overlap
    for t, who, label in clip["marks"][:shown]:
        c = USER if who == "u" else MODEL
        lx, width = x(t) + 6, len(label) * 7.6 + 10
        row = 0 if lx > ends[0] else (1 if lx > ends[1] else 0)
        ends[row] = lx + width
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{blk["u"][0] - (10 if row else 28)}" y2="{blk["m"][1]}" stroke="{c}" '
                   f'stroke-width="1.5" stroke-dasharray="3 3" opacity="0.8"/>')
        out.append(f'<text x="{lx:.1f}" y="{blk["u"][0] - (14 if row else 32)}" font-family="-apple-system,Helvetica,sans-serif" '
                   f'font-size="14" fill="{c}">{html.escape(label)}</text>')
    out.append("</g>")
    return "".join(out)


def card(pair, key, span, step):
    """step: (active clip index, marks shown on it, caption)."""
    active, shown, caption = step
    title, sub = TITLES[key]
    svg = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="position:absolute;inset:0">']
    for i, (clip, blk) in enumerate(zip(pair, BLOCKS)):
        svg.append(block_svg(clip, blk, span, i == active, shown if i == active else len(clip["marks"])))
    svg.append("</svg>")
    accent = LIVE if pair[active]["system"] == "live" else RT
    now = SYS[pair[active]["system"]]
    return f"""<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;width:{W}px;height:{H}px;background:{PAPER};position:relative;overflow:hidden;font-family:-apple-system,Helvetica,Arial,sans-serif">
<div style="position:absolute;left:40px;top:34px;right:40px">
  <div style="font:600 40px/1.1 Georgia,serif;color:{INK};letter-spacing:-0.5px">{html.escape(title)}</div>
  <div style="margin-top:10px;font:15px/1.4 Georgia,serif;color:{INK3}">{html.escape(sub)}</div>
</div>
<div style="position:absolute;left:40px;right:40px;top:592px;border-top:1px solid {RULE};padding-top:20px">
  <div style="display:flex;gap:14px;align-items:baseline;font:600 26px/1.3 -apple-system,Helvetica,sans-serif;color:{INK}">
    <span style="font:700 14px Menlo;color:{accent};letter-spacing:1px">{html.escape(now)}</span>
    <span>{html.escape(caption or "")}</span>
  </div>
</div>
<div style="position:absolute;left:40px;right:40px;bottom:20px;display:flex;justify-content:space-between;font:13px Menlo;color:{INK3}">
  <span>sahasrarjn.com/blog/gpt-live-vs-realtime</span><span>audio: real API sessions, synthetic caller voice</span>
</div>
{''.join(svg)}
</body></html>"""


def render(html_text, png, tmp):
    page = tmp / (png.stem + ".html")
    page.write_text(html_text)
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


def steps_for(pair, span_a):
    """(global start time, card step) for every state change across both clips."""
    out = [(0.0, (0, 0, "playing"))]
    for i, (t, _, label) in enumerate(pair[0]["marks"]):
        out.append((t / 1000, (0, i + 1, label)))
    start_b = span_a + GAP_S
    out.append((start_b, (1, 0, "same question, same caller voice")))
    for i, (t, _, label) in enumerate(pair[1]["marks"]):
        out.append((start_b + t / 1000, (1, i + 1, label)))
    return out


def main():
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for n, (key, pair) in enumerate(data["pairs"].items(), start=1):
        if ONLY and key not in ONLY:
            continue
        spans = [(c["window"][1] - c["window"][0]) / 1000 for c in pair]
        span_ms = max(spans) * 1000
        total = spans[0] + GAP_S + spans[1]
        steps = steps_for(pair, spans[0])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            lines = []
            for k, (t, step) in enumerate(steps):
                png = tmp / f"s{k}.png"
                render(card(pair, key, span_ms, step), png, tmp)
                end = steps[k + 1][0] if k + 1 < len(steps) else total
                lines += [f"file '{png}'", f"duration {max(0.04, end - t):.3f}"]
            lines.append(f"file '{tmp / f's{len(steps) - 1}.png'}'")
            (tmp / "list.txt").write_text("\n".join(lines) + "\n")

            audio = tmp / "audio.m4a"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-i", str(POST / "audio" / f"{pair[0]['name']}.mp3"),
                            "-i", str(POST / "audio" / f"{pair[1]['name']}.mp3"),
                            "-filter_complex", f"[0:a]apad=pad_dur={GAP_S}[a0];[a0][1:a]concat=n=2:v=0:a=1[a]",
                            "-map", "[a]", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(audio)], check=True)

            s = SCALE
            heads = []
            for i, (blk, span) in enumerate(zip(BLOCKS, spans)):
                t0 = 0.0 if i == 0 else spans[0] + GAP_S
                x = f"{PX0 * s}+(t-{t0:.3f})/{span:.3f}*{(PX1 - PX0) * s * (span * 1000 / span_ms):.1f}"
                heads.append((x, round(blk["u"][0] * s - 8), round((blk["m"][1] - blk["u"][0]) * s + 16),
                              f"between(t,{t0:.3f},{t0 + span:.3f})"))
            graph = "[0:v]fps=30,format=yuv420p[v0];"
            for i, (x, y, _h, enable) in enumerate(heads):
                graph += f"[v{i}][{i + 2}:v]overlay=x='{x}':y={y}:eval=frame:enable='{enable}':shortest=0[v{i + 1}];"
            graph = graph.rstrip(";").replace(f"[v{len(heads)}];", f"[v{len(heads)}]")
            out = OUT / f"{n}-{key}.mp4"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                   "-i", str(audio)]
            for _x, _y, h, _e in heads:
                cmd += ["-f", "lavfi", "-i", f"color=c=0x1B1D21:s=4x{h}:r=30"]
            cmd += ["-filter_complex", graph, "-map", f"[v{len(heads)}]", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)]
            subprocess.run(cmd, check=True)
            print(f"wrote {out} ({total:.1f} s)")


if __name__ == "__main__":
    main()
