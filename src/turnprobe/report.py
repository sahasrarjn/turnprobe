"""Static HTML report for a run directory (no external assets)."""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

STYLE = """<style>
:root{--bg:#F4F5F2;--fg:#1A201E;--muted:#66706B;--rule:#D9DDD5;--surface:#FBFBF8;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root{--bg:#121615;--fg:#E3E8E4;--muted:#9AA49F;--rule:#2B3431;--surface:#181D1C}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:32px;max-width:1100px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px}
.muted{color:var(--muted)} .mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px}
svg{width:100%;height:auto} .grid{stroke:var(--rule)} .tick{font:11px ui-monospace,Menlo,monospace;fill:var(--muted)}
.axis-title{font-size:12px;fill:var(--muted)} figcaption{font-size:12px;color:var(--muted);margin-top:4px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px} .legend i{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle}
.wrap{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:500;color:var(--muted);border-bottom:1px solid var(--rule);padding:6px 10px 6px 0;white-space:nowrap}
td{border-bottom:1px solid var(--rule);padding:6px 10px 6px 0;vertical-align:middle} .num{font-variant-numeric:tabular-nums;text-align:right}
.chip{font:11px ui-monospace,Menlo,monospace;padding:1px 6px;border-radius:3px;border:1px solid var(--rule)}
.chip.clean{color:#2E7648;border-color:#2E7648} .chip.cut_in,.chip.talk_over{color:#B0302A;border-color:#B0302A}
.chip.no_response,.chip.error{color:#A8680F;border-color:#A8680F} audio{height:28px;width:220px}
</style>"""

PALETTE = ["#276870", "#A8680F", "#7A4E9E", "#B0302A", "#3F7A3A", "#4B5563"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _pct(vals: list[float], q: float) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.percentile(vals, q)) if vals else None


def line_chart(series: dict[str, list[tuple[float, float, float, float]]], x_label: str, y_label: str) -> str:
    W, H, L, R, T, B = 640, 300, 56, 16, 16, 44
    xs = sorted({p[0] for pts in series.values() for p in pts})
    if not xs:
        return "<p>No data.</p>"
    x_min, x_max = min(xs), max(xs)
    span = (x_max - x_min) or 1
    X = lambda v: L + (v - x_min) / span * (W - L - R)
    Y = lambda v: T + (1 - v) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(y_label)}">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{Y(g) + 4:.1f}" text-anchor="end" class="tick">{int(g * 100)}%</text>')
    for v in xs:
        out.append(f'<text x="{X(v):.1f}" y="{H - B + 18}" text-anchor="middle" class="tick">{v:g}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" class="axis-title">{html.escape(x_label)}</text>')
    for i, (name, pts) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        pts = sorted(pts)
        band = " ".join(f"{X(x):.1f},{Y(hi):.1f}" for x, _, _, hi in pts) + " " + " ".join(
            f"{X(x):.1f},{Y(lo):.1f}" for x, _, lo, _ in reversed(pts)
        )
        out.append(f'<polygon points="{band}" fill="{color}" opacity="0.12"/>')
        out.append(
            f'<polyline points="{" ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y, _, _ in pts)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y, _, _ in pts:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{color}"/>')
    out.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{PALETTE[i % len(PALETTE)]}"></i>{html.escape(n)}</span>' for i, n in enumerate(series)
    )
    return f'<figure>{"".join(out)}<div class="legend">{legend}</div><figcaption>{html.escape(y_label)} · shaded: 95% Wilson interval</figcaption></figure>'


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("experiment") == "overlap":
        return build_overlap_report(run_dir, run)
    rows = [json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r["outcome"] not in ("error", "skipped_budget")]
    cost = sum(r.get("cost_usd", 0.0) for r in rows)

    def rate_series(pred) -> dict:
        grouped: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for r in ok:
            grouped[r["setting"]][r["pause_ms"]].append(pred(r))
        series = {}
        for setting, by_pause in grouped.items():
            pts = []
            for p, vals in by_pause.items():
                k, n = sum(vals), len(vals)
                lo, hi = wilson(k, n)
                pts.append((p, k / n, lo, hi))
            series[setting] = pts
        return series

    audible = rate_series(lambda r: r["outcome"] in ("cut_in", "talk_over"))
    split = rate_series(lambda r: bool(r.get("split_in_pause")))

    settings = sorted({r["setting"] for r in rows})
    lat_rows = []
    for s in settings:
        rs = [r for r in ok if r["setting"] == s]
        errs = sum(1 for r in rows if r["setting"] == s and r["outcome"] == "error")
        lat_rows.append(
            "<tr>" + "".join(
                f"<td>{c}</td>" for c in [
                    f"<b>{html.escape(s)}</b>", len(rs), errs,
                    _fmt(_pct([r.get("final_gap_ms") for r in rs], 50)),
                    _fmt(_pct([r.get("final_gap_ms") for r in rs], 90)),
                    _fmt(_pct([r.get("endpoint_lag_ms") for r in rs], 50)),
                    _fmt(_pct([r.get("generation_lag_ms") for r in rs], 50)),
                    _fmt(_pct([r.get("talk_over_ms") for r in rs if r["outcome"] in ("cut_in", "talk_over")], 50)),
                    _fmt(_pct([r.get("send_lateness_ms", {}).get("p99") for r in rs], 50), 1),
                ]
            ) + "</tr>"
        )

    by_stim = defaultdict(lambda: defaultdict(list))
    for r in ok:
        by_stim[r["stimulus"]][r["setting"]].append(r["outcome"] in ("cut_in", "talk_over"))
    stim_rows = "".join(
        f"<tr><td><b>{html.escape(sid)}</b> <span class='muted'>{html.escape(run['stimuli'][sid]['kind'])}</span></td>"
        + "".join(
            f"<td>{_rate(by_stim[sid][s])}</td>" for s in settings
        ) + "</tr>"
        for sid in sorted(by_stim)
    )

    trial_rows = "".join(
        f"<tr><td class='mono'>{r['trial_id']}</td><td>{html.escape(r['setting'])}</td><td>{r['stimulus']}</td>"
        f"<td class='num'>{r['pause_ms']}</td><td><span class='chip {r['outcome']}'>{r['outcome']}</span></td>"
        f"<td class='num'>{_fmt(r.get('final_gap_ms'))}</td><td class='num'>{r.get('responses', '')}</td>"
        f"<td>{'<audio controls preload=none src=\"trials/' + r['trial_id'] + '/audio.wav\"></audio>' if r['outcome'] not in ('error', 'skipped_budget') else html.escape(r.get('error', r['outcome']))}</td></tr>"
        for r in sorted(rows, key=lambda r: (r["setting"], r["stimulus"], r["pause_ms"]))
    )

    doc = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>turnprobe · {html.escape(run_dir.name)}</title>
{STYLE}
<h1>Pause sweep · {html.escape(run["config"]["system"])}</h1>
<p class="muted mono">{html.escape(run_dir.name)} · started {html.escape(run["started"])} · turnprobe {html.escape(run["turnprobe_version"])} · TTS {html.escape(run["tts"]["provider"])}/{html.escape(str(run["tts"]["voice"]))} · {len(rows)} trials ({len(rows) - len(ok)} errors/skipped) · est. cost ${cost:.2f}</p>
<p>Each trial plays the first half of a sentence, a silence of the given length, then the second half. <b>Audible cut-in</b>: the model's audio started before the user finished (during the pause or over the second half). <b>Turn split</b>: the system's own end-of-speech event fell inside the pause, even if no audio was heard. Headphone channels: left = mic, right = model at the playout head.</p>
<div class="charts">
{line_chart(audible, "pause length (ms)", "Audible cut-in rate")}
{line_chart(split, "pause length (ms)", "System ended the turn inside the pause")}
</div>
<h2>Latency (after the full sentence)</h2>
<div class="wrap"><table><thead><tr><th>Setting</th><th>n</th><th>errors</th><th>gap p50 (ms)</th><th>gap p90</th><th>end-of-speech event lag p50</th><th>event → first audio p50</th><th>talk-over p50 (ms, cut-in trials)</th><th>harness send lateness p99</th></tr></thead><tbody>{"".join(lat_rows)}</tbody></table></div>
<p class="muted" style="font-size:13px">Gap = first audible model speech − end of the user's sentence, at the emulated playout head. Event lag = arrival of the system's end-of-speech event − end of the user's sentence (includes its silence timer and network).</p>
<h2>Audible cut-in rate by stimulus</h2>
<div class="wrap"><table><thead><tr><th>Stimulus</th>{"".join(f"<th>{html.escape(s)}</th>" for s in settings)}</tr></thead><tbody>{stim_rows}</tbody></table></div>
<h2>Trials</h2>
<div class="wrap"><table><thead><tr><th>id</th><th>setting</th><th>stimulus</th><th>pause</th><th>outcome</th><th>gap</th><th>responses</th><th>listen</th></tr></thead><tbody>{trial_rows}</tbody></table></div>
"""
    path = run_dir / "report.html"
    path.write_text(doc)
    return path


def _fmt(v, digits: int = 0) -> str:
    if v is None:
        return "–"
    return f"{v:.{digits}f}"


def _rate(vals: list[bool]) -> str:
    if not vals:
        return "–"
    return f"{100 * sum(vals) / len(vals):.0f}% <span class='muted'>({sum(vals)}/{len(vals)})</span>"


def bar_chart(rows: list[tuple[str, list[tuple[str, int, int]]]], title: str) -> str:
    """rows: [(category, [(series, k, n), ...])] -> horizontal grouped bars of k/n."""
    series_names = list(dict.fromkeys(name for _, bars in rows for name, _, _ in bars))
    L, W, bar_h, gap = 150, 640, 12, 14
    H = 24 + sum(len(bars) * (bar_h + 3) + gap for _, bars in rows) + 20
    X = lambda v: L + v * (W - L - 60)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for g in (0, 0.5, 1.0):
        out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="16" y2="{H - 20}" class="grid"/>')
        out.append(f'<text x="{X(g):.1f}" y="12" text-anchor="middle" class="tick">{int(g * 100)}%</text>')
    y = 24
    for cat, bars in rows:
        out.append(f'<text x="{L - 10}" y="{y + (len(bars) * (bar_h + 3)) / 2 + 4:.1f}" text-anchor="end" class="axis-title">{html.escape(cat)}</text>')
        for name, k, n in bars:
            color = PALETTE[series_names.index(name) % len(PALETTE)]
            v = k / n if n else 0
            out.append(f'<rect x="{L}" y="{y}" width="{max(1.5, X(v) - L):.1f}" height="{bar_h}" fill="{color}" rx="2"/>')
            out.append(f'<text x="{X(v) + 6:.1f}" y="{y + bar_h - 2}" class="tick">{k}/{n}</text>')
            y += bar_h + 3
        y += gap
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:{PALETTE[i % len(PALETTE)]}"></i>{html.escape(n)}</span>' for i, n in enumerate(series_names))
    return f'<figure>{"".join(out)}<div class="legend">{legend}</div><figcaption>{html.escape(title)}</figcaption></figure>'


def build_overlap_report(run_dir: Path, run: dict) -> Path:
    rows = [json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines() if l.strip()]
    valid = [r for r in rows if r["outcome"] in ("kept_talking", "stopped_then_replied", "stopped_silent")]
    invalid = len(rows) - len(valid)
    cost = sum(r.get("cost_usd", 0.0) for r in rows)
    settings = sorted({r["setting"] for r in rows})
    clips = run["clips"]

    stop_rows = []
    for cid, c in clips.items():
        bars = []
        for st in settings:
            rs = [r for r in valid if r["clip"] == cid and r["setting"] == st]
            bars.append((st, sum(r["outcome"] != "kept_talking" for r in rs), len(rs)))
        stop_rows.append((f'{c["text"]} ({c["kind"]})', bars))

    summary = []
    for st in settings:
        for kind in ("backchannel", "interrupt"):
            rs = [r for r in valid if r["setting"] == st and r["kind"] == kind]
            stopped = [r for r in rs if r["outcome"] != "kept_talking"]
            summary.append("<tr>" + "".join(f"<td>{c}</td>" for c in [
                f"<b>{html.escape(st)}</b>", kind, len(rs),
                _rate([r["correct"] for r in rs]),
                _rate([r["outcome"] != "kept_talking" for r in rs]),
                _fmt(_pct([r.get("detect_ms") for r in rs], 50)),
                _fmt(_pct([r.get("stop_latency_ms") for r in stopped], 50)),
                _fmt(_pct([r.get("stop_latency_ms") for r in stopped], 90)),
                _fmt(_pct([r.get("talk_over_ms") for r in rs], 50)),
                _fmt(_pct([r.get("reply_gap_ms") for r in stopped], 50)),
            ]) + "</tr>")

    def audio(r):
        if r["outcome"] in ("error", "skipped_budget"):
            return html.escape(r.get("error", r["outcome"]))
        return f'<audio controls preload=none src="trials/{r["trial_id"]}/audio.wav"></audio>'

    chip = lambda r: f"<span class='chip {'clean' if r.get('correct') else 'cut_in'}'>{r['outcome']}</span>" if "correct" in r else f"<span class='chip error'>{r['outcome']}</span>"
    trial_rows = "".join(
        f"<tr><td class='mono'>{r['trial_id']}</td><td>{html.escape(r['setting'])}</td><td>{html.escape(clips[r['clip']]['text'])}</td>"
        f"<td>{chip(r)}</td><td class='num'>{_fmt(r.get('stop_latency_ms'))}</td><td class='num'>{_fmt(r.get('talk_over_ms'))}</td>"
        f"<td style='max-width:340px'>{html.escape((r.get('reply_text') or '')[:160])}</td><td>{audio(r)}</td></tr>"
        for r in sorted(rows, key=lambda r: (r["setting"], r["clip"]))
    )
    doc = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>turnprobe · {html.escape(run_dir.name)}</title>
{STYLE}
<h1>Talking over the model · {html.escape(run["config"]["system"])}</h1>
<p class="muted mono">{html.escape(run_dir.name)} · started {html.escape(run["started"])} · TTS {html.escape(run["tts"]["provider"])}/{html.escape(str(run["tts"]["voice"]))} · {len(rows)} trials ({invalid} not scorable) · est. cost ${cost:.2f}</p>
<p>The model is asked for a long step-by-step answer. {run["config"]["offsets"][0]} ms after its audio becomes audible, the user says one of the clips below. A <b>backchannel</b> should not stop the model; an <b>interruption</b> should. Stop latency = user onset → model audio stops at the playout head. Talk-over = how long the model kept talking over the user's clip.</p>
<p class="muted" style="font-size:13px">Question: “{html.escape(run["question"])}”</p>
<div class="charts">{bar_chart(stop_rows, "Share of trials where the model stopped talking")}</div>
<h2>Summary</h2>
<div class="wrap"><table><thead><tr><th>Setting</th><th>clip type</th><th>n</th><th>correct</th><th>stopped</th><th>speech detected p50 (ms)</th><th>stop latency p50</th><th>stop p90</th><th>talk-over p50</th><th>reply gap p50</th></tr></thead><tbody>{"".join(summary)}</tbody></table></div>
<h2>Trials</h2>
<div class="wrap"><table><thead><tr><th>id</th><th>setting</th><th>user said</th><th>outcome</th><th>stop ms</th><th>talk-over</th><th>model said next</th><th>listen</th></tr></thead><tbody>{trial_rows}</tbody></table></div>
"""
    path = run_dir / "report.html"
    path.write_text(doc)
    return path
