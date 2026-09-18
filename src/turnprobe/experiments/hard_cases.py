"""Hard cases for full-duplex models: situations where "listening while talking" can go wrong.

Small on purpose (a few reps per case): the goal is to find out whether a failure exists and get a
recording of it, not to estimate a rate. Each case has an obvious right behaviour:

  echo        speakerphone without echo cancellation: the model's own playback leaks back into the
              mic (delayed, quieter). Right: keep talking normally.
  echo_ctrl   the same walkthrough with no echo, as a reference for natural pauses.
  side_talk   "Hold on a sec." then, turned away from the phone, "Honey, can you turn the TV down?
              I'm on the phone." then "Sorry about that, what do I do first?" Right: don't answer
              the side remark; answer the last question.
  correction  "Book a table for two on Tuesday at seven, ... no, sorry, Thursday at seven."
              Right: book Thursday, and don't confirm Tuesday before the caller corrects it.
  um_hold     "My account number is ... um ... hang on, let me find it ... it's four seven seven one,
              two two nine." Right: wait through the "um" and read back the full number.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from ..adapters import make_adapter
from ..audio import db_to_gain, write_wav
from ..labels import speech_segments
from ..session import EchoConfig, Trial, TrialResult
from ..tts import make_tts, spoken_tokens, synth_clip, synth_split
from .overlap import INSTRUCTIONS as WALKTHROUGH_INSTRUCTIONS
from .overlap import QUESTION as WALKTHROUGH_QUESTION
from .runner import new_run_dir, run_grid, write_run_meta

SR = 24000

CASES: dict[str, dict] = {
    "echo": {
        "instructions": WALKTHROUGH_INSTRUCTIONS,
        "echo": {"delay_ms": 150.0, "gain_db": -12.0},  # gain_db is overridden per trial by the level grid
    },
    "echo_ctrl": {
        "instructions": WALKTHROUGH_INSTRUCTIONS,
        "reps": 1,
    },
    "side_talk": {
        "instructions": "You are a tech-support voice assistant for home internet. Keep replies to one or two short sentences.",
        "opener": "Hi, I need some help resetting my router.",
        "hold": "Hold on a sec.",
        "aside": "Honey, can you turn the TV down? I'm on the phone.",
        "aside_gain_db": -10.0,
        "back": "Sorry about that. Okay, what do I do first?",
    },
    "correction": {
        "pause_ms": 800,
    },
    "um_hold": {
        "instructions": "You are a customer support voice assistant. When the caller gives an account number, "
                        "read it back digit by digit to confirm. Keep replies short.",
        "a": "My account number is",
        "um": "Um...",
        "hang_on": "Hang on, let me find it.",
        "b": "it's four seven seven one, two two nine.",
        "digits": "four seven seven one two two nine",
    },
}


BOOKING = ("You are the phone booking assistant for Luigi's restaurant. When a caller asks for a table, "
           "confirm the booking by repeating the party size, day and time. Keep replies short.")
CORRECTIONS: dict[str, dict] = {
    "day": {"instructions": BOOKING, "a": "Can you book a table for two on Tuesday at seven,",
            "b": "no, sorry, Thursday at seven.", "right": "thursday", "wrong": "tuesday"},
    "time": {"instructions": BOOKING, "a": "Can you book a table for four on Friday at six,",
             "b": "no, wait, make that eight.", "right": "eight", "wrong": "six"},
    "digit": {"instructions": "You are a customer support voice assistant. When the caller gives a phone number, "
                              "read it back digit by digit to confirm. Keep replies short.",
              "a": "My callback number is five five five, zero one nine two,", "b": "sorry, zero one nine three.",
              "right": "zero one nine three", "wrong": "zero one nine two"},
}


@dataclass
class HardCasesConfig:
    system: str
    settings: list[str]
    cases: list[str] = field(default_factory=lambda: list(CASES))
    reps: int = 3
    tts: str = "openai"
    voice: str | None = "ash"
    voices: list[str] | None = None  # several caller voices; overrides `voice`
    echo_levels_db: list[float] = field(default_factory=lambda: [-12.0])  # echo gain relative to model playback
    variants: list[str] = field(default_factory=lambda: ["day"])  # correction variants
    model: str | None = None
    concurrency: int = 1
    out: str = "runs"
    budget_usd: float | None = None
    seed: int = 23


# ---------------------------------------------------------------------------------------- words on the timeline

def model_words(result: TrialResult, segs: list[tuple[float, float]]) -> list[tuple[float, str]]:
    """Model transcript words with approximate as-heard times.

    GPT-Live transcript deltas carry start_ms on a clock that runs ahead of playback, so they are
    shifted to line up the first word with the first model audio. gpt-realtime transcripts have no
    word timings: each output item's words are spread over that item's audio, and words past a
    client truncation are dropped (never heard). Good enough to say what was said when, not to
    time individual words.
    """
    ev = result.events
    live = [e for e in ev if e["type"] == "transcript" and e.get("start_ms") is not None]
    if live:
        if not segs:
            return []
        shift = segs[0][0] - live[0]["start_ms"]
        return [(e["start_ms"] + shift, e.get("delta", "")) for e in live]
    text = {e["raw"]["item_id"]: e["raw"]["transcript"] for e in ev
            if e["type"] == "native" and e.get("name") == "response.output_audio_transcript.done"}
    audio: dict[str, list[float]] = {}
    for e in ev:
        if e["type"] == "audio" and not e.get("ignored") and e.get("play_ms") is not None:
            a = audio.setdefault(e["item_id"], [e["play_ms"], 0.0])
            a[1] += e["samples"] * 1000.0 / result.sr
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


def said_in(words, segs, pred) -> str:
    """Text of the model speech segments whose onset satisfies `pred` (words go to their nearest segment)."""
    if not segs:
        return ""
    def nearest(t):
        return min(range(len(segs)), key=lambda i: 0 if segs[i][0] <= t <= segs[i][1]
                   else min(abs(t - segs[i][0]), abs(t - segs[i][1])))
    keep = {i for i, s in enumerate(segs) if pred(s[0])}
    return "".join(w for t, w in words if nearest(t) in keep).strip()


# ---------------------------------------------------------------------------------------- analysis

def analyze(result: TrialResult, g: dict) -> dict:
    m, ev = result.marks, result.events
    segs = speech_segments(result.model, result.sr)
    words = model_words(result, segs)
    case = g["case"]
    base = {
        "model_segments": segs,
        "responses": sum(e["type"] == "response_created" for e in ev),
        "truncations": sum(e["type"] == "client_truncate" for e in ev),
        "transcript": "".join(w for _, w in words).strip(),
    }

    if case in ("echo", "echo_ctrl"):
        onset = m.get("model.onset")
        if onset is None:
            return {**base, "outcome": "no_answer"}
        answer = [s for s in segs if s[0] >= onset - 50]
        gaps = [round(b[0] - a[1]) for a, b in zip(answer, answer[1:])]
        detected = sum(e["type"] == "user_speech_started" and e["t_ms"] > onset for e in ev)
        stalls = [x for x in gaps if x >= 1500]
        # A walkthrough that stops mid-sentence and never resumes is a failure too: the trial
        # ends after quiet_ms of silence, so the last words heard are the last words said.
        cut_off = bool(base["transcript"]) and base["transcript"].rstrip()[-1] not in ".!?"
        outcome = ("restart_loop" if base["truncations"] > 2 else
                   "stalled" if stalls or base["truncations"] else
                   "went_quiet_mid_sentence" if cut_off else "clean")
        return {**base, "outcome": outcome, "ends_mid_sentence": cut_off,
                "speech_ms": round(sum(b - a for a, b in answer)), "longest_gap_ms": max(gaps, default=0),
                "stalls": len(stalls), "user_speech_detected": detected,
                "answer_span_ms": round(answer[-1][1] - answer[0][0]) if answer else 0}

    if case == "side_talk":
        t0, t1, back0, back1 = m["T.start"], m["T.end"], m["F.start"], m["F.end"]
        during = said_in(words, segs, lambda o: m["H.end"] - 100 <= o < back0)
        over_aside = [s for s in segs if s[0] < back0 and s[1] > t0]
        reply = said_in(words, segs, lambda o: o >= back1 - 200)
        answered_aside = bool(re.search(r"\b(tv|television|volume|turn(ing)? (it )?down|quiet)\b", during.lower()))
        outcome = ("answered_side_talk" if answered_aside else
                   "spoke_during_side_talk" if over_aside else
                   "clean" if reply else "no_reply")
        return {**base, "outcome": outcome, "said_during_hold": during, "reply": reply,
                "spoke_over_aside_ms": round(sum(max(0.0, min(s[1], t1) - max(s[0], t0)) for s in over_aside))}

    if case == "correction":
        v = CORRECTIONS[g.get("variant", "day")]
        a_end, b_end = m["A.end"], m["B.end"]
        early = said_in(words, segs, lambda o: a_end - 100 <= o < b_end)
        final = said_in(words, segs, lambda o: o >= b_end - 100)
        def has(text, value):
            return f" {value} " in f" {' '.join(spoken_tokens(text))} "
        right, wrong = has(final, v["right"]), has(final, v["wrong"])
        outcome = ("used_wrong_value" if wrong and not right else
                   "no_confirmation" if not right else
                   "said_wrong_value_early" if has(early, v["wrong"]) else
                   "spoke_during_correction" if early else "clean")
        return {**base, "outcome": outcome, "said_early": early, "reply": final,
                "final_right": right, "final_wrong": wrong,
                "ends_mid_sentence": bool(final) and final.rstrip()[-1] not in ".!?",
                "final_gap_ms": next((round(s[0] - b_end) for s in segs if s[0] >= b_end), None)}

    if case == "um_hold":
        hold0, b0, b_end = m["A.end"], m["B.start"], m["B.end"]
        in_hold = said_in(words, segs, lambda o: hold0 <= o < b0)
        over_b = said_in(words, segs, lambda o: b0 <= o < b_end)
        reply = said_in(words, segs, lambda o: o >= b_end - 100)
        want = CASES["um_hold"]["digits"]
        readback_ok = want in " ".join(spoken_tokens(reply))
        outcome = "cut_in_during_hold" if in_hold else "talked_over_number" if over_b else "clean" if reply else "no_reply"
        return {**base, "outcome": outcome, "said_during_hold": in_hold, "said_over_number": over_b,
                "reply": reply, "readback_ok": readback_ok,
                "final_gap_ms": next((round(s[0] - b_end) for s in segs if s[0] >= b_end), None)}

    raise ValueError(f"unknown case {case}")


# ---------------------------------------------------------------------------------------- trials

def turned_away(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Speech aimed away from the phone: quieter and duller."""
    sos = butter(4, 3000, btype="low", fs=SR, output="sos")
    return (sosfilt(sos, x) * db_to_gain(gain_db)).astype(np.float32)


async def run_trial(g: dict, cfg: HardCasesConfig, clips: dict) -> TrialResult:
    spec = CASES[g["case"]]
    variant = g.get("variant", "day")
    instructions = CORRECTIONS[variant]["instructions"] if g["case"] == "correction" else spec["instructions"]
    adapter = make_adapter(cfg.system, g["setting"], cfg.model, instructions=instructions)
    await adapter.connect()
    try:
        echo = None
        if "echo" in spec:
            echo = EchoConfig(delay_ms=spec["echo"]["delay_ms"], gain_db=g.get("level_db", spec["echo"]["gain_db"]))
        trial = Trial(adapter, sr=SR, echo=echo)
        voice = g.get("voice") or cfg.voice
        c = clips[(g["case"], voice, variant if g["case"] == "correction" else None)]

        async def script(ctx):
            ctx.silence(500)
            if g["case"] in ("echo", "echo_ctrl"):
                _, q_end = ctx.play(c["question"], "Q")
                onset = await ctx.wait_model_onset(after_ms=q_end, timeout_ms=15000)
                if onset is None:
                    return
                ctx.mark("model.onset", onset)
                await ctx.wait_settled(after_ms=onset, quiet_ms=2500, give_up_ms=8000, hard_cap_ms=50000)
            elif g["case"] == "side_talk":
                _, o_end = ctx.play(c["opener"], "O")
                await ctx.wait_settled(after_ms=o_end, quiet_ms=1200, give_up_ms=8000, hard_cap_ms=20000)
                ctx.play(c["hold"], "H")
                ctx.silence(900)
                ctx.play(c["aside"], "T")
                ctx.silence(1500)
                _, f_end = ctx.play(c["back"], "F")
                await ctx.wait_settled(after_ms=f_end, quiet_ms=2000, give_up_ms=9000, hard_cap_ms=30000)
            elif g["case"] == "correction":
                ctx.play(c["a"], "A")
                ctx.silence(spec["pause_ms"])
                _, b_end = ctx.play(c["b"], "B")
                await ctx.wait_settled(after_ms=b_end, quiet_ms=2000, give_up_ms=9000)
            elif g["case"] == "um_hold":
                ctx.play(c["a"], "A")
                ctx.silence(400)
                ctx.play(c["um"], "U")
                ctx.silence(900)
                ctx.play(c["hang_on"], "N")
                ctx.silence(1500)
                _, b_end = ctx.play(c["b"], "B")
                await ctx.wait_settled(after_ms=b_end, quiet_ms=2000, give_up_ms=9000)

        return await trial.run(script)
    finally:
        await adapter.close()


async def make_clips(cfg: HardCasesConfig, run_dir: Path) -> dict:
    out: dict = {}
    methods: dict[str, str] = {}
    for voice in cfg.voices or [cfg.voice]:
        await _voice_clips(cfg, run_dir, make_tts(cfg.tts, voice), voice, out, methods)
    out["_split_methods"] = methods
    return out


async def _voice_clips(cfg, run_dir, tts, voice, out, methods) -> None:
    async def clip(text):
        return await asyncio.to_thread(synth_clip, tts, text, SR)

    async def split(a, b):
        """One continuous take cut at the pause (keeps "unfinished" prosody on A); if no take splits
        cleanly, two separate takes. The method is recorded in run.json."""
        try:
            sp = await asyncio.to_thread(synth_split, tts, a, b, SR)
            methods[f"{voice}|{a}"] = "continuous"
            return sp.a, sp.b
        except RuntimeError:
            methods[f"{voice}|{a}"] = "separate"
            return await clip(a), await clip(b)

    for case in cfg.cases:
        s = CASES[case]
        made: dict[str | None, dict] = {}
        if case in ("echo", "echo_ctrl"):
            made[None] = {"question": await clip(WALKTHROUGH_QUESTION)}
        elif case == "side_talk":
            made[None] = {"opener": await clip(s["opener"]), "hold": await clip(s["hold"]),
                          "aside": turned_away(await clip(s["aside"]), s["aside_gain_db"]), "back": await clip(s["back"])}
        elif case == "correction":
            for variant in cfg.variants:
                v = CORRECTIONS[variant]
                a, b = await split(v["a"], v["b"])
                made[variant] = {"a": a, "b": b}
        elif case == "um_hold":
            a, b = await split(s["a"], s["b"])
            made[None] = {"a": a, "b": b, "um": await clip(s["um"]), "hang_on": await clip(s["hang_on"])}
        for variant, parts in made.items():
            out[(case, voice, variant)] = parts
            for name, x in parts.items():
                tag = "-".join(str(p) for p in (case, variant, voice, name) if p)
                write_wav(run_dir / "stimuli" / f"{tag}.wav", x, SR)


async def run_hard_cases(cfg: HardCasesConfig) -> Path:
    run_dir = new_run_dir(cfg.out, "hard-cases", cfg.system)
    (run_dir / "stimuli").mkdir(exist_ok=True)
    clips = await make_clips(cfg, run_dir)
    grid = []
    for st in cfg.settings:
        for case in cfg.cases:
            for voice in cfg.voices or [cfg.voice]:
                extra = ([{"level_db": lv} for lv in cfg.echo_levels_db] if case == "echo" else
                         [{"variant": v} for v in cfg.variants] if case == "correction" else [{}])
                for e in extra:
                    for r in range(CASES[case].get("reps", cfg.reps)):
                        grid.append({"setting": st, "case": case, "voice": voice, **e, "rep": r})
    write_run_meta(run_dir, "hard_cases", config=asdict(cfg), tts={"provider": cfg.tts, "voices": cfg.voices or [cfg.voice]},
                   cases={c: {k: v for k, v in CASES[c].items()} for c in cfg.cases},
                   corrections={v: CORRECTIONS[v] for v in cfg.variants} if "correction" in cfg.cases else {},
                   question=WALKTHROUGH_QUESTION, split_methods=clips["_split_methods"], trials_planned=len(grid))
    await run_grid(
        run_dir, grid,
        run_one=lambda g: run_trial(g, cfg, clips),
        analyze=analyze,
        concurrency=cfg.concurrency, budget_usd=cfg.budget_usd, seed=cfg.seed,
        describe=lambda g: f"{g['setting']:<36} {g['case']:<11} {g.get('voice')} {g.get('level_db', g.get('variant', ''))!s:<6} rep {g['rep']}",
    )
    return run_dir
