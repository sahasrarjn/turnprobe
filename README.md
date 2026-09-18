# turnprobe

Controlled experiments on how realtime voice models decide when you're done talking: when they cut you off, when they yield, and how long they make you wait. Measured where a listener would hear it.

Accompanies the post **[Measuring turn detection in OpenAI's Realtime API](https://www.sahasrarjn.com/blog/realtime-turn-detection/)**.

This repo contains the harness, every experiment config, and per-trial results for 893 trials against `gpt-realtime-2.1` and `gpt-live-1` (September 2026). The stereo recording of every trial is attached to the [data release](https://github.com/sahasrarjn/turnprobe/releases/tag/data-2026-09-13).

## What it measures

Most voice-agent latency numbers come from server timestamps or byte arrival times. Neither is what a person hears. Realtime servers stream audio faster than it plays, and whether the model "interrupted" you depends on whether its audio actually played before you finished.

turnprobe runs each trial as a fresh session and:

1. **Streams user audio in real time on one clock.** 20 ms frames, each sent when a microphone would have delivered it, so server-side silence timers see realistic timing.
2. **Places model audio on an emulated playout timeline.** A chunk plays at `max(arrival, end of queued audio)`. On interruption the client flushes playback and sends `conversation.item.truncate` with the milliseconds actually played, as OpenAI recommends.
3. **Records everything.** A stereo WAV (left: what the model heard, right: model audio at the playout head), every server event with its arrival time, and the exact session config.
4. **Calibrates itself.** A mock realtime system with known timing (`turnprobe calibrate`) measures harness error: +10 ms constant offset, zero spread (the offline speech detector's resolution).

## Experiments

| Command | Question |
|---|---|
| `turnprobe pause-sweep` | How long can you pause mid-sentence before the system takes the turn? Speaks the first half of a sentence, holds an exact silence, speaks the rest. Halves are cut from one continuous TTS take (so the first half sounds unfinished) and verified by transcription. |
| `turnprobe overlap` | Does a backchannel ("mm-hm") stop the model like a real interruption ("wait, stop")? The user's clip is anchored 2.5 s after the model's audio becomes audible. |
| `turnprobe fragments` | Control: does semantic VAD judge the whole sentence, or only what came after the last pause? Same sentence whole, second half alone, and split with a pause. |
| `turnprobe hard-cases` | Situations where listening while talking can go wrong: the model's own audio echoing back from a speakerphone, the caller turning away to talk to someone else, a mid-sentence self-correction ("Tuesday, no, sorry, Thursday"), and a filled pause ("um... hang on"). A few trials per case, to find out whether a failure happens at all. |

## Results snapshot (gpt-realtime-2.1)

Pause sweep: 12 sentences (6 unfinished clauses, 3 dictations, 3 complete-sounding first halves) × 2 voices × 5 pauses (0.3–2.5 s). n ≈ 112–119 per setting.

| Setting | Turn ended during a pause ≥ 0.6 s | Heard the model before finishing | …on unfinished clauses | Reply gap after finishing (median / p90) |
|---|---|---|---|---|
| `server_vad`, 500 ms | 88/89 | 39/112 | 17/56 | 1.38 s / 1.82 s |
| `semantic_vad`, high | 50/94 | 20/116 | 2/56 | 1.79 s / 3.29 s |
| `semantic_vad`, auto | 31/92 | 13/116 | 2/57 | 1.82 s / 4.97 s |
| `semantic_vad`, low | 32/95 | 15/119 | 1/59 | 1.98 s / 8.69 s |

- Server VAD's turn decision lands ~590 ms after speech ends. Below ~1.2 s pauses the reply is usually cancelled before any audio plays, so the split is silent.
- Semantic VAD's extra wait before deciding piles up at ceilings: p99 of 2.5 s (high), 3.95 s (auto), 7.95 s (low).
- Fragments control: "Can you tell me what the capital of Australia is?" gets a reply in 1.5–1.6 s whole, but 9.0–9.2 s when "Australia is?" follows a 1.5 s pause, the same as "Australia is?" alone (semantic low).
- Overlap: the model stopped in 56/56 trials (backchannels included), 85–189 ms after the user started speaking. After a lone "yeah", "okay" or "right, right", semantic VAD (auto) typically waited 4–5 s before resuming.

GPT-Live-1 (full duplex, no turn settings) on the same tests: talked before the user finished in 20/72 pause trials (15 of them on complete-sounding first halves), median reply gap 1.23 s (p90 1.47 s), kept talking through 11/12 backchannels, stopped for 8/9 interruptions but took ~1.1–2.1 s to do so.

All numbers come from one client network and synthetic voices, in September 2026. Hosted models change without notice: treat these as a dated measurement.

## Data

```
runs/<run>/run.json              experiment config, TTS voice, stimuli
runs/<run>/summary.jsonl         one row per trial: grid parameters, outcome, metrics, cost
runs/<run>/trials/<id>/trial.json   marks (sample-exact user audio times), analysis, session config
runs/<run>/trials/<id>/events.jsonl every server event, with arrival time on the trial clock (audio bytes omitted)
runs/<run>/trials/<id>/audio.mp3    in the release zip, same layout
```

| Run | Model | Experiment | Trials | Spend |
|---|---|---|---|---|
| `20260913-190315-pause-sweep-openai` | gpt-realtime-2.1 | pause sweep v1 (4 sentences × 3 repeats) | 216 | $3.27 |
| `20260913-190825-overlap-openai` | gpt-realtime-2.1 | overlap, voice coral | 42 | $3.17 |
| `20260913-225604-pause-sweep-openai` | gpt-realtime-2.1 | pause sweep v2 (12 sentences × 2 voices) | 480 | $8.22 |
| `20260913-225604-overlap-openai` | gpt-realtime-2.1 | overlap, voice ash | 14 | $1.13 |
| `20260913-225604-fragments-openai` | gpt-realtime-2.1 | fragments control | 48 | $0.69 |
| `20260913-223544-pause-sweep-openai_live` | gpt-live-1 | pause sweep | 72 | $0.78 |
| `20260913-223544-overlap-openai_live` | gpt-live-1 | overlap | 21 | $0.51 |
| `20260914-195354-pause-sweep-openai_live` | gpt-live-1 | pause sweep (12 sentences × 2 voices) | 119 | $1.35 |
| `20260914-201027-overlap-openai_live` | gpt-live-1 | overlap, voice ash | 14 | $0.28 |
| `20260914-201445-fragments-openai_live` | gpt-live-1 | fragments control | 24 | $0.23 |
| `20260916-155723-hard-cases-openai_live` | gpt-live-1 | hard cases, first pass (4 cases, voice ash) | 13 | $0.22 |
| `20260916-160415-hard-cases-openai` | gpt-realtime-2.1 | hard cases, first pass (server_vad) | 13 | $1.31 |
| `20260916-221637-hard-cases-openai_live` | gpt-live-1 | echo levels + correction variants, 2 voices | 62 | $1.17 |
| `20260916-221657-hard-cases-openai` | gpt-realtime-2.1 | echo levels + correction variants, 2 voices | 62 | $7.24 |
| `calibration/` | mock | harness calibration | – | $0 |

Notes on the data, so nothing surprises you:

- **v1 repeats are not independent.** Identical audio sent to a near-deterministic classifier gives near-identical results; v1's 3 repeats per condition are effectively 1. v2 uses sentence and voice variety instead.
- **v2 had a network outage** partway through. Failed trials were re-run in place (rows marked `"rerun": true`; their `cost_usd` includes both attempts). 11 trials that hit the re-run budget remain `error`. The post's analysis also excludes trials where the audio pacer fell more than 20 ms behind (`send_lateness_ms.p99 > 20`, 6 trials, all just before the outage).
- **GPT-Live cost** is estimated from session length ($0.05/min); the API did not report billed seconds before close.
- **Echo trials on gpt-realtime cost about 4× a normal trial**, because the model restarts its answer over and over. In `20260916-221657-hard-cases-openai` the $7 cap stopped 18 of 62 trials (rows marked `skipped_budget`); the grid is shuffled, so the gaps fall at random.
- **The "time" correction variant is a flawed stimulus.** In "a table for four on Friday at six, no, wait, make that eight", both models mostly read "eight" as the party size rather than the time. Treat that variant as ambiguous rather than as a turn-taking result.
- **Six GPT-Live trials in the first hard-cases pass were re-run** after the sender fell behind real time (network, not the harness: the sender now records where it fell behind in `send_lateness_ms.spikes`).

## Reproduce

```bash
uv sync
cp .env.example .env        # OPENAI_API_KEY
uv run pytest -q
uv run turnprobe calibrate

uv run turnprobe pause-sweep --system openai --model gpt-realtime-2.1 --tts openai --voices coral,ash --stimuli all \
  --setting server_vad:silence_duration_ms=500 --setting semantic_vad:eagerness=auto \
  --setting semantic_vad:eagerness=high --setting semantic_vad:eagerness=low \
  --pauses 300,600,900,1500,2500 --reps 1 --concurrency 4 --budget-usd 8

uv run turnprobe overlap --system openai --model gpt-realtime-2.1 --tts openai --voice ash \
  --setting server_vad:silence_duration_ms=500 --setting semantic_vad:eagerness=auto --reps 1 --concurrency 2

uv run turnprobe fragments --system openai --model gpt-realtime-2.1 \
  --setting semantic_vad:eagerness=low --setting semantic_vad:eagerness=auto --concurrency 2

uv run turnprobe pause-sweep --system openai_live --setting default: --tts openai --voice coral \
  --pauses 300,600,900,1200,1800,2500 --reps 3 --concurrency 3

uv run turnprobe hard-cases --system openai_live --setting default: --voices coral,ash \
  --cases echo,echo_ctrl,correction --echo-levels=-24,-18,-12 --variants day,digit --reps 5 --budget-usd 2

uv run turnprobe report runs/<run>      # static HTML report with charts and per-trial audio
uv run turnprobe reanalyze runs/<run>   # re-score saved trials with current analysis code
uv run turnprobe rerun runs/<run> --dry-run   # find trials affected by a harness fix
```

Every run takes `--budget-usd`: no new trial starts once estimated spend (from per-response usage) reaches it. Run long sweeps under `caffeinate -i` on macOS so the machine can't sleep mid-run.

## Layout

```
src/turnprobe/session.py        real-time pacer, trial clock, reactive fixture scripts
src/turnprobe/playout.py        emulated client speaker: scheduling, flush, played-ms accounting
src/turnprobe/adapters/         openai_realtime (Realtime API), openai_live (GPT-Live), mock (calibration)
src/turnprobe/experiments/      pause_sweep, overlap, fragments, hard_cases, runner, reanalyze, rerun
src/turnprobe/labels.py         offline speech segmentation of the model channel
src/turnprobe/tts.py            stimulus synthesis with continuous-take splitting and transcript verification
scripts/build_post1_data.py     builds post 1's chart data and audio clips from the runs
scripts/build_post2_data.py     the same for post 2 (paired clips: same prompt on both models)
scripts/make_x_videos.py        renders post 1's traces as captioned videos for an X thread
scripts/make_x_videos2.py       the same for post 2's paired clips (one model after the other)
```

## License

Code: MIT (`LICENSE`). Data (`runs/` and the release audio): CC BY 4.0 (`DATA_LICENSE.md`). Model audio and transcripts were generated by OpenAI models; user audio was synthesized with OpenAI TTS.
