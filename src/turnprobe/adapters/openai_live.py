"""OpenAI GPT-Live (full-duplex) over WebSocket.

Docs: https://developers.openai.com/api/docs/guides/voice-websockets?api=live
      https://developers.openai.com/api/docs/guides/live-conversations

Unlike the Realtime API there are no turn-detection settings and no speech started/stopped
events: the model decides when to listen and speak. So the client never flushes playback;
everything is measured from the audio the server actually sends.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

from websockets.asyncio.client import connect

from .base import Adapter

DEFAULT_INSTRUCTIONS = (
    "You are a friendly voice assistant on a phone line. Keep replies to one or two short "
    "sentences. Answer directly; do not delegate."
)


class OpenAILive(Adapter):
    name = "openai_live"
    DEFAULT_MODEL = "gpt-live-1"
    URL = "wss://api.openai.com/v1/live/sessions"
    PRICE_PER_MINUTE = 0.05
    flush_on_user_speech = False

    def __init__(self, model: str = DEFAULT_MODEL, voice: str | None = None,
                 instructions: str = DEFAULT_INSTRUCTIONS):
        super().__init__()
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.session: dict | None = None
        self.usage_seconds: float | None = None
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._started = asyncio.Event()
        self._closed = asyncio.Event()
        self._start_error: str | None = None

    async def connect(self) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._ws = await connect(self.URL, additional_headers={"Authorization": f"Bearer {key}"}, max_size=None)
        self._reader = asyncio.create_task(self._read())
        audio: dict = {"format": {"type": "audio/pcm", "rate": self.sample_rate}}
        if self.voice:
            audio["output"] = {"voice": self.voice}
        session: dict = {"model": self.model, "instructions": self.instructions, "audio": audio}
        # No backend: delegation must be "client" or "responses"; omitting it gives "client", and
        # the instructions tell the model to answer directly. Any delegation events are logged.
        await self._send({"type": "session.start", "event_id": "start", "session": session})
        done, _ = await asyncio.wait(
            [asyncio.create_task(self._started.wait()), asyncio.create_task(self._closed.wait())],
            timeout=15, return_when=asyncio.FIRST_COMPLETED,
        )
        if not self._started.is_set():
            raise RuntimeError(f"GPT-Live session did not start: {self._start_error or 'timeout'}")

    async def _send(self, msg: dict) -> None:
        await self._ws.send(json.dumps(msg))

    async def send_audio(self, pcm: bytes) -> None:
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t == "session.output_audio.delta":
                    self.emit("audio", item_id="live", response_id="live", pcm=base64.b64decode(ev["delta"]))
                elif t == "session.output_transcript.delta":
                    self.emit("transcript", response_id="live", delta=ev.get("delta", ""),
                              start_ms=ev.get("start_ms"), end_ms=ev.get("end_ms"))
                elif t == "session.input_transcript.delta":
                    self.emit("user_transcript", delta=ev.get("delta", ""),
                              start_ms=ev.get("start_ms"), end_ms=ev.get("end_ms"))
                elif t == "error":
                    msg = (ev.get("error") or {}).get("message") or ev.get("message")
                    if not self._started.is_set():
                        self._start_error = msg or json.dumps(ev)[:300]
                    self.emit("error", message=msg, raw=ev)
                else:
                    if t == "session.started":
                        self.session = ev.get("session")
                        self._started.set()
                    elif t in ("session.usage.updated", "session.closed"):
                        usage = ev.get("usage") or ev
                        secs = usage.get("seconds")
                        if isinstance(secs, (int, float)):
                            self.usage_seconds = max(self.usage_seconds or 0.0, float(secs))
                        if t == "session.closed":
                            self._closed.set()
                    self.emit("native", name=t, raw=ev)
        except Exception as exc:
            self.emit("error", message=f"reader: {type(exc).__name__}: {exc}", raw={})
        finally:
            self._closed.set()
            self._q.put_nowait(None)

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._send({"type": "session.close"})
                await asyncio.wait_for(self._closed.wait(), timeout=4)
            except Exception:
                pass
            await self._ws.close()
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader, timeout=3)
            except Exception:
                self._reader.cancel()

    def describe(self) -> dict:
        return {"system": self.name, "model": self.model, "voice": self.voice,
                "usage_seconds": self.usage_seconds, "session": self.session}
