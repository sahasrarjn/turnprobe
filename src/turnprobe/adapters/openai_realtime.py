"""OpenAI Realtime API (GA) over WebSocket.

Docs: https://developers.openai.com/api/docs/guides/realtime-vad
      https://developers.openai.com/api/docs/guides/realtime-conversations
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
    "sentences. If the caller seems to be mid-sentence, you may wait for them to finish."
)


class OpenAIRealtime(Adapter):
    name = "openai"
    DEFAULT_MODEL = "gpt-realtime-2.1"
    URL = "wss://api.openai.com/v1/realtime?model={model}"

    def __init__(
        self,
        turn_detection: dict,
        model: str = DEFAULT_MODEL,
        voice: str = "marin",
        instructions: str = DEFAULT_INSTRUCTIONS,
    ):
        super().__init__()
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.turn_detection = turn_detection
        self.flush_on_user_speech = bool(turn_detection.get("interrupt_response", True))
        self.session: dict | None = None
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._updated = asyncio.Event()

    async def connect(self) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; copy .env.example to .env and add it")
        self._ws = await connect(
            self.URL.format(model=self.model),
            additional_headers={"Authorization": f"Bearer {key}"},
            max_size=None,
        )
        self._reader = asyncio.create_task(self._read())
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "output_modalities": ["audio"],
                    "instructions": self.instructions,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": self.sample_rate},
                            "turn_detection": self.turn_detection,
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": self.sample_rate},
                            "voice": self.voice,
                        },
                    },
                },
            }
        )
        await asyncio.wait_for(self._updated.wait(), timeout=15)

    async def _send(self, msg: dict) -> None:
        await self._ws.send(json.dumps(msg))

    async def send_audio(self, pcm: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def truncate(self, item_id: str, audio_end_ms: int) -> None:
        await self._send(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": int(audio_end_ms),
            }
        )

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t == "response.output_audio.delta":
                    self.emit(
                        "audio",
                        item_id=ev.get("item_id", ""),
                        response_id=ev.get("response_id", ""),
                        pcm=base64.b64decode(ev["delta"]),
                    )
                elif t == "input_audio_buffer.speech_started":
                    self.emit("user_speech_started", audio_ms=ev.get("audio_start_ms"), item_id=ev.get("item_id"))
                elif t == "input_audio_buffer.speech_stopped":
                    self.emit("user_speech_stopped", audio_ms=ev.get("audio_end_ms"), item_id=ev.get("item_id"))
                elif t == "response.created":
                    self.emit("response_created", response_id=ev.get("response", {}).get("id"))
                elif t == "response.done":
                    resp = ev.get("response", {})
                    self.emit(
                        "response_done",
                        response_id=resp.get("id"),
                        status=resp.get("status"),
                        status_details=resp.get("status_details"),
                        usage=resp.get("usage"),
                    )
                elif t == "response.output_audio_transcript.delta":
                    self.emit("transcript", response_id=ev.get("response_id"), delta=ev.get("delta", ""))
                elif t == "error":
                    self.emit("error", message=ev.get("error", {}).get("message"), raw=ev)
                else:
                    if t in ("session.created", "session.updated"):
                        self.session = ev.get("session")
                        if t == "session.updated":
                            self._updated.set()
                    self.emit("native", name=t, raw=_strip_audio(ev))
        finally:
            self._q.put_nowait(None)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader, timeout=5)
            except (asyncio.TimeoutError, Exception):
                self._reader.cancel()

    def describe(self) -> dict:
        return {
            "system": self.name,
            "model": self.model,
            "voice": self.voice,
            "turn_detection": self.turn_detection,
            "session": self.session,
        }


def _strip_audio(ev: dict) -> dict:
    return {k: ("<omitted>" if k in ("delta", "audio") else v) for k, v in ev.items()}
