"""System adapters. Each maps a vendor protocol onto normalized events."""

from __future__ import annotations

from .base import Adapter, Event


def parse_setting(spec: str) -> tuple[str, dict]:
    """'server_vad:silence_duration_ms=500,threshold=0.5' -> ('server_vad', {...})."""
    kind, _, rest = spec.partition(":")
    params: dict = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        params[k.strip()] = _coerce(v.strip())
    return kind.strip(), params


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def make_adapter(system: str, setting: str, model: str | None = None, instructions: str | None = None) -> Adapter:
    kind, params = parse_setting(setting)
    if system == "mock":
        from .mock import MockBot

        return MockBot(**params)
    if system == "openai":
        from .openai_realtime import OpenAIRealtime

        extra = {"instructions": instructions} if instructions else {}
        return OpenAIRealtime(turn_detection={"type": kind, **params}, model=model or OpenAIRealtime.DEFAULT_MODEL, **extra)
    if system == "openai_live":
        from .openai_live import OpenAILive

        extra = {"instructions": instructions + " Answer directly; do not delegate."} if instructions else {}
        return OpenAILive(model=model or OpenAILive.DEFAULT_MODEL, **params, **extra)
    raise ValueError(f"unknown system: {system}")


__all__ = ["Adapter", "Event", "make_adapter", "parse_setting"]
