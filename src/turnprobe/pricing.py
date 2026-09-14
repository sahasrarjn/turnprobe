"""Cost estimates from reported usage. USD per 1M tokens, from
https://developers.openai.com/api/docs/pricing (checked 2026-09-13)."""

from __future__ import annotations

OPENAI_REALTIME = {
    "gpt-realtime-2.1": dict(text_in=4.00, cached_in=0.40, text_out=24.00, audio_in=32.00, cached_audio_in=0.40, audio_out=64.00),
    "gpt-realtime-2.1-mini": dict(text_in=0.60, cached_in=0.06, text_out=2.40, audio_in=10.00, cached_audio_in=0.30, audio_out=20.00),
    "gpt-realtime-2": dict(text_in=4.00, cached_in=0.40, text_out=24.00, audio_in=32.00, cached_audio_in=0.40, audio_out=64.00),
    "gpt-realtime": dict(text_in=4.00, cached_in=0.40, text_out=16.00, audio_in=32.00, cached_audio_in=0.40, audio_out=64.00),
    "gpt-realtime-mini": dict(text_in=0.60, cached_in=0.06, text_out=2.40, audio_in=10.00, cached_audio_in=0.30, audio_out=20.00),
}


def openai_usage_cost(model: str, usage: dict | None) -> float:
    if not usage:
        return 0.0
    p = OPENAI_REALTIME.get(model, OPENAI_REALTIME["gpt-realtime-2.1"])
    inp, out = usage.get("input_token_details") or {}, usage.get("output_token_details") or {}
    cached = inp.get("cached_tokens_details") or {}
    c_text, c_audio = cached.get("text_tokens", 0), cached.get("audio_tokens", 0)
    return (
        (inp.get("text_tokens", 0) - c_text) * p["text_in"]
        + c_text * p["cached_in"]
        + (inp.get("audio_tokens", 0) - c_audio) * p["audio_in"]
        + c_audio * p["cached_audio_in"]
        + out.get("text_tokens", 0) * p["text_out"]
        + out.get("audio_tokens", 0) * p["audio_out"]
    ) / 1e6
