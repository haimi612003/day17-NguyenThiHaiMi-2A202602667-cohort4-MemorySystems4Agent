"""Small helpers for live (LangChain) mode. Only imported when a live agent runs."""

from __future__ import annotations

from typing import Any


def extract_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def usage_from_messages(messages: list[Any]) -> tuple[int, int]:
    """Sum provider-reported (input_tokens, output_tokens) across AI messages."""

    prompt_tokens = output_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        prompt_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    return prompt_tokens, output_tokens
