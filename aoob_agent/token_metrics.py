"""Token estimation helpers for per-investigation efficiency metrics."""

from __future__ import annotations

from typing import Any, Optional


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token) when provider usage is unavailable."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_token_estimate(msg: Any) -> int:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    else:
        text = str(content or "")
    usage = getattr(msg, "usage_metadata", None) or getattr(msg, "response_metadata", {}).get(
        "token_usage"
    )
    if isinstance(usage, dict):
        total = usage.get("total_tokens") or usage.get("total")
        if isinstance(total, int) and total > 0:
            return total
        inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        if isinstance(inp, int) and isinstance(out, int) and inp + out > 0:
            return inp + out
    return estimate_tokens(text)


def messages_token_total(messages: list) -> int:
    return sum(message_token_estimate(m) for m in messages)


def naive_full_file_tokens(source_lines: list[str]) -> int:
    """Upper-bound tokens if the agent loaded the entire translation unit."""
    if not source_lines:
        return 1
    body = "\n".join(source_lines[1:] if len(source_lines) > 1 else source_lines)
    return max(1, estimate_tokens(body))


def compute_token_savings_percent(
    tokens_used: int, naive_tokens: int
) -> float:
    if naive_tokens <= 0:
        return 0.0
    savings = 1.0 - (tokens_used / naive_tokens)
    return round(max(0.0, min(100.0, savings * 100.0)), 4)
