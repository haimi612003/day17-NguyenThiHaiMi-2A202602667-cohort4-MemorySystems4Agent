"""Deterministic response generation shared by both agents in offline mode.

Kept separate from the agents so the *only* difference between baseline and
advanced is where the facts come from (current thread vs. User.md), not how
answers are phrased. That keeps the benchmark comparison fair.
"""

from __future__ import annotations

import re

from memory_store import FIELD_LABELS, LIST_FIELDS, is_question, split_sentences

# question keyword -> profile field(s) it asks for
FIELD_KEYWORDS = (
    ("name", ("tên", "là ai", "biết dũngct")),
    ("profession", ("nghề",)),
    ("location", ("ở đâu", "nơi ở", "còn ở", "đang ở")),
    ("favorite_drink", ("đồ uống", "uống gì")),
    ("favorite_food", ("món ăn",)),
    ("pet", ("nuôi", "thú cưng")),
    ("response_style", ("style", "kiểu trả lời")),
    ("interests", ("quan tâm",)),
)
SUMMARY_KEYWORDS = ("tóm tắt", "là ai", "mô tả ngắn")
RECALL_RE = re.compile(
    r"^(?:nhắc lại|tóm tắt)\b|nhắc lại giúp|bạn có biết|bạn thử nhớ lại|bạn có thể nhắc",
    re.IGNORECASE,
)


def requested_fields(message: str) -> list[str]:
    lowered = message.lower()
    fields = [name for name, keys in FIELD_KEYWORDS if any(k in lowered for k in keys)]
    if any(k in lowered for k in SUMMARY_KEYWORDS):
        for extra in ("name", "profession", "interests"):
            if extra not in fields:
                fields.append(extra)
    return fields


def is_recall_request(message: str) -> bool:
    """A question / request to recall facts, as opposed to the user sharing facts."""

    sentences = split_sentences(message)
    asks = any(is_question(s) or RECALL_RE.search(s) for s in sentences)
    return asks and bool(requested_fields(message))


def _bullet_limit(style: str) -> int | None:
    match = re.search(r"(\d+) bullet", style or "")
    return int(match.group(1)) if match else None


def compose_answer(message: str, facts: dict[str, str], source: str) -> str:
    """Answer a recall request from `facts`; say plainly what is unknown."""

    lines, missing = [], []
    for field_name in requested_fields(message):
        value = facts.get(field_name)
        if value:
            if field_name in LIST_FIELDS and field_name == "interests":
                value = ", ".join(value.split(", ")[:3])
            lines.append(f"{FIELD_LABELS[field_name]}: {value}")
        else:
            missing.append(FIELD_LABELS[field_name].lower())

    if not lines:
        return f"Mình không có thông tin về {', '.join(missing)} trong {source}."

    if missing:
        lines.append(f"Chưa có trong {source}: {', '.join(missing)}")

    # Respect the user's own style preference, e.g. "3 bullet".
    limit = _bullet_limit(facts.get("response_style", ""))
    if limit and len(lines) > limit:
        lines = lines[: limit - 1] + ["; ".join(lines[limit - 1 :])]
    return "\n".join(f"- {line}" for line in lines)
