from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token, 0 for empty text.

    Not tokenizer-exact, but stable and deterministic, which is what an offline
    benchmark needs. Both agents use the same estimator, so comparisons are fair.
    """

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Entity extraction (bonus): message -> structured fact candidates
# ---------------------------------------------------------------------------

# Single-valued fields: a newer confident value replaces the old one (conflict handling).
SINGLE_FIELDS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")
# Multi-valued fields: items accumulate, ranked by mentions with decay (memory decay).
LIST_FIELDS = ("interests", "response_style")

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm kỹ thuật",
    "response_style": "Style trả lời",
}

CITIES = (
    "Hồ Chí Minh", "TP.HCM", "Sài Gòn", "Hà Nội", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ",
    "Nha Trang", "Đà Lạt", "Quy Nhơn", "Vũng Tàu", "Hội An", "Biên Hòa", "Vinh", "Buôn Ma Thuột",
    "Quảng Ngãi", "Quảng Nam", "Bình Dương", "Hạ Long", "Thái Nguyên", "Phú Quốc",
)
CITY_RE = "(" + "|".join(re.escape(c) for c in sorted(CITIES, key=len, reverse=True)) + ")"

ROLE_RE = (
    r"((?:[A-Za-z][A-Za-z0-9+#.\-]*\s+){0,2}"
    r"(?:engineer|developer|scientist|manager|designer|analyst|researcher|architect|consultant))\b"
)

TECH_TERMS = (
    "AI ứng dụng", "AI agent", "async Python", "Python", "MLOps", "RAG", "evaluation",
    "memory architecture", "LangChain", "LangGraph", "machine learning", "deep learning", "LLM",
)

# Cues that make a clause NOT describe the user's current state.
NEGATION_CUES = (
    "không còn", "không phải", "chứ không", "đừng", "chỉ là", "đùa", "lúc đầu", "trước đó",
    "trước đây", "nếu", "hay là", "họp", "như ví dụ cũ", "thông tin cũ", "nghề cũ",
)
# Sentences that are hypothetical / jokes: skip location & profession entirely.
SENTENCE_SKIP_CUES = ("đùa", "nếu ", "giả sử")
# Words that signal a fresh correction, which raises confidence.
CORRECTION_CUES = ("đính chính", "thực ra", "cập nhật", "chuyển sang", "giờ ", "từ tuần này")
PRESENT_CUES = ("đang", "hiện", "giờ", "vẫn", "giai đoạn này", "hiện tại")
# Ephemeral context: true this week/today, not a stable preference.
EPHEMERAL_CUES = (
    "hôm nay", "tuần này", "tối nay", "chiều nay", "sáng nay", "hôm qua", "tối qua", "tạm thời",
    "cho cuộc benchmark này",
)
QUESTION_RE = re.compile(r"(\?\s*$|\b(gì|ở đâu|là ai|thế nào|bao nhiêu)\b)", re.IGNORECASE)

STYLE_PREF_CUES = ("muốn", "thích", "hãy", "ưu tiên", "style", "nhớ", "giữ")
STYLE_ANSWER_CUES = ("trả lời", "giải thích", "style", "câu trả lời", "trình bày", "bullet")
INTEREST_CUES = ("thích", "quan tâm", "học thêm", "đang học")


@dataclass
class FactCandidate:
    field: str
    value: str
    confidence: float
    reason: str = ""


def split_sentences(message: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", _nfc(message).strip())
    return [p.strip() for p in parts if p.strip()]


def split_clauses(sentence: str) -> list[str]:
    parts = re.split(r"\s*[:;,]\s*|\s+(?:chứ|nhưng|dù)\s+", sentence)
    return [p.strip(" .!") for p in parts if p and p.strip(" .!")]


def is_question(sentence: str) -> bool:
    return bool(QUESTION_RE.search(sentence))


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def _clean_value(value: str) -> str:
    value = re.split(r"\s+(?:như cũ|nhưng|mỗi ngày|và thấy|vì|để)\b", value)[0]
    return value.strip(" .,!;:")


def _capitalized_prefix(text: str, max_words: int = 4) -> str:
    words = []
    for word in text.split()[:max_words]:
        word = word.strip(".,!?;:")
        if not word or not word[0].isupper():
            break
        words.append(word)
    return " ".join(words)


def _extract_name(clause: str) -> FactCandidate | None:
    match = re.search(
        r"(?:(?:^|\b)(?:mình|tôi|em)\s+tên(?:\s+là)?|tên\s+(?:mình|tôi|em)\s+là|^tên)\s+(.+)",
        clause,
        re.IGNORECASE,
    )
    if not match:
        return None
    name = _capitalized_prefix(match.group(1))
    return FactCandidate("name", name, 0.95, "giới thiệu tên") if name else None


def _extract_location(clause: str) -> FactCandidate | None:
    if _has_any(clause, NEGATION_CUES):
        return None
    patterns = (
        (rf"từ\s+{CITY_RE}\s+sang\s+{CITY_RE}", 2, 0.9, "cập nhật từ A sang B"),
        (rf"nơi ở[^,.]*?\b(?:là|sang)\s+{CITY_RE}", 1, 0.9, "nêu rõ nơi ở"),
        (rf"\bở\s+{CITY_RE}", 1, 0.75, "nói đang ở"),
    )
    for pattern, group, confidence, reason in patterns:
        match = re.search(pattern, clause)
        if match:
            if group == 1 and confidence < 0.9:
                if _has_any(clause, PRESENT_CUES):
                    confidence += 0.1
                if _has_any(clause, CORRECTION_CUES):
                    confidence += 0.05
            return FactCandidate("location", match.group(group), round(min(confidence, 0.95), 2), reason)
    return None


def _extract_profession(clause: str) -> FactCandidate | None:
    if _has_any(clause, NEGATION_CUES):
        return None
    match = re.search(rf"(?:làm|là|sang|nghề)\s+{ROLE_RE}", clause, re.IGNORECASE)
    if not match:
        return None
    confidence = 0.85 + (0.05 if _has_any(clause, CORRECTION_CUES + PRESENT_CUES) else 0.0)
    return FactCandidate("profession", match.group(1).strip(), round(confidence, 2), "nêu nghề nghiệp")


def _extract_drink_food_pet(sentence: str) -> list[FactCandidate]:
    found: list[FactCandidate] = []
    match = re.search(r"đồ uống (?:yêu thích|ưa thích|ruột)(?: của mình)?(?: vẫn)? là ([^.,!?;]+)", sentence, re.I)
    if match:
        found.append(FactCandidate("favorite_drink", _clean_value(match.group(1)), 0.95, "nêu đồ uống yêu thích"))
    else:
        match = re.search(r"(?<!đồ )\buống ([^.,!?;]+)", sentence, re.I)
        if match:
            # Habit, not an explicit preference: below threshold, can only reinforce.
            found.append(FactCandidate("favorite_drink", _clean_value(match.group(1)), 0.55, "chỉ nhắc đang uống"))
    match = re.search(r"món (?:ăn )?(?:yêu thích|ưa thích|ruột)(?: của mình)?(?: là)? ([^.,!?;]+)", sentence, re.I)
    if match:
        found.append(FactCandidate("favorite_food", _clean_value(match.group(1)), 0.95, "nêu món ăn yêu thích"))
    match = re.search(r"\bnuôi (?:một |1 )?(?:bé |con )?([^\s.,!?;]+)(?: tên ([^\s.,!?;]+))?", sentence, re.I)
    if match:
        pet = match.group(1) + (f" tên {match.group(2)}" if match.group(2) else "")
        found.append(FactCandidate("pet", pet, 0.9, "nêu thú cưng"))
    return found


def _extract_interests(sentence: str) -> list[FactCandidate]:
    if not _has_any(sentence, INTEREST_CUES) or _has_any(sentence, EPHEMERAL_CUES):
        return []
    remaining = sentence
    found = []
    for term in TECH_TERMS:  # ordered longest/specific first
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", remaining):
            found.append(FactCandidate("interests", term, 0.8, "nêu sở thích kỹ thuật"))
            remaining = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", remaining)
    return found


def _extract_style(sentence: str) -> list[FactCandidate]:
    lowered = sentence.lower()
    if not (_has_any(lowered, STYLE_PREF_CUES) and _has_any(lowered, STYLE_ANSWER_CUES)):
        return []
    if _has_any(lowered, EPHEMERAL_CUES):
        return []
    tags = []
    if "ngắn" in lowered or re.search(r"\bgọn\b", lowered):
        tags.append("ngắn gọn")
    bullets = re.search(r"(\d+)\s+bullet", lowered)
    if bullets:
        tags.append(f"{bullets.group(1)} bullet")
    elif "bullet" in lowered:
        tags.append("dạng bullet")
    if "ví dụ" in lowered:
        tags.append("có ví dụ thực tế")
    if "trade-off" in lowered:
        tags.append("so sánh trade-off")
    if "rõ ý" in lowered:
        tags.append("rõ ý")
    if "cấu trúc" in lowered:
        tags.append("có cấu trúc")
    return [FactCandidate("response_style", tag, 0.8, "nêu style trả lời") for tag in tags]


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Entity extraction with a confidence score per fact.

    Rules of thumb:
    - question sentences never produce facts ("Mình tên gì?")
    - negated / hypothetical / joking clauses are ignored ("chỉ là câu đùa", "không còn ở")
    - ephemeral context ("tuần này", "tạm thời") is not stored as a stable preference
    """

    candidates: list[FactCandidate] = []
    for sentence in split_sentences(message):
        if is_question(sentence):
            continue
        skip_state_facts = _has_any(sentence, SENTENCE_SKIP_CUES)
        for clause in split_clauses(sentence):
            name = _extract_name(clause)
            if name:
                candidates.append(name)
            if skip_state_facts:
                continue
            for extractor in (_extract_location, _extract_profession):
                fact = extractor(clause)
                if fact:
                    candidates.append(fact)
        candidates.extend(_extract_drink_food_pet(sentence))
        candidates.extend(_extract_interests(sentence))
        candidates.extend(_extract_style(sentence))
    return candidates


def extract_profile_updates(message: str, min_confidence: float = 0.7) -> dict[str, str]:
    """Convert raw user text into stable profile facts (only confident ones).

    Single-valued fields keep the last confident value in the message (the
    correction wins); list fields are joined with ", ".
    """

    updates: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    for fact in extract_profile_candidates(message):
        if fact.confidence < min_confidence:
            continue
        if fact.field in LIST_FIELDS:
            items = lists.setdefault(fact.field, [])
            if fact.value not in items:
                items.append(fact.value)
        else:
            updates[fact.field] = fact.value
    for field_name, items in lists.items():
        updates[field_name] = ", ".join(items)
    return updates


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------


@dataclass
class FactRecord:
    value: str
    confidence: float
    mentions: int
    seen: int  # profile turn counter when last confirmed


@dataclass
class UserProfile:
    user_id: str
    turn: int = 0
    facts: dict[str, FactRecord] = field(default_factory=dict)
    lists: dict[str, dict[str, FactRecord]] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)


@dataclass
class ProfileUpdateReport:
    written: dict[str, str] = field(default_factory=dict)
    reinforced: list[str] = field(default_factory=list)
    rejected_low_confidence: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


_FACT_LINE = re.compile(r"^- (\w+): (.+?) _\(conf ([\d.]+) · mentions (\d+) · seen #(\d+)\)_$")
_LIST_ITEM = re.compile(r"^(.+) \(x(\d+), #(\d+)\)$")
MAX_CORRECTIONS = 5


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user).

    Raw text API: path_for / read_text / write_text / edit_text / file_size.
    Structured API: load_profile / save_profile / apply_candidates / facts /
    upsert_fact / render_for_prompt.
    """

    root_dir: Path
    confidence_threshold: float = 0.7
    max_list_items: int = 6
    decay_rate: float = 0.9

    # ----- raw text API -------------------------------------------------

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", unicodedata.normalize("NFKD", user_id).encode("ascii", "ignore").decode())
        slug = slug.strip("_") or "anonymous"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User.md: {user_id}\n\n_Chưa có thông tin._\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def reset(self, user_id: str) -> None:
        path = self.path_for(user_id)
        if path.exists():
            path.unlink()

    # ----- structured API -----------------------------------------------

    def load_profile(self, user_id: str) -> UserProfile:
        profile = UserProfile(user_id=user_id)
        path = self.path_for(user_id)
        if not path.exists():
            return profile
        section = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.rstrip()
            if line.startswith("## "):
                section = line[3:].strip().lower()
                continue
            turn = re.match(r"^_turns observed: (\d+)_$", line)
            if turn:
                profile.turn = int(turn.group(1))
                continue
            if not line.startswith("- "):
                continue
            if section.startswith("facts"):
                match = _FACT_LINE.match(line)
                if match:
                    key, value, conf, mentions, seen = match.groups()
                    profile.facts[key] = FactRecord(value, float(conf), int(mentions), int(seen))
            elif section.startswith("lists"):
                key, _, raw_items = line[2:].partition(": ")
                items: dict[str, FactRecord] = {}
                for raw in raw_items.split(" | "):
                    match = _LIST_ITEM.match(raw.strip())
                    if match:
                        items[match.group(1)] = FactRecord(match.group(1), 0.8, int(match.group(2)), int(match.group(3)))
                profile.lists[key] = items
            elif section.startswith("corrections"):
                profile.corrections.append(line[2:])
        return profile

    def save_profile(self, profile: UserProfile) -> Path:
        lines = [f"# User.md: {profile.user_id}", "", f"_turns observed: {profile.turn}_", "", "## Facts"]
        for key in SINGLE_FIELDS:
            if key in profile.facts:
                rec = profile.facts[key]
                lines.append(
                    f"- {key}: {rec.value} _(conf {rec.confidence:.2f} · mentions {rec.mentions} · seen #{rec.seen})_"
                )
        lines += ["", "## Lists"]
        for key in LIST_FIELDS:
            items = self.ranked_items(profile, key)
            if items:
                lines.append(f"- {key}: " + " | ".join(f"{r.value} (x{r.mentions}, #{r.seen})" for r in items))
        if profile.corrections:
            lines += ["", "## Corrections (giá trị cũ, không dùng làm hiện tại)"]
            lines += [f"- {c}" for c in profile.corrections[-MAX_CORRECTIONS:]]
        return self.write_text(profile.user_id, "\n".join(lines) + "\n")

    def item_score(self, profile: UserProfile, record: FactRecord) -> float:
        """Memory decay: mentions weighted down by how long ago they were last seen."""

        return record.mentions * (self.decay_rate ** max(0, profile.turn - record.seen))

    def ranked_items(self, profile: UserProfile, key: str) -> list[FactRecord]:
        items = profile.lists.get(key, {}).values()
        return sorted(items, key=lambda r: (-self.item_score(profile, r), -r.seen))

    def apply_candidates(self, user_id: str, candidates: list[FactCandidate], message: str = "") -> ProfileUpdateReport:
        """Gate candidates by confidence, resolve conflicts, decay lists, persist."""

        profile = self.load_profile(user_id)
        profile.turn += 1
        report = ProfileUpdateReport()
        lowered_message = _nfc(message).lower()

        for fact in candidates:
            if fact.field in LIST_FIELDS:
                self._upsert_list_item(profile, fact, report)
            else:
                self._upsert_single(profile, fact, report)

        # Reinforce existing facts that the user repeats without an extraction pattern.
        for key, rec in profile.facts.items():
            if rec.seen != profile.turn and rec.value.lower() in lowered_message:
                rec.mentions += 1
                rec.seen = profile.turn
                report.reinforced.append(key)

        # Memory decay: cap list sizes, dropping the weakest items first.
        for key in LIST_FIELDS:
            ranked = self.ranked_items(profile, key)
            for stale in ranked[self.max_list_items:]:
                del profile.lists[key][stale.value]

        changed = bool(report.written or report.reinforced or report.corrections)
        if changed or self.path_for(user_id).exists():
            self.save_profile(profile)
        return report

    def _upsert_single(self, profile: UserProfile, fact: FactCandidate, report: ProfileUpdateReport) -> None:
        current = profile.facts.get(fact.field)
        same = current is not None and current.value.casefold() == fact.value.casefold()
        if same:
            # Even a low-confidence repeat confirms what we already know.
            current.mentions += 1
            current.seen = profile.turn
            current.confidence = max(current.confidence, fact.confidence)
            report.reinforced.append(fact.field)
            return
        if fact.confidence < self.confidence_threshold:
            report.rejected_low_confidence.append(f"{fact.field}={fact.value} ({fact.confidence:.2f})")
            return
        if current is not None:
            # Conflict handling: newest confident value wins, old one goes to history only.
            note = f"{fact.field}: {current.value} → {fact.value} (turn #{profile.turn})"
            profile.corrections.append(note)
            profile.corrections = profile.corrections[-MAX_CORRECTIONS:]
            report.corrections.append(note)
        profile.facts[fact.field] = FactRecord(fact.value, fact.confidence, 1, profile.turn)
        report.written[fact.field] = fact.value

    def _upsert_list_item(self, profile: UserProfile, fact: FactCandidate, report: ProfileUpdateReport) -> None:
        if fact.confidence < self.confidence_threshold:
            report.rejected_low_confidence.append(f"{fact.field}={fact.value} ({fact.confidence:.2f})")
            return
        items = profile.lists.setdefault(fact.field, {})
        if fact.value in items:
            items[fact.value].mentions += 1
            items[fact.value].seen = profile.turn
            report.reinforced.append(fact.field)
        else:
            items[fact.value] = FactRecord(fact.value, fact.confidence, 1, profile.turn)
            report.written[fact.field] = fact.value

    def facts(self, user_id: str) -> dict[str, str]:
        """Flat view: field -> current value (lists joined by ', ' in priority order)."""

        profile = self.load_profile(user_id)
        view = {key: rec.value for key, rec in profile.facts.items()}
        for key in LIST_FIELDS:
            values = [r.value for r in self.ranked_items(profile, key)]
            if any(re.fullmatch(r"\d+ bullet", v) for v in values):
                values = [v for v in values if v != "dạng bullet"]  # "3 bullet" is more specific
            if values:
                view[key] = ", ".join(values)
        return view

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 0.95) -> ProfileUpdateReport:
        return self.apply_candidates(user_id, [FactCandidate(key, value, confidence, "manual")])

    def render_for_prompt(self, user_id: str) -> str:
        """Compact view injected into the prompt: values only, no bookkeeping metadata."""

        view = self.facts(user_id)
        if not view:
            return ""
        lines = [f"- {FIELD_LABELS.get(k, k)}: {v}" for k, v in view.items()]
        return "User profile (User.md):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def _gist(text: str, max_chars: int = 110) -> str:
    """Pick the most information-dense sentence (most named entities / numbers)."""

    sentences = split_sentences(text) or [text]

    def density(sentence: str) -> int:
        tokens = sentence.split()
        return sum(1 for t in tokens if t[:1].isupper() or any(ch.isdigit() for ch in t))

    best = max(sentences, key=density)
    return best if len(best) <= max_chars else best[: max_chars - 1].rstrip() + "…"


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic extractive summary: one gist line per user message, newest kept.

    Assistant replies are dropped: in this lab they mostly acknowledge the user,
    and stable facts already live in User.md. An LLM summarizer can replace this.
    """

    lines = [f"- {_gist(m['content'])}" for m in messages if m.get("role") == "user" and m.get("content")]
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - Keeps the most recent `keep_messages` messages verbatim
    - When messages + summary exceed `threshold_tokens`, older messages are
      folded into a bounded summary
    - Tracks compactions and summarizer cost (input read / summary written)
    """

    threshold_tokens: int
    keep_messages: int
    max_summary_items: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.context(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.context_tokens(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        new_lines = summarize_messages(older, max_items=self.max_summary_items)
        previous = thread["summary"]
        merged = [line for line in (previous.splitlines() + new_lines.splitlines()) if line.strip()]
        merged = merged[-self.max_summary_items :]
        # Anti-thrashing: if the summary alone nears the threshold, every append would
        # re-trigger compaction. Cap it at a third of the budget (oldest lines go first).
        while len(merged) > 1 and estimate_tokens("\n".join(merged)) > self.threshold_tokens // 3:
            merged.pop(0)
        summary = "\n".join(merged)

        # A real summarizer reads the old summary + older messages and writes a new summary.
        thread["summarizer_input_tokens"] += estimate_tokens(previous) + sum(estimate_tokens(m["content"]) for m in older)
        thread["summary_output_tokens"] += estimate_tokens(summary)
        thread["summary"] = summary
        thread["messages"] = recent
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
                "summarizer_input_tokens": 0,
                "summary_output_tokens": 0,
            }
        return self.state[thread_id]

    def context_tokens(self, thread_id: str) -> int:
        thread = self.context(thread_id)
        return estimate_tokens(thread["summary"]) + sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"]) if thread_id in self.state else 0
