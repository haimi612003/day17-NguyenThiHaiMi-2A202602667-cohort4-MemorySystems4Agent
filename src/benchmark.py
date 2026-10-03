from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    # Prompt tokens of each conversation turn, in order (for the growth trace).
    turn_prompt_tokens: list[int] = field(default_factory=list)


COLUMNS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)

UNKNOWN_MARKERS = ("không có thông tin", "chưa có trong", "không biết")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = [data]
    for conversation in data:
        if not {"id", "user_id", "turns", "recall_questions"} <= conversation.keys():
            raise ValueError(f"Malformed conversation in {path}: {conversation.get('id')}")
    return data


def _hits(answer: str, expected: list[str]) -> int:
    normalized = _norm(answer)
    return sum(1 for item in expected if _norm(item) in normalized)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1].

    - 0.6 coverage: share of expected facts present
    - 0.2 concision: full marks up to 60 tokens, linear decay to 0 at 200
    - 0.2 structure/honesty: bulleted answer, or an explicit "không có thông tin"
      (admitting a gap beats rambling); minus 0.1 if it mixes known and unknown
    """

    if not answer.strip():
        return 0.0
    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    tokens = estimate_tokens(answer)
    concision = 1.0 if tokens <= 60 else max(0.0, 1 - (tokens - 60) / 140)
    lowered = _norm(answer)
    admits_gap = any(marker in lowered for marker in UNKNOWN_MARKERS)
    structured = answer.lstrip().startswith("- ")
    structure = 1.0 if structured or admits_gap else 0.5
    score = 0.6 * coverage + 0.2 * concision + 0.2 * structure
    if admits_gap and coverage > 0:
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 3)


def llm_judge_quality(judge_model, question: str, answer: str, expected: list[str]) -> float | None:
    """Live-mode judge: ask a model for a 0-10 score; None if the call fails."""

    prompt = (
        "Chấm câu trả lời của trợ lý từ 0 đến 10 theo: đúng fact mong đợi, ngắn gọn, đúng style.\n"
        f"Câu hỏi: {question}\nFact mong đợi: {', '.join(expected)}\nCâu trả lời: {answer}\n"
        "Chỉ trả về một số nguyên."
    )
    try:
        from live_utils import extract_text

        text = extract_text(judge_model.invoke(prompt))
        match = re.search(r"\d+(?:\.\d+)?", text)
        return round(min(10.0, float(match.group())) / 10, 3) if match else None
    except Exception:
        return None


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    judge_model=None,
    verbose: bool = False,
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. Feed every turn into one thread per conversation.
    2. Track agent tokens and prompt tokens processed (all threads, incl. recall).
    3. Ask each recall question in a FRESH thread (cross-session).
    4. Average recall and quality; record User.md growth and compactions.
    """

    users = sorted({c["user_id"] for c in conversations})
    sizes_before = {u: _memory_size(agent, u) for u in users}
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    turn_prompt_tokens: list[int] = []

    for conversation in conversations:
        user_id, conv_id = conversation["user_id"], conversation["id"]
        thread_id = f"{agent_name}:{conv_id}"
        for turn in conversation["turns"]:
            turn_prompt_tokens.append(agent.reply(user_id, thread_id, turn)["prompt_tokens"])
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for index, item in enumerate(conversation["recall_questions"]):
            recall_thread = f"{agent_name}:{conv_id}:recall-{index}"
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            judged = llm_judge_quality(judge_model, item["question"], answer, expected) if judge_model else None
            quality_scores.append(judged if judged is not None else heuristic_quality(answer, expected))
            agent_tokens += agent.token_usage(recall_thread)
            prompt_tokens += agent.prompt_token_usage(recall_thread)
            if verbose:
                print(f"[{agent_name}] {conv_id} Q{index}: {item['question']}\n{answer}\n")

    growth = sum(_memory_size(agent, u) - sizes_before[u] for u in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=compactions,
        turn_prompt_tokens=turn_prompt_tokens,
    )


def _memory_size(agent, user_id: str) -> int:
    return agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        (
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        )
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        return "\n".join(lines)


def _delta_line(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def pct(new: int, old: int) -> str:
        return f"{(new - old) / old * 100:+.1f}%" if old else "n/a"

    total_base = baseline.prompt_tokens_processed + baseline.agent_tokens_only
    total_adv = advanced.prompt_tokens_processed + advanced.agent_tokens_only
    return (
        f"Advanced vs Baseline: prompt tokens {pct(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"agent tokens {pct(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"total usage (prompt + agent) {pct(total_adv, total_base)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.2f}"
    )


def format_prompt_trace(baseline: BenchmarkRow, advanced: BenchmarkRow, every: int = 2) -> str:
    """Per-turn prompt size: baseline grows linearly, advanced plateaus after compaction."""

    headers = ("Turn", "Baseline prompt", "Advanced prompt", "Saved")
    table = []
    for index, (base, adv) in enumerate(zip(baseline.turn_prompt_tokens, advanced.turn_prompt_tokens), start=1):
        if index % every == 0 or index == len(baseline.turn_prompt_tokens):
            table.append((index, base, adv, f"{(base - adv) / base * 100:+.0f}%" if base else "n/a"))
    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        return "\n".join(" | ".join(str(c) for c in row) for row in [headers, *table])


def run_suite(
    title: str,
    dataset: Path,
    config,
    live: bool,
    judge_model=None,
    verbose: bool = False,
    show_trace: bool = False,
) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    # Start every run from an empty User.md so results are reproducible.
    store = UserProfileStore(config.state_dir / "profiles")
    for user_id in {c["user_id"] for c in conversations}:
        store.reset(user_id)

    baseline = BaselineAgent(config, force_offline=not live)
    advanced = AdvancedAgent(config, force_offline=not live)
    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config, judge_model, verbose),
        run_agent_benchmark("Advanced", advanced, conversations, config, judge_model, verbose),
    ]
    turns = sum(len(c["turns"]) for c in conversations)
    print(f"\n## {title}")
    print(f"Dataset: {dataset.name} · {len(conversations)} conversation(s) · {turns} turns · mode={advanced.mode}\n")
    print(format_rows(rows))
    print("\n" + _delta_line(rows[0], rows[1]))
    if show_trace:
        print("\nPrompt tokens per turn (same thread):\n")
        print(format_prompt_trace(rows[0], rows[1]))
    return rows


def main() -> None:
    """Run the Standard benchmark and the Long-Context Stress benchmark."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark")
    parser.add_argument("--live", action="store_true", help="use the configured LLM instead of offline mode")
    parser.add_argument("--verbose", action="store_true", help="print every recall answer")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    judge_model = None
    if args.live and config.judge_model.is_configured():
        from model_provider import build_chat_model

        judge_model = build_chat_model(config.judge_model)

    print(
        f"Compact threshold = {config.compact_threshold_tokens} tokens · keep {config.compact_keep_messages} messages · "
        f"confidence threshold = {config.profile_confidence_threshold}"
    )
    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, args.live, judge_model, args.verbose)
    run_suite(
        "Long-Context Stress Benchmark",
        config.data_dir / "advanced_long_context.json",
        config,
        args.live,
        judge_model,
        args.verbose,
        show_trace=True,
    )


if __name__ == "__main__":
    main()
