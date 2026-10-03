"""Ablation study: turn off one bonus guardrail at a time and re-run the Advanced agent.

Shows what each bonus buys in recall / memory size, using the same datasets
as `benchmark.py`. Run from the repo root:  python src/ablation.py
"""

from __future__ import annotations

import dataclasses
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import memory_store
from agent_advanced import AdvancedAgent
from benchmark import load_conversations, run_agent_benchmark
from config import load_config
from memory_store import FactRecord, UserProfileStore


def _keep_first_value(self, profile, fact, report):
    """No conflict handling: the first value ever written is never replaced."""

    if fact.field in profile.facts or fact.confidence < self.confidence_threshold:
        return
    profile.facts[fact.field] = FactRecord(fact.value, fact.confidence, 1, profile.turn)
    report.written[fact.field] = fact.value


# Small synthetic probe for guardrails the shipped datasets never trigger:
# a weak drink mention (confidence threshold) and many one-off interests (decay / cap).
PROBE = [
    {
        "id": "probe-01",
        "user_id": "probe_lan",
        "turns": [
            "Mình tên là Lan, đồ uống yêu thích là trà sữa.",
            "Trưa nay trời nóng nên mình uống nước dừa.",
            "Mình thích Python.",
            "Mình quan tâm RAG.",
            "Mình quan tâm evaluation.",
            "Mình quan tâm LangChain.",
            "Mình quan tâm LangGraph.",
            "Mình quan tâm machine learning.",
            "Mình quan tâm deep learning.",
            "Mình quan tâm LLM.",
            "Mình quan tâm MLOps.",
            "Mình thích Python.",
        ],
        "recall_questions": [
            {"question": "Đồ uống yêu thích của mình là gì?", "expected_contains": ["trà sữa"]},
            {"question": "Mình tên gì và mối quan tâm kỹ thuật chính là gì?", "expected_contains": ["Lan", "Python"]},
        ],
    }
]

VARIANTS = {
    "Full (all guardrails)": {},
    "No conflict handling": {"patches": [("memory_store.UserProfileStore._upsert_single", _keep_first_value)]},
    "No noise/negation filter": {
        "patches": [("memory_store.NEGATION_CUES", ()), ("memory_store.SENTENCE_SKIP_CUES", ())]
    },
    "No question filter": {"patches": [("memory_store.is_question", lambda _sentence: False)]},
    "No confidence threshold": {"config": {"profile_confidence_threshold": 0.0}},
    "No decay / no list cap": {"config": {"memory_decay_rate": 1.0, "max_list_items": 1000}},
}


def run_variant(name: str, spec: dict, datasets: dict[str, list]) -> dict[str, object]:
    base = load_config(Path(__file__).resolve().parent.parent)
    results: dict[str, object] = {"Variant": name}
    with ExitStack() as stack:
        for target, value in spec.get("patches", []):
            stack.enter_context(mock.patch(target, value))
        for label, conversations in datasets.items():
            state_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            config = dataclasses.replace(base, state_dir=state_dir, **spec.get("config", {}))
            agent = AdvancedAgent(config, force_offline=True)
            row = run_agent_benchmark("Advanced", agent, conversations, config)
            results[f"{label} recall"] = f"{row.recall_score:.2f}"
            results[f"{label} User.md bytes"] = row.memory_growth_bytes
            user_id = conversations[0]["user_id"]
            facts = agent.profile_store.facts(user_id)
            if label == "Probe":
                results["Probe drink"] = facts.get("favorite_drink", "-")
            else:
                results[f"{label} location/profession"] = f"{facts.get('location', '-')} / {facts.get('profession', '-')}"
    return results


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    datasets = {
        "Standard": load_conversations(root / "data" / "conversations.json"),
        "Stress": load_conversations(root / "data" / "advanced_long_context.json"),
        "Probe": PROBE,
    }
    rows = [run_variant(name, spec, datasets) for name, spec in VARIANTS.items()]
    headers = list(rows[0].keys())
    try:
        from tabulate import tabulate

        print(tabulate([[r[h] for h in headers] for r in rows], headers=headers, tablefmt="github"))
    except ImportError:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
