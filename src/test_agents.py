from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import (
    CompactMemoryManager,
    FactCandidate,
    UserProfileStore,
    extract_profile_candidates,
    extract_profile_updates,
)
from model_provider import normalize_provider

ROOT = Path(__file__).resolve().parent.parent

LONG_TURN = (
    "Mình kể thêm một đoạn dài về tin tức để làm phình ngữ cảnh: NASA Artemis III, X-59 bay siêu thanh, "
    "WMO cảnh báo El Nino và kế hoạch điện sạch của British Columbia đều cho thấy bài toán vận hành "
    "phải cân bằng giữa hiệu năng, rủi ro và chi phí, không chỉ là tối ưu một con số đơn lẻ. "
)


def make_config(tmp_path: Path, threshold: int = 200, keep: int = 2):
    """Isolated config: state goes into tmp_path, small threshold so compaction happens fast."""

    config = load_config(ROOT)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(
        config,
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
    )


def facts_of(agent: AdvancedAgent, user_id: str) -> dict[str, str]:
    return agent.profile_store.facts(user_id)


# ---------------------------------------------------------------------------
# Core tests required by the rubric
# ---------------------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user = "dũng ct/../x"  # unsafe id must be sanitized into one folder

    assert store.file_size(user) == 0
    assert "Chưa có thông tin" in store.read_text(user)

    path = store.write_text(user, "# User.md\n- location: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert tmp_path / "profiles" in path.parents
    assert ".." not in path.relative_to(tmp_path).parts
    assert store.file_size(user) == len("# User.md\n- location: Đà Nẵng\n".encode("utf-8"))

    assert store.edit_text(user, "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text(user) and "Đà Nẵng" not in store.read_text(user)
    assert store.edit_text(user, "không tồn tại", "x") is False


def test_structured_profile_round_trip(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("u1", "name", "DũngCT")
    store.upsert_fact("u1", "interests", "Python")
    # A brand-new store instance (= new process) must read the same facts back from disk.
    reloaded = UserProfileStore(tmp_path / "profiles")
    assert reloaded.facts("u1") == {"name": "DũngCT", "interests": "Python"}


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=200, keep=2)
    agent = AdvancedAgent(config, force_offline=True)
    for _ in range(6):
        agent.reply("u1", "long-thread", LONG_TURN)

    thread = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") >= 1
    assert len(thread["messages"]) <= config.compact_keep_messages + 1
    assert thread["summary"], "older messages must be folded into a summary"
    # Short threads must NOT compact.
    agent.reply("u1", "short-thread", "Chào bạn.")
    assert agent.compaction_count("short-thread") == 0


def test_compact_summary_stays_bounded() -> None:
    manager = CompactMemoryManager(threshold_tokens=100, keep_messages=2, max_summary_items=3)
    for index in range(40):
        manager.append("t", "user", f"Lượt {index}: {LONG_TURN}")
    summary_lines = str(manager.context("t")["summary"]).splitlines()
    assert len(summary_lines) <= 3
    assert manager.compaction_count("t") > 5


def test_compaction_does_not_thrash(tmp_path: Path) -> None:
    """Regression: a summary close to the threshold used to re-trigger compaction on every append."""

    config = make_config(tmp_path, threshold=300, keep=2)
    agent = AdvancedAgent(config, force_offline=True)
    for _ in range(14):
        agent.reply("u1", "t", LONG_TURN)
    # 28 appends (user + assistant); far fewer compactions than appends.
    assert agent.compaction_count("t") <= 7
    from memory_store import estimate_tokens

    assert estimate_tokens(str(agent.compact_memory.context("t")["summary"])) <= config.compact_threshold_tokens // 3


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "Đồ uống yêu thích là cà phê sữa đá.",
    ]
    for turn in turns:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    base_answer = baseline.reply("dungct", "session-2", question)["response"]
    adv_answer = advanced.reply("dungct", "session-2", question)["response"]

    assert recall_points(adv_answer, ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points(base_answer, ["DũngCT", "cà phê sữa đá"]) == 0.0
    assert baseline.memory_file_size("dungct") == 0

    # A new AdvancedAgent instance (new process) still remembers via User.md.
    fresh = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in fresh.reply("dungct", "session-3", question)["response"]


def test_baseline_remembers_within_same_thread(tmp_path: Path) -> None:
    baseline = BaselineAgent(make_config(tmp_path), force_offline=True)
    baseline.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT.")
    assert "DũngCT" in baseline.reply("dungct", "t1", "Mình tên gì?")["response"]
    assert "DũngCT" not in baseline.reply("dungct", "t2", "Mình tên gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=300, keep=2)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    base_prompts, adv_prompts = [], []
    for _ in range(14):
        base_prompts.append(baseline.reply("u1", "long", LONG_TURN)["prompt_tokens"])
        adv_prompts.append(advanced.reply("u1", "long", LONG_TURN)["prompt_tokens"])

    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    # Baseline keeps growing; advanced plateaus around the threshold.
    assert base_prompts[-1] > base_prompts[4] * 2
    assert max(adv_prompts[4:]) < config.compact_threshold_tokens * 1.6
    assert adv_prompts[-1] < base_prompts[-1] / 2


def test_short_conversation_advanced_can_cost_more(tmp_path: Path) -> None:
    """Honest trade-off: on a short thread, User.md overhead outweighs compaction."""

    config = make_config(tmp_path, threshold=2000)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in ("Mình tên là DũngCT.", "Mình ở Huế.", "Mình thích Python.", "Mình làm MLOps engineer."):
        baseline.reply("u1", "short", turn)
        advanced.reply("u1", "short", turn)
    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


# ---------------------------------------------------------------------------
# Bonus behaviour: conflict handling, confidence threshold, noise, decay
# ---------------------------------------------------------------------------


def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u1", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("u1", "t2", "À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("u1", "t3", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    facts = facts_of(agent, "u1")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    answer = agent.reply("u1", "t4", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "MLOps engineer" in answer and "Huế" in answer
    assert "backend" not in answer and "Đà Nẵng" not in answer
    # Old values are kept only as labelled history.
    text = agent.profile_store.read_text("u1")
    assert "Đà Nẵng → Huế" in text and "backend engineer → MLOps engineer" in text


@pytest.mark.parametrize(
    "noise",
    [
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại.",
        "Nếu sau này mình có nhắc lại Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại nhé.",
        "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé, vì đó là thông tin cũ.",
    ],
    ids=["joke-product-manager", "meeting-in-hanoi", "conditional-danang", "do-not-say-backend"],
)
def test_noise_does_not_overwrite_profile(tmp_path: Path, noise: str) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u1", "t1", "Mình đang ở Huế và đang làm MLOps engineer.")
    agent.reply("u1", "t1", noise)
    facts = facts_of(agent, "u1")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"


def test_questions_never_write_facts(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u1", "t1", "Bạn có thể nhắc lại tên mình không?")
    agent.reply("u1", "t1", "Hiện tại mình đang ở đâu?")
    assert facts_of(agent, "u1") == {}
    assert agent.memory_file_size("u1") == 0


def test_confidence_threshold_blocks_weak_facts(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    # "vẫn uống X" is a habit, not a stated favourite -> below threshold.
    agent.reply("u1", "t1", "Mình vẫn uống trà đá như cũ.")
    assert "favorite_drink" not in facts_of(agent, "u1")
    assert agent.last_update_report.rejected_low_confidence

    # ...but a weak mention can still reinforce a fact we already trust.
    agent.reply("u1", "t1", "Đồ uống yêu thích là cà phê sữa đá.")
    agent.reply("u1", "t1", "Mình vẫn uống cà phê sữa đá nhưng đang cố giảm.")
    profile = agent.profile_store.load_profile("u1")
    assert profile.facts["favorite_drink"].value == "cà phê sữa đá"
    assert profile.facts["favorite_drink"].mentions == 2


def test_ephemeral_context_is_not_persisted() -> None:
    assert extract_profile_updates("Tuần này mình đang ôn lại async Python.") == {}
    assert "interests" in extract_profile_updates("Mình thích Python và AI ứng dụng.")


def test_memory_decay_caps_lists_and_prefers_recent(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", max_list_items=3, decay_rate=0.5)
    for term in ("RAG", "evaluation", "LangChain", "LangGraph"):
        store.apply_candidates("u1", [FactCandidate("interests", term, 0.8)])
    interests = store.facts("u1")["interests"].split(", ")
    assert len(interests) == 3
    assert "RAG" not in interests  # oldest, never repeated -> decayed out
    assert interests[0] == "LangGraph"  # most recent ranks first


def test_entity_extraction_on_dataset_traps() -> None:
    fields = {c.field: c.value for c in extract_profile_candidates(
        "Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng."
    )}
    assert fields == {"location": "Đà Nẵng"}
    pet = extract_profile_updates("Mình nuôi một bé corgi tên Bơ.")
    assert pet == {"pet": "corgi tên Bơ"}
    assert extract_profile_updates("Mình tên là DũngCT Stress, hiện ở Huế.")["name"] == "DũngCT Stress"


# ---------------------------------------------------------------------------
# Benchmark + provider wiring
# ---------------------------------------------------------------------------


def test_benchmark_tells_the_expected_story(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep=4)
    standard = load_conversations(ROOT / "data" / "conversations.json")
    stress = load_conversations(ROOT / "data" / "advanced_long_context.json")

    std_base = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), standard, config)
    std_adv = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), standard, config)
    st_base = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), stress, config)
    st_adv = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), stress, config)

    assert std_base.recall_score == 0 and st_base.recall_score == 0
    assert std_adv.recall_score == 1.0 and st_adv.recall_score == 1.0
    assert std_adv.memory_growth_bytes > 0 and std_base.memory_growth_bytes == 0
    # Short threads: no compaction, advanced pays more prompt tokens.
    assert std_adv.compactions == 0
    assert std_adv.prompt_tokens_processed > std_base.prompt_tokens_processed
    # Long thread: compaction kicks in and cuts prompt tokens.
    assert st_adv.compactions >= 2
    assert st_adv.prompt_tokens_processed < st_base.prompt_tokens_processed


def test_normalize_provider_aliases() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider(" OpenRouter ") == "openrouter"
    assert normalize_provider("google") == "gemini"
    with pytest.raises(ValueError):
        normalize_provider("not-a-provider")


def test_live_mode_wiring_with_fake_model(tmp_path: Path, monkeypatch) -> None:
    """Exercise the LangChain path without network by swapping in a fake chat model."""

    pytest.importorskip("langchain")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    import agent_advanced
    import agent_baseline

    class ToolFreeFakeModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    scripts = [
        # baseline: plain replies
        [AIMessage(content=f"live reply {i}") for i in range(10)],
        # advanced: first call a memory tool, then answer
        [
            AIMessage(content="", tool_calls=[{"name": "save_user_fact", "args": {"field": "pet", "value": "corgi tên Bơ"}, "id": "c1"}]),
            AIMessage(content="live reply after tool"),
        ],
    ]

    def fake_builder(_config):
        return ToolFreeFakeModel(messages=iter(scripts.pop(0)))

    monkeypatch.setattr(agent_baseline, "build_chat_model", fake_builder)
    monkeypatch.setattr(agent_advanced, "build_chat_model", fake_builder)
    config = make_config(tmp_path)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, provider="ollama"))

    baseline = BaselineAgent(config)
    advanced = AdvancedAgent(config)
    assert baseline.mode == "live" and advanced.mode == "live"

    assert baseline.reply("u1", "t1", "Chào bạn")["response"] == "live reply 0"
    second = baseline.reply("u1", "t1", "Mình tên là DũngCT.")
    assert second["response"] == "live reply 1"
    assert len(baseline.sessions["t1"].messages) == 4  # checkpointer kept the thread

    result = advanced.reply("u1", "t1", "Mình tên là DũngCT.")
    assert result["mode"] == "live" and result["response"] == "live reply after tool"
    facts = facts_of(advanced, "u1")
    assert facts["name"] == "DũngCT"  # deterministic extractor
    assert facts["pet"] == "corgi tên Bơ"  # written by the LLM through the tool
