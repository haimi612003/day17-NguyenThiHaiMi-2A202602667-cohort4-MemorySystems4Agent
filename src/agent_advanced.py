from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    FIELD_LABELS,
    CompactMemoryManager,
    ProfileUpdateReport,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
)
from model_provider import build_chat_model
from offline_responder import compose_answer, is_recall_request

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt có bộ nhớ dài hạn. Hồ sơ người dùng (User.md) bên dưới là nguồn sự thật "
    "cho các fact ổn định; luôn dùng giá trị mới nhất, bỏ qua giá trị trong mục Corrections. "
    "Tôn trọng style trả lời trong hồ sơ."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory.

    Per turn:
    message -> extract facts (with confidence) -> User.md
            -> compact memory append (auto-compact past threshold)
            -> prompt = system + User.md view + summary + recent messages
            -> response -> token counters
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles",
            confidence_threshold=self.config.profile_confidence_threshold,
            max_list_items=self.config.max_list_items,
            decay_rate=self.config.memory_decay_rate,
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.last_update_report: ProfileUpdateReport | None = None
        self._active_user_id: str | None = None  # read by live tools
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ----- shared memory pipeline ------------------------------------------

    def _remember(self, user_id: str, thread_id: str, message: str) -> tuple[ProfileUpdateReport, int, int]:
        """Steps 1-3: update User.md, append to compact memory; return summarizer cost delta."""

        report = self.profile_store.apply_candidates(user_id, extract_profile_candidates(message), message)
        self.last_update_report = report
        thread = self.compact_memory.context(thread_id)
        before_in, before_out = thread["summarizer_input_tokens"], thread["summary_output_tokens"]
        self.compact_memory.append(thread_id, "user", message)
        return (
            report,
            thread["summarizer_input_tokens"] - before_in,
            thread["summary_output_tokens"] - before_out,
        )

    def _record(self, thread_id: str, response: str, prompt_tokens: int, agent_tokens: int) -> None:
        compaction_input, compaction_output = self._pending_compaction_cost(thread_id)
        self.compact_memory.append(thread_id, "assistant", response)
        compaction_input2, compaction_output2 = self._pending_compaction_cost(thread_id)
        # Compaction is not free: a summarizer reads old messages and writes a summary.
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens + (compaction_input2 - compaction_input)
        )
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + agent_tokens + (compaction_output2 - compaction_output)
        )

    def _pending_compaction_cost(self, thread_id: str) -> tuple[int, int]:
        thread = self.compact_memory.context(thread_id)
        return thread["summarizer_input_tokens"], thread["summary_output_tokens"]

    # ----- offline path -------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        report, compaction_in, compaction_out = self._remember(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self._record(
            thread_id,
            response,
            prompt_tokens + compaction_in,
            estimate_tokens(message) + estimate_tokens(response) + compaction_out,
        )
        return {
            "response": response,
            "mode": "offline",
            "prompt_tokens": prompt_tokens,
            "agent_tokens": self.token_usage(thread_id),
            "profile_updates": report.written,
            "corrections": report.corrections,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system + User.md view + summary + kept messages."""

        thread = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.render_for_prompt(user_id))
            + estimate_tokens(str(thread["summary"]))
            + sum(estimate_tokens(m["content"]) for m in thread["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer from persisted memory (works in a brand-new thread)."""

        if is_recall_request(message):
            return compose_answer(message, self.profile_store.facts(user_id), source="User.md")

        report = self.last_update_report
        if report and report.written:
            saved = "; ".join(f"{FIELD_LABELS.get(k, k)} = {v}" for k, v in report.written.items())
            response = f"Đã lưu vào User.md: {saved}."
            if report.corrections:
                response += " Đã cập nhật fact cũ: " + "; ".join(report.corrections) + "."
            return response
        return "Đã nhận."

    # ----- live path ----------------------------------------------------------

    def _build_live_messages(self, user_id: str, thread_id: str) -> list[dict[str, str]]:
        thread = self.compact_memory.context(thread_id)
        system = ADVANCED_SYSTEM_PROMPT + "\n\n" + self.profile_store.read_text(user_id)
        if thread["summary"]:
            system += "\n\nTóm tắt phần hội thoại cũ đã được nén:\n" + str(thread["summary"])
        return [{"role": "system", "content": system}] + list(thread["messages"])

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        from live_utils import extract_text, usage_from_messages

        report, compaction_in, compaction_out = self._remember(user_id, thread_id, message)
        live_messages = self._build_live_messages(user_id, thread_id)
        self._active_user_id = user_id
        result = self.langchain_agent.invoke({"messages": live_messages})
        messages = result["messages"]
        response = extract_text(messages[-1])
        prompt_tokens, output_tokens = usage_from_messages(messages[len(live_messages) :])
        if prompt_tokens == 0:
            prompt_tokens = sum(estimate_tokens(m["content"]) for m in live_messages)
            output_tokens = estimate_tokens(response)
        self._record(thread_id, response, prompt_tokens + compaction_in, estimate_tokens(message) + output_tokens + compaction_out)
        return {
            "response": response,
            "mode": "live",
            "prompt_tokens": prompt_tokens,
            "agent_tokens": self.token_usage(thread_id),
            "profile_updates": report.written,
            "corrections": report.corrections,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: chat model + tools to read / edit User.md.

        - `build_chat_model(self.config.model)` for the selected provider
        - short-term state + summary come from CompactMemoryManager, so compaction
          is measured identically in offline and live mode
        - the dynamic prompt (User.md + summary) is rebuilt on every turn
        """

        if not self.config.model.is_configured():
            return None
        try:
            from langchain.agents import create_agent
            from langchain_core.tools import tool

            model = build_chat_model(self.config.model)
        except Exception:
            return None

        store = self.profile_store

        @tool
        def read_user_profile() -> str:
            """Read the current user's User.md profile."""

            return store.read_text(self._active_user_id or "anonymous")

        @tool
        def save_user_fact(field: str, value: str) -> str:
            """Save one stable fact about the user. field is one of: name, location, profession,
            favorite_drink, favorite_food, pet, interests, response_style."""

            # Same gate as the extractor: confidence + conflict handling in UserProfileStore.
            report = store.upsert_fact(self._active_user_id or "anonymous", field, value, confidence=0.9)
            return f"saved={report.written} corrections={report.corrections}"

        return create_agent(model, tools=[read_user_profile, save_user_fact])
