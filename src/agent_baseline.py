from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from offline_responder import compose_answer, is_recall_request

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt. Trả lời ngắn gọn dựa trên cuộc trò chuyện hiện tại. "
    "Nếu không biết thông tin thì nói thẳng là không biết."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Remembers what was said in the same thread (full history is re-sent every turn)
    - No `User.md`, no compaction
    - A new thread id starts from zero, so long-term facts are forgotten
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        # Prompt = system prompt + the whole thread so far (grows every turn).
        prompt_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )

        if is_recall_request(message):
            # Short-term memory only: facts come from earlier messages of THIS thread.
            facts: dict[str, str] = {}
            for previous in session.messages[:-1]:
                if previous["role"] == "user":
                    facts.update(extract_profile_updates(previous["content"]))
            response = compose_answer(message, facts, source="cuộc trò chuyện hiện tại")
        else:
            response = "Đã nhận."

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)
        return {
            "response": response,
            "mode": "offline",
            "prompt_tokens": prompt_tokens,
            "agent_tokens": session.token_usage,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        from live_utils import extract_text, usage_from_messages

        session = self.sessions.setdefault(thread_id, SessionState())
        before = len(session.messages)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result["messages"]
        new_messages = messages[before:]
        response = extract_text(messages[-1])
        prompt_tokens, output_tokens = usage_from_messages(new_messages)
        if prompt_tokens == 0:  # provider did not report usage
            prompt_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
                estimate_tokens(extract_text(m)) for m in messages[:-1]
            )
            output_tokens = estimate_tokens(response)

        session.messages = [{"role": getattr(m, "type", "user"), "content": extract_text(m)} for m in messages]
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + output_tokens
        return {
            "response": response,
            "mode": "live",
            "prompt_tokens": prompt_tokens,
            "agent_tokens": session.token_usage,
        }

    def _maybe_build_langchain_agent(self):
        """Build `create_agent` + `InMemorySaver` when deps and credentials exist."""

        if not self.config.model.is_configured():
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
        except Exception:
            return None
        # The checkpointer is keyed by thread_id: that IS the short-term memory.
        return create_agent(model, tools=[], system_prompt=BASELINE_SYSTEM_PROMPT, checkpointer=InMemorySaver())
