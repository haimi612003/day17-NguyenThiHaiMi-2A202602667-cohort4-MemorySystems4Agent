from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


# Default model per provider, used when LLM_MODEL is not set.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "local-model",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Env var holding the API key / base URL for each provider.
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "custom": "CUSTOM_BASE_URL",
    "gemini": None,
    "anthropic": "ANTHROPIC_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs, models."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Bonus knobs (see memory_store.py).
    profile_confidence_threshold: float = 0.7
    max_list_items: int = 6
    memory_decay_rate: float = 0.9


def _env(name: str | None, default: str | None = None) -> str | None:
    if not name:
        return default
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<prefix>PROVIDER`, `<prefix>MODEL`, ... from the environment."""

    raw_provider = _env(f"{prefix}PROVIDER", fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    same_as_fallback = fallback is not None and provider == fallback.provider
    model_name = _env(
        f"{prefix}MODEL",
        fallback.model_name if same_as_fallback else DEFAULT_MODELS[provider],
    )
    temperature = float(_env(f"{prefix}TEMPERATURE", "0") or 0)
    api_key = _env(API_KEY_ENV[provider]) or _env("GOOGLE_API_KEY" if provider == "gemini" else None)
    base_url = _env(BASE_URL_ENV[provider])
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def _load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # Minimal fallback parser: KEY=VALUE lines, no interpolation.
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a populated LabConfig.

    Env vars:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE: main agent model
    - JUDGE_PROVIDER / JUDGE_MODEL: optional judge model (defaults to the main model)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
      CUSTOM_BASE_URL / CUSTOM_API_KEY, OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS / COMPACT_KEEP_MESSAGES: compact memory knobs
    - PROFILE_CONFIDENCE_THRESHOLD: minimum confidence before writing to User.md
    - LAB_STATE_DIR: override where `state/` lives
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = Path(_env("LAB_STATE_DIR", str(root / "state"))).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM_")
    judge_model = _provider_config("JUDGE_", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(_env("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(_env("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=float(_env("PROFILE_CONFIDENCE_THRESHOLD", "0.7")),
        max_list_items=int(_env("PROFILE_MAX_LIST_ITEMS", "6")),
        memory_decay_rate=float(_env("PROFILE_DECAY_RATE", "0.9")),
    )
