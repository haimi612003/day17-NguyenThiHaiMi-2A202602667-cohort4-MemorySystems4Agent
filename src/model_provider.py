from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

# Common typos / alternative names people put in `.env`.
PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "vllm": "custom",
    "lmstudio": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}

# Providers that can run without an API key (local servers).
KEYLESS_PROVIDERS = {"ollama"}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_configured(self) -> bool:
        """True when this provider has enough settings to make a live call."""

        if self.provider in KEYLESS_PROVIDERS:
            return True
        if self.provider == "custom":
            return bool(self.base_url)
        return bool(self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; raise on unknown providers."""

    key = (value or "").strip().lower()
    if key not in PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider '{value}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    return PROVIDER_ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are lazy so offline mode works without any provider SDK installed.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        elif provider == "custom":
            # Many OpenAI-compatible servers ignore the key but the client requires one.
            kwargs["api_key"] = "not-needed"
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": config.model_name, "temperature": config.temperature, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    # openrouter
    try:
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
    except ImportError:
        # OpenRouter is OpenAI-compatible, so ChatOpenAI is a safe fallback.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )
