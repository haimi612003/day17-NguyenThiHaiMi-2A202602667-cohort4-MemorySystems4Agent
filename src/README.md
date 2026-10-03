# src/: Completed lab

| File | Role |
|---|---|
| `model_provider.py` | `ProviderConfig`, `normalize_provider()` (aliases such as `anthorpic`), `build_chat_model()` for openai / custom / gemini / anthropic / ollama / openrouter |
| `config.py` | `LabConfig` + `load_config()`: `.env`, paths, compact knobs, main + judge model, bonus knobs |
| `memory_store.py` | `estimate_tokens()`, entity extraction with confidence (`extract_profile_candidates`, `extract_profile_updates`), `UserProfileStore` (User.md read/write/edit + structured facts, conflict handling, decay), `summarize_messages()`, `CompactMemoryManager` |
| `offline_responder.py` | Deterministic answer composer shared by both agents, so the benchmark compares memory, not phrasing |
| `agent_baseline.py` | Agent A: within-thread memory only (`InMemorySaver` in live mode) |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact memory (live mode: `create_agent` with User.md tools) |
| `live_utils.py` | Text and usage extraction for LangChain messages |
| `benchmark.py` | Standard + Long-Context Stress benchmark, 6 required columns, per-turn prompt trace |
| `ablation.py` | Turns off each bonus guardrail to measure its effect |
| `test_agents.py` | 22 tests: User.md, compaction, cross-session recall, prompt load, corrections, noise, confidence, decay, live wiring |

Run from the repo root:

```bash
python src/benchmark.py        # add --verbose to see answers, --live to use a real LLM
python src/ablation.py
pytest src/test_agents.py -v
```

Analysis and bonus write-up: [`../REPORT.md`](../REPORT.md).
