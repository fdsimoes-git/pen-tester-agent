# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A locally-running penetration testing agent powered by LLMs via Ollama. The agent proposes security testing commands and requires explicit user approval before executing each one. Includes an MLX-based GRPO fine-tuning module for Apple Silicon.

## Commands

```bash
# Run the agent (interactive or with a task)
uv run pen-tester-agent
uv run pen-tester-agent "scan open ports on 192.168.1.1"

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_agent.py
uv run pytest tests/test_agent.py::test_tool_call_approved

# Lint (matches CI)
python -m pylint -v src/pen_tester_agent/

# Fine-tune a model (optional mlx dependencies required)
uv run pen-tester-agent-train --low-memory --max-steps 250
```

Package manager is `uv`. Python 3.11+.

## Architecture

**Agent loop** (`agent.py`): Orchestrates the cycle of LLM reasoning → ACTION JSON extraction → user approval → tool execution → result feedback. The LLM responds with free-text reasoning followed by `ACTION: {"tool": "...", "args": {...}}` which is parsed via `find_action()` (brace-depth JSON extractor that handles nested objects, markdown code blocks, and multiple ACTION blocks). Bash commands stream output live via `_execute_bash_streaming()` using `select`-based polling. Non-bash tools run with a spinner. LLM calls also show a spinner. Tool timeout is 60s with partial output capture on expiry.

**UI layer** (`ui.py`): All terminal output goes through `rich` (panels, markdown rendering, themed colors). Interactive menus use `simple-term-menu` for arrow-key navigation. Falls back to numbered input in non-TTY environments (tests). Every user interaction point offers quit and generate-report options.

**Provider abstraction** (`providers/`): `ModelProvider` ABC decouples the agent from LLM backends. Currently only `OllamaProvider` (default model: `qwen2.5-coder:3b`). Adding a new provider means implementing a single `chat(messages) -> str` method.

**Tool system** (`tools/`): Each tool is a `Tool` subclass with `name`, `description`, `parameters` (JSON Schema), and `execute(**kwargs) -> ToolResult`. `ToolRegistry` collects tools and auto-generates the LLM system prompt schema. Tools: `bash`, `read_file`, `write_file`, `http_request`, `cve_search`, `done`. Most have `requires_approval=True`.

**Context management** (`context.py`): Keeps the LLM context within a token budget by compressing older tool outputs into summary snippets while always preserving the system prompt, original task, and the most recent messages.

**Executor** (`executor.py`): Thin wrapper that delegates to `ui.show_tool_approval_flow()` for the interactive approval menu.

**Prompts** (`prompts.py`): Builds the system prompt dynamically from the tool registry, defining six pen-testing domains and the ACTION response format. `find_action()` handles inline JSON, markdown code blocks, and colon-less ACTION formats.

**Training module** (`training/`): MLX GRPO fine-tuning on Apple Silicon. Uses LoRA weight swapping (single base model, no 3-model copies) for 8GB memory efficiency. Reward functions score format compliance, tool selection, command quality, and explanation quality.

## Testing

Tests use `FakeProvider` to mock LLM responses, `monkeypatch` for user input, and `pytest-httpx` for HTTP mocking. The `default_registry()` fixture in `conftest.py` provides a standard tool set. Arrow-key menus fall back to numbered input (`"0"` = first option, `"1"` = second, etc.) in non-TTY test environments.

## CI

GitHub Actions runs pylint on Python 3.11, 3.12, 3.13 on every push.
