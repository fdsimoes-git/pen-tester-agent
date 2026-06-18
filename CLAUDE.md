# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A locally-running penetration testing agent powered by LLMs via Ollama. The agent proposes security testing commands and requires explicit user approval before executing each one. Includes an MLX-based GRPO fine-tuning module for Apple Silicon.

## Commands

```bash
# Run the agent (interactive or with a task)
uv run pen-tester-agent
uv run pen-tester-agent "scan open ports on 192.168.1.1"
# Runtime knobs: --model (default qwen3.6:35b), --max-iterations (50),
# --max-context-tokens (32000), --num-ctx (Ollama window; defaults to
# max-context-tokens + 8192), --no-plan (skip the upfront planning step)
uv run pen-tester-agent --model llama3.1:8b --max-iterations 5 "enum subdomains of example.com"

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_agent.py
uv run pytest tests/test_agent.py::test_tool_call_approved

# Lint (matches CI)
python -m pylint -v src/pen_tester_agent/

# Fine-tune a model (Apple Silicon only; needs the `training` extra: uv sync --group training)
uv run pen-tester-agent-train --low-memory --max-steps 250
```

Package manager is `uv`. Python 3.11+.

## Architecture

**Agent loop** (`agent.py`): Orchestrates the cycle of LLM reasoning → ACTION JSON extraction → user approval → tool execution → result feedback. The LLM responds with free-text reasoning followed by `ACTION: {"tool": "...", "args": {...}}` which is parsed via `find_action()` (brace-depth JSON extractor that handles nested objects, markdown code blocks, and multiple ACTION blocks). Each turn runs through `_run_iteration()`. Bash commands stream output live via `_execute_bash_streaming()` using `select`-based polling. Non-bash tools run with a spinner. LLM calls also show a spinner. Tool timeout is 60s with partial output capture on expiry. The loop runs `--max-iterations` turns per batch, then `ui.prompt_max_iterations()` offers continue/report/quit instead of silently dropping the session; it also returns early when the LLM calls the `done` tool (the only completion signal). When `plan=True` (CLI default; pass `--no-plan` to disable), `_run_planning()` first drafts a numbered plan and pins it via `ContextManager.set_plan()` so it survives compression for the whole session.

**UI layer** (`ui.py`): All terminal output goes through `rich` (panels, markdown rendering, themed colors). Interactive menus use `simple-term-menu` for arrow-key navigation. Falls back to numbered input in non-TTY environments (tests). Every user interaction point offers quit and generate-report options.

**Provider abstraction** (`providers/`): `ModelProvider` ABC decouples the agent from LLM backends. Currently only `OllamaProvider` (default model: `qwen3.6:35b`), which passes an explicit `num_ctx` so Ollama doesn't silently truncate the prompt to its ~4K default window. Adding a new provider means implementing a single `chat(messages) -> str` method.

**Tool system** (`tools/`): Each tool is a `Tool` subclass with `name`, `description`, `parameters` (JSON Schema), and `execute(**kwargs) -> ToolResult`. `ToolRegistry` collects tools and auto-generates the LLM system prompt schema. Tools: `bash`, `read_file`, `write_file`, `http_request`, `cve_search`, `done`. Most have `requires_approval=True`.

**Context management** (`context.py`): Keeps the LLM context within a token budget by compressing older tool outputs into summary snippets while always preserving the system prompt, original task, and the most recent messages.

**Executor** (`executor.py`): Thin wrapper that delegates to `ui.show_tool_approval_flow()` for the interactive approval menu.

**Prompts** (`prompts.py`): Builds the system prompt dynamically from the tool registry, defining six pen-testing domains and the ACTION response format. `find_action()` handles inline JSON, markdown code blocks, and colon-less ACTION formats.

**Training module** (`training/`): MLX GRPO fine-tuning on Apple Silicon. Uses LoRA weight swapping (single base model, no 3-model copies) for 8GB memory efficiency. Reward functions score format compliance, tool selection, command quality, and explanation quality.

## Conventions and invariants

Non-obvious rules enforced across the codebase — check changes against these before touching display, subprocess, or parser code (the Copilot review rules in `.github/instructions/` are the canonical list):

- **Rich markup injection**: never interpolate dynamic content (tool output, LLM responses, error text, user input) into Rich markup like `f"[red]{var}[/]"` — brackets in the value corrupt or crash output. Every such path in `ui.py` uses `Text()` objects or `markup=False`; keep it that way.
- **`shell=True` is intentional**: the bash tool runs LLM-proposed commands through the shell *by design* — the user approves each one. Don't treat it as a command-injection bug to fix.
- **Streaming is POSIX/Windows-split**: `_execute_bash_streaming()` uses `select()` + `os.read()` on POSIX and falls back to `communicate()` on Windows (`sys.platform == "win32"`); both guard `proc.kill()` against `ProcessLookupError`. Only the bash tool streams and enforces the 60s timeout — other tools are fast and intentionally have neither.
- **One ACTION parser**: `find_action()` in `prompts.py` (brace-depth counting, not regex) is the only parser; its `_ActionMatch` exposes `.group(1)` (JSON string) and `.start` (offset — slice with it, don't re-search the text).

## Testing

Tests use `FakeProvider` to mock LLM responses, `monkeypatch` for user input, and `pytest-httpx` for HTTP mocking. The `default_registry()` fixture in `conftest.py` provides a standard tool set. Arrow-key menus fall back to numbered input (`"0"` = first option, `"1"` = second, etc.) when `NO_TTY` (or `TERM=dumb`) is set — that env detection in `ui.py` is what makes the suite non-interactive. Report generation (`_generate_report` in `agent.py`) truncates long history with a head+tail split so the most recent findings survive.

## CI

GitHub Actions: `pylint.yml` lints on Python 3.11/3.12/3.13 every push (matches the lint command above); `claude-review.yml` runs an LLM PR review on open/sync and on `@claude` re-pings (gated to repo members, needs the `CLAUDE_CODE_OAUTH_TOKEN` secret); `publish.yml` publishes to PyPI. Review standards live in `.github/CODE_REVIEW.md`.
