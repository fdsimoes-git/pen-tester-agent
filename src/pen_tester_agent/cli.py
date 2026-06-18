"""Command-line interface for pen-tester-agent."""

import argparse

from .agent import agent_loop
from .providers import OllamaProvider
from .tools import default_registry
from . import ui



def main() -> None:
    """Parse arguments and start the agent loop."""
    parser = argparse.ArgumentParser(
        prog="pen-tester-agent",
        description="Penetration testing agent powered by local LLMs via Ollama",
    )
    parser.add_argument(
        "task", nargs="?", default=None,
        help="Task to perform (interactive if omitted)",
    )
    parser.add_argument(
        "--model", default="qwen3.6:35b",
        help="Ollama model to use (default: qwen3.6:35b)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=50,
        help="Iterations before prompting to continue/report (default: 50)",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=32000,
        help="Max context token budget for the prompt (default: 32000)",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help="Ollama context window (num_ctx). Defaults to "
             "max-context-tokens + 8192 headroom for the response.",
    )
    parser.add_argument(
        "--no-plan", action="store_true",
        help="Skip the upfront planning step.",
    )
    args = parser.parse_args()

    num_ctx = (
        args.num_ctx if args.num_ctx is not None
        else args.max_context_tokens + 8192
    )
    provider = OllamaProvider(model=args.model, num_ctx=num_ctx)
    registry = default_registry()

    if args.task:
        agent_loop(
            args.task, provider, registry,
            max_iterations=args.max_iterations,
            max_context_tokens=args.max_context_tokens,
            plan=not args.no_plan,
        )
        return

    ui.show_banner()

    while True:
        choice = ui.show_menu()

        if choice == "quit":
            ui.show_goodbye()
            break

        if choice == "task":
            task = ui.prompt_task()
            if not task:
                ui.show_no_task()
                continue
            agent_loop(
                task, provider, registry,
                max_iterations=args.max_iterations,
                max_context_tokens=args.max_context_tokens,
                plan=not args.no_plan,
            )
