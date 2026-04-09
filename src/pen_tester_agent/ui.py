"""Terminal UI components for pen-tester-agent."""

import json
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from simple_term_menu import TerminalMenu

THEME = Theme({
    "assistant": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "tool": "bold magenta",
    "info": "dim white",
    "menu.option": "bold cyan",
})

console = Console(theme=THEME)

BANNER = (
    "[bold green]"
    "  ____              _____         _\n"
    " |  _ \\ ___  _ __  |_   _|__  ___| |_ ___ _ __\n"
    " | |_) / _ \\| '_ \\   | |/ _ \\/ __| __/ _ \\ '__|\n"
    " |  __/  __/| | | |  | |  __/\\__ \\ ||  __/ |\n"
    " |_|   \\___||_| |_|  |_|\\___||___/\\__\\___|_|\n"
    "              A G E N T   v1.0.0"
    "[/]"
)

MAIN_MENU_OPTIONS = [
    "  New penetration test task",
    "  Quit",
]
MAIN_MENU_KEYS = ["task", "quit"]

# Shared menu style kwargs
_MENU_STYLE = {
    "menu_cursor": "❯ ",
    "menu_cursor_style": ("fg_cyan", "bold"),
    "menu_highlight_style": ("fg_cyan", "bold"),
}


def _has_tty():
    """Check if a real terminal is available for interactive menus."""
    if os.environ.get("TERM") == "dumb" or os.environ.get("NO_TTY"):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except AttributeError:
        return False


def _arrow_menu(options, title=None):
    """Show an arrow-key menu and return the selected index (or None).

    Falls back to numbered input when no TTY is available (e.g. in tests).
    """
    if _has_tty():
        menu = TerminalMenu(options, title=title, **_MENU_STYLE)
        return menu.show()

    # Fallback for non-TTY (tests)
    if title:
        console.print(f"[bold]{title}[/]")
    for i, opt in enumerate(options):
        console.print(f"  [{i}] {opt.strip()}", markup=False)
    choice = input("").strip()
    try:
        idx = int(choice)
        if 0 <= idx < len(options):
            return idx
        return None
    except ValueError:
        return None


def show_banner():
    """Display the application banner."""
    console.print(
        Panel(
            BANNER,
            subtitle="Penetration Testing Assistant — Powered by Ollama",
            border_style="green",
            padding=(0, 2),
        )
    )


def show_menu():
    """Display an arrow-key navigable main menu and return the choice."""
    console.print()
    index = _arrow_menu(MAIN_MENU_OPTIONS, title="  Main Menu")
    if index is None:
        return "quit"
    return MAIN_MENU_KEYS[index]


def prompt_task():
    """Prompt the user for a penetration testing task."""
    console.print("\n[bold]Describe your task:[/] ", end="")
    return input("").strip()


def prompt_report_path():
    """Prompt the user for a report output path."""
    console.print(
        "[bold]Report output path[/] [info](default: pentest-report.txt)[/]: ",
        end="",
    )
    path = input("").strip()
    return path or "pentest-report.txt"


def show_goodbye():
    """Display a goodbye message."""
    console.print("\n[info]Goodbye![/]\n")


def show_no_task():
    """Display a message when no task is provided."""
    console.print("[warning]No task provided.[/]")


# -- Agent loop output --

def show_assistant(content):
    """Display an assistant response with markdown rendering."""
    console.print(
        Panel(
            Markdown(content),
            title="🤖 Assistant",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def show_warning(msg):
    """Display a warning message."""
    console.print(Text(f"⚠  {msg}", style="warning"))


def show_error(msg):
    """Display an error message."""
    console.print(Text(f"✖  {msg}", style="error"))


def show_success(msg):
    """Display a success message."""
    console.print(Text(f"✔  {msg}", style="success"))


def show_tool_executing(name):
    """Display a tool execution indicator."""
    console.print(Text(f"⚙  Executing: {name}", style="tool"))


def spinner_llm():
    """Return a rich Status context manager for LLM thinking."""
    return console.status("[cyan]🤖 Thinking...[/]", spinner="dots")


def spinner_tool(name):
    """Return a rich Status context manager for tool execution."""
    return console.status(f"[magenta]⚙  Running: {name}[/]", spinner="dots")


def stream_line(line):
    """Print a single line of streamed tool output."""
    console.print(
        Text.assemble(("  │ ", "dim"), line),
        end="",
        highlight=False,
    )


def show_tool_result(output, success):
    """Display a tool result in a styled panel."""
    style = "green" if success else "red"
    title = "✔ Result" if success else "✖ Result"
    console.print(
        Panel(Text(str(output)), title=title, border_style=style, padding=(0, 1))
    )


def prompt_followup():
    """Arrow-key menu after task completion: reply, report, or quit.

    Returns the user's text input, or special signals:
    - "" means quit
    - "__report__" means generate report
    """
    index = _arrow_menu([
        "  Continue chatting",
        "  Generate report from this session",
        "  Quit",
    ], title="  What next?")
    if index is None or index == 2:
        return ""
    if index == 1:
        return "__report__"
    console.print("[bold]Follow-up task:[/] ", end="")
    return input("").strip()


def prompt_user_input():
    """Arrow-key menu when the agent asks a question: reply, report, or quit.

    Returns the user's text input, or special signals:
    - "" means quit
    - "__report__" means generate report
    """
    index = _arrow_menu([
        "  Reply to assistant",
        "  Generate report from this session",
        "  Quit",
    ], title="  What next?")
    if index is None or index == 2:
        return ""
    if index == 1:
        return "__report__"
    console.print("[bold cyan]You:[/] ", end="")
    return input("").strip()


# -- Executor output --

MAX_ARG_DISPLAY_LENGTH = 200


def show_tool_approval(tool_name, args):
    """Display a tool approval request in a styled panel."""
    lines = Text()
    lines.append(f"Tool: {tool_name}\n", style="bold")
    for key, value in args.items():
        display = value if isinstance(value, str) else json.dumps(value)
        if len(str(display)) > MAX_ARG_DISPLAY_LENGTH:
            display = str(display)[:MAX_ARG_DISPLAY_LENGTH] + "..."
        lines.append(f"  {key}: {display}\n")

    console.print(
        Panel(
            lines,
            title="⚠ Tool Approval Required",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def prompt_approval():
    """Arrow-key menu for tool approval. Returns 'y', 'n', or 'e'."""
    index = _arrow_menu([
        "  Approve",
        "  Reject",
        "  Edit args",
    ], title="  Execute?")
    if index is None or index == 1:
        return "n"
    if index == 2:
        return "e"
    return "y"


def show_current_args(args):
    """Display current args for editing."""
    console.print("[info]Current args:[/]")
    console.print(json.dumps(args, indent=2), markup=False)


def prompt_edit_args():
    """Prompt for edited args as JSON."""
    console.print(
        "[bold]Enter new args as JSON[/] [info](empty to cancel)[/]: ", end=""
    )
    return input("").strip()


def show_info(msg):
    """Display an informational message."""
    console.print(Text(str(msg), style="info"))


def show_tool_approval_flow(tool_name, args):
    """Run the full tool approval interaction.

    Returns (approved_args, execute) tuple.
    """
    show_tool_approval(tool_name, args)

    while True:
        response = prompt_approval()
        if response == "y":
            return args, True
        if response == "n":
            return None, False
        if response == "e":
            show_current_args(args)
            edited = prompt_edit_args()
            if edited:
                try:
                    new_args = json.loads(edited)
                    if not isinstance(new_args, dict):
                        show_error("Args must be a JSON object, not "
                                   f"{type(new_args).__name__}.")
                        continue
                    return new_args, True
                except json.JSONDecodeError:
                    show_error(
                        "Invalid JSON. Select Edit again to retry."
                    )
            else:
                return None, False
