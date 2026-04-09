"""Core agent loop that orchestrates tool calls via an LLM."""

import json
import os
import select
import subprocess
import sys
import time

from .context import ContextManager
from .prompts import find_action, build_system_prompt
from .providers.base import ModelProvider
from .tools.base import ToolRegistry, ToolResult
from . import ui

TOOL_TIMEOUT = 60


def _execute_bash_streaming(approved_args):
    """Execute a bash command with live-streamed output.

    Uses select() + os.read() on POSIX for non-blocking streaming.
    Falls back to communicate() on Windows where select() doesn't
    work on pipes.
    """
    command = approved_args.get("command", "")
    timeout = approved_args.get("timeout", TOOL_TIMEOUT)
    try:
        timeout = int(timeout)
        if timeout <= 0:
            timeout = TOOL_TIMEOUT
    except (TypeError, ValueError):
        timeout = TOOL_TIMEOUT
    if not command:
        return ToolResult(output="Error: no command provided", success=False)

    # Windows: select() doesn't support pipes; fall back to communicate()
    if sys.platform == "win32":
        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            stdout, _ = proc.communicate(timeout=timeout)
            for line in (stdout or "").splitlines(keepends=True):
                ui.stream_line(line)
            return ToolResult(output=stdout if stdout else "(no output)")
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            stdout, _ = proc.communicate()
            msg = f"Command timed out after {timeout}s."
            if stdout:
                msg += f"\nPartial output:\n{stdout}"
            return ToolResult(output=msg, success=False)
        except OSError as exc:
            return ToolResult(output=f"Error: {exc}", success=False)

    # POSIX: non-blocking streaming via select() + os.read()
    # Merge stderr into stdout to avoid interleaved/corrupted lines
    lines = []
    try:
        proc = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        start = time.monotonic()
        buf = b""

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise subprocess.TimeoutExpired(command, timeout)

            readable, _, _ = select.select(
                [proc.stdout] if proc.stdout else [], [], [], 0.1,
            )

            for stream in readable:
                chunk = os.read(stream.fileno(), 4096)
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        line_bytes, buf = buf.split(b"\n", 1)
                        decoded = (
                            line_bytes.decode("utf-8", errors="replace")
                            + "\n"
                        )
                        lines.append(decoded)
                        ui.stream_line(decoded)

            if proc.poll() is not None:
                # Process exited — drain remaining pipe data
                if proc.stdout:
                    remaining = proc.stdout.read()
                    if remaining:
                        buf += remaining
                # Flush buffer
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    decoded = (
                        line_bytes.decode("utf-8", errors="replace") + "\n"
                    )
                    lines.append(decoded)
                    ui.stream_line(decoded)
                if buf:
                    decoded = buf.decode("utf-8", errors="replace")
                    lines.append(decoded)
                    ui.stream_line(decoded)
                break

        output = "".join(lines)
        return ToolResult(output=output if output else "(no output)")

    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        stdout_remaining, _ = proc.communicate()
        partial = "".join(lines)
        if stdout_remaining:
            partial += stdout_remaining.decode("utf-8", errors="replace")
        msg = f"Command timed out after {timeout}s."
        if partial:
            msg += f"\nPartial output:\n{partial}"
        return ToolResult(output=msg, success=False)
    except OSError as exc:
        return ToolResult(output=f"Error: {exc}", success=False)


def _execute_tool(tool, approved_args):
    """Execute a tool with appropriate UI feedback.

    Bash commands are streamed live with timeout enforcement.
    Non-bash tools run with a spinner but no timeout.
    """
    if tool.name == "bash":
        return _execute_bash_streaming(approved_args)
    with ui.spinner_tool(tool.name):
        return tool.execute(**approved_args)


def _generate_report(ctx, provider, registry, max_context_tokens):
    """Generate a report based on the current session context."""
    history = ""
    for msg in ctx.get_messages():
        if msg["role"] == "system":
            continue
        history += f"[{msg['role']}]: {msg['content']}\n\n"

    # Cap history to fit within context budget (reserve ~2000 chars for prompt)
    max_history_chars = max(0, max_context_tokens * 4 - 2000)
    if len(history) > max_history_chars:
        history = history[:max_history_chars] + "\n\n[... history truncated ...]"

    report_task = (
        "Based on the penetration testing session below, write a professional "
        "penetration test report in the following structure:\n\n"
        "1. EXECUTIVE SUMMARY — High-level overview of the engagement scope, "
        "approach, and key findings.\n"
        "2. SCOPE & METHODOLOGY — Target systems, tools used, and testing "
        "methodology.\n"
        "3. FINDINGS — Each finding as a numbered item with: description, "
        "affected host/service, evidence (include exact command outputs), "
        "severity (Critical/High/Medium/Low/Info), and any CVE references.\n"
        "4. RECOMMENDATIONS — Prioritised remediation steps for each finding.\n"
        "5. CONCLUSION — Summary of the overall security posture.\n\n"
        "Include ALL relevant tool outputs, port scan results, service "
        "versions, error messages, and any other evidence gathered during "
        "the session. Be thorough and specific — do not omit findings.\n\n"
        f"--- Session History ---\n{history}"
    )
    report_ctx = ContextManager(
        build_system_prompt(registry), report_task,
        max_context_tokens=max_context_tokens,
    )

    path = ui.prompt_report_path()
    with ui.spinner_llm():
        content = provider.chat(report_ctx.get_messages())

    # Strip any ACTION blocks from the response — we just want the text
    report_text = content
    action_match = find_action(content)
    if action_match:
        report_text = content[:action_match.start].strip()

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
    except OSError as exc:
        ui.show_assistant(report_text)
        ui.show_error(f"Could not save report to {path}: {exc}")
        return

    ui.show_assistant(report_text)
    ui.show_success(f"Report saved to {path}")


def _handle_action(content, action_match, ctx, registry, provider,
                   max_iterations, max_context_tokens):
    """Process a parsed ACTION from the assistant response.

    Returns True if the agent should stop (user quit), False otherwise.
    """
    try:
        action = json.loads(action_match.group(1))
    except json.JSONDecodeError:
        ui.show_warning("Could not parse tool call JSON. Asking LLM to retry.")
        ctx.add_assistant(content)
        ctx.add_user(
            "Your ACTION was not valid JSON. "
            "Please try again with correct JSON format."
        )
        return False

    tool_name = action.get("tool", "")
    args = action.get("args", {})

    if not isinstance(tool_name, str) or not isinstance(args, dict):
        ui.show_warning("Invalid ACTION format. Asking LLM to retry.")
        ctx.add_assistant(content)
        ctx.add_user(
            "Your ACTION had an invalid format. 'tool' must be a string "
            "and 'args' must be a JSON object. Please try again."
        )
        return False

    tool = registry.get(tool_name)
    if tool is None:
        available = ", ".join(t.name for t in registry.list_tools())
        ui.show_warning(f"Unknown tool: {tool_name}")
        ctx.add_assistant(content)
        ctx.add_user(
            f"Unknown tool '{tool_name}'. Available tools: {available}"
        )
        return False

    if tool.requires_approval:
        approved_args, execute = ui.show_tool_approval_flow(tool_name, args)
    else:
        approved_args, execute = args, True

    if not (execute and approved_args is not None):
        ui.show_error("Tool call cancelled.")
        ctx.add_assistant(content)
        ctx.add_user(
            f"Tool call '{tool_name}' cancelled by user. "
            "Suggest an alternative or ask what to do."
        )
        return False

    ui.show_tool_executing(tool_name)
    result = _execute_tool(tool, approved_args)

    if result.terminates:
        ui.show_success(f"Done: {result.output}")
        ctx.add_assistant(content)
        ctx.add_tool_result(tool_name, result.output)
        follow_up = ui.prompt_followup()
        if not follow_up:
            return True
        if follow_up == "__report__":
            _generate_report(ctx, provider, registry, max_context_tokens)
            return True
        ctx.add_user(follow_up)
    else:
        # Bash output was already streamed live — skip the result panel
        if tool_name != "bash":
            ui.show_tool_result(result.output, result.success)
        elif not result.success:
            ui.show_tool_result(result.output, result.success)
        ctx.add_assistant(content)
        ctx.add_tool_result(tool_name, result.output)

    return False


def agent_loop(
    task: str,
    provider: ModelProvider,
    registry: ToolRegistry,
    max_iterations: int = 15,
    max_context_tokens: int = 6000,
) -> None:
    """Run the agent loop: propose tool calls, get approval, execute, repeat."""
    system_prompt = build_system_prompt(registry)
    ctx = ContextManager(
        system_prompt, task, max_context_tokens=max_context_tokens,
    )

    for _ in range(max_iterations):
        with ui.spinner_llm():
            content = provider.chat(ctx.get_messages())

        action_match = find_action(content)

        if action_match:
            # Show only the reasoning, omit the ACTION block
            reasoning = content[:action_match.start].rstrip()
            if reasoning:
                ui.show_assistant(reasoning)
        else:
            ui.show_assistant(content)

        if action_match:
            if _handle_action(content, action_match, ctx, registry,
                              provider, max_iterations, max_context_tokens):
                return
        else:
            ctx.add_assistant(content)
            follow_up = ui.prompt_user_input()
            if not follow_up:
                return
            if follow_up == "__report__":
                _generate_report(
                    ctx, provider, registry, max_context_tokens,
                )
                return
            ctx.add_user(follow_up)

    ui.show_warning("Reached maximum iterations. Stopping.")
