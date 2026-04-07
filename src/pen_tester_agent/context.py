"""Context management with output truncation and compression."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def truncate_output(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars, appending a marker if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


class ContextManager:
    """Manages the conversation message list with truncation and summarization.

    Keeps the context within a token budget so the LLM doesn't silently
    lose the system prompt or early conversation history.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        system_prompt: str,
        task: str,
        *,
        max_context_tokens: int = 6000,
        max_output_chars: int = 2000,
        preserve_recent: int = 6,
    ) -> None:
        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        self.max_context_tokens = max_context_tokens
        self.max_output_chars = max_output_chars
        self.preserve_recent = preserve_recent

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self._messages.append({"role": "assistant", "content": content})

    def add_user(self, content: str) -> None:
        """Append a user message."""
        self._messages.append({"role": "user", "content": content})

    def add_tool_result(self, tool_name: str, output: str) -> None:
        """Append a tool result as a user message, truncating if needed."""
        truncated = truncate_output(output, self.max_output_chars)
        self._messages.append({
            "role": "user",
            "content": f"Tool '{tool_name}' output:\n{truncated}",
        })

    def token_estimate(self) -> int:
        """Estimate total tokens across all messages."""
        return sum(estimate_tokens(m["content"]) for m in self._messages)

    def get_messages(self) -> list[dict]:
        """Return the message list, compressing older messages if over budget."""
        if self.token_estimate() > self.max_context_tokens:
            self._compress()
        return self._messages

    def _compress(self) -> None:
        """Summarize older tool outputs to reclaim context space.

        Strategy: keep the system prompt (index 0), the original task (index 1),
        and the last `preserve_recent` messages intact. Compress everything in
        between by replacing tool output messages with short summaries.
        """
        # Protected: system prompt + original task at the front,
        # and the N most recent messages at the tail
        protected_head = 2
        protected_tail = min(self.preserve_recent, len(self._messages) - protected_head)

        if protected_head + protected_tail >= len(self._messages):
            return  # nothing to compress

        compressible_start = protected_head
        compressible_end = len(self._messages) - protected_tail

        compressed = []
        for msg in self._messages[compressible_start:compressible_end]:
            content = msg["content"]
            role = msg["role"]

            if role == "user" and content.startswith("Tool '"):
                # Extract tool name and first line of output for a summary
                first_line = content.split("\n", 2)
                tool_header = first_line[0]  # "Tool 'bash' output:"
                # Get a brief snippet of the output
                if len(first_line) > 1:
                    snippet = first_line[1][:100]
                    summary = f"{tool_header} {snippet}..."
                else:
                    summary = tool_header
                compressed.append({"role": role, "content": summary})
            elif role == "assistant" and len(content) > 200:
                # Trim long assistant reasoning, keep the ACTION line if present
                action_idx = content.find("ACTION:")
                if action_idx >= 0:
                    compressed.append({"role": role, "content": content[action_idx:]})
                else:
                    compressed.append({"role": role, "content": content[:200] + "..."})
            else:
                compressed.append(msg)

        self._messages = (
            self._messages[:protected_head]
            + compressed
            + self._messages[compressible_end:]
        )
