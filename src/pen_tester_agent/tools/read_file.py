"""File reading tool."""

import os

from .base import Tool, ToolResult


class ReadFileTool(Tool):
    """Read contents of a file from the filesystem."""

    name = "read_file"
    description = (
        "Read and return the contents of a file. Useful for analyzing "
        "configuration files for misconfigurations, reviewing logs for "
        "attack signatures, examining system files, and reading scan output."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute or relative file path to read",
        },
        "max_lines": {
            "type": "integer",
            "description": "Maximum number of lines to return (default: all)",
            "default": 0,
        },
    }
    requires_approval = True

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        max_lines = kwargs.get("max_lines", 0)

        if not path:
            return ToolResult(output="Error: no path provided", success=False)

        path = os.path.expanduser(path)

        if not os.path.exists(path):
            return ToolResult(output=f"Error: file not found: {path}", success=False)

        if not os.path.isfile(path):
            return ToolResult(output=f"Error: not a file: {path}", success=False)

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if max_lines > 0:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
            return ToolResult(output=content if content else "(empty file)")
        except OSError as exc:
            return ToolResult(output=f"Error reading file: {exc}", success=False)
