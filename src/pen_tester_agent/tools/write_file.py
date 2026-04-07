"""File writing tool."""

import os

from .base import Tool, ToolResult


class WriteFileTool(Tool):
    """Write content to a file on the filesystem."""

    name = "write_file"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Useful for saving penetration test reports, "
        "documenting findings, and writing scan configurations."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute or relative file path to write to",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
    }
    requires_approval = True

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not path:
            return ToolResult(output="Error: no path provided", success=False)

        path = os.path.expanduser(path)

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(output=f"Successfully wrote {len(content)} bytes to {path}")
        except OSError as exc:
            return ToolResult(output=f"Error writing file: {exc}", success=False)
