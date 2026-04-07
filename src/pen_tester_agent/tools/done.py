"""Task completion signal tool."""

from .base import Tool, ToolResult


class DoneTool(Tool):
    """Signal that the current task is complete."""

    name = "done"
    description = (
        "Use this tool when the task is complete. "
        "Provide a summary of what was accomplished."
    )
    parameters = {
        "summary": {
            "type": "string",
            "description": "A brief summary of what was accomplished",
        },
    }
    requires_approval = False

    def execute(self, **kwargs) -> ToolResult:
        summary = kwargs.get("summary", "Task complete.")
        return ToolResult(output=summary, terminates=True)
