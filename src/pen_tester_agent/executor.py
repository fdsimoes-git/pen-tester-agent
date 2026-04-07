"""Interactive tool-call approval for the agent loop."""

import json

MAX_ARG_DISPLAY_LENGTH = 200


def ask_tool_approval(tool_name: str, args: dict) -> tuple[dict | None, bool]:
    """Request user approval before executing a tool call."""
    print(f"\n⚠️  Proposed tool call: {tool_name}")
    for key, value in args.items():
        display = value if isinstance(value, str) else json.dumps(value)
        # Truncate long values for display
        if len(str(display)) > MAX_ARG_DISPLAY_LENGTH:
            display = str(display)[:MAX_ARG_DISPLAY_LENGTH] + "..."
        print(f"   {key}: {display}")

    while True:
        response = input("Execute? (y/n/e to edit): ").lower().strip()
        if response == "y":
            return args, True
        if response == "n":
            return None, False
        if response == "e":
            print(f"Current args: {json.dumps(args, indent=2)}")
            edited = input("Enter new args as JSON (or empty to cancel): ").strip()
            if edited:
                try:
                    new_args = json.loads(edited)
                    return new_args, True
                except json.JSONDecodeError:
                    print("Invalid JSON. Try again or press Enter to cancel.")
            else:
                return None, False
        else:
            print("Please type 'y', 'n', or 'e'")
