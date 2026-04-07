from pen_tester_agent.context import ContextManager, estimate_tokens, truncate_output


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hello") == 1

    def test_longer_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestTruncateOutput:
    def test_short_text_unchanged(self):
        assert truncate_output("hello", max_chars=100) == "hello"

    def test_exact_limit_unchanged(self):
        text = "x" * 100
        assert truncate_output(text, max_chars=100) == text

    def test_over_limit_truncated(self):
        text = "x" * 200
        result = truncate_output(text, max_chars=100)
        assert len(result) < 200
        assert result.endswith("... (truncated)")
        assert result.startswith("x" * 100)

    def test_default_limit(self):
        short = "hello"
        assert truncate_output(short) == short
        long = "x" * 5000
        result = truncate_output(long)
        assert "truncated" in result


class TestContextManager:
    def test_initial_messages(self):
        ctx = ContextManager("system prompt", "do something")
        msgs = ctx.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "system prompt"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "do something"

    def test_add_assistant(self):
        ctx = ContextManager("sys", "task")
        ctx.add_assistant("I'll help")
        msgs = ctx.get_messages()
        assert len(msgs) == 3
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "I'll help"

    def test_add_user(self):
        ctx = ContextManager("sys", "task")
        ctx.add_user("more info")
        msgs = ctx.get_messages()
        assert len(msgs) == 3
        assert msgs[2]["role"] == "user"

    def test_add_tool_result_short(self):
        ctx = ContextManager("sys", "task", max_output_chars=1000)
        ctx.add_tool_result("bash", "hello world")
        msgs = ctx.get_messages()
        assert msgs[2]["content"] == "Tool 'bash' output:\nhello world"

    def test_add_tool_result_truncates_long_output(self):
        ctx = ContextManager("sys", "task", max_output_chars=50)
        long_output = "x" * 200
        ctx.add_tool_result("bash", long_output)
        msgs = ctx.get_messages()
        assert "truncated" in msgs[2]["content"]
        assert len(msgs[2]["content"]) < 200

    def test_token_estimate(self):
        ctx = ContextManager("a" * 400, "b" * 400)
        # ~200 tokens for system + task
        assert ctx.token_estimate() > 0

    def test_compression_preserves_system_prompt(self):
        ctx = ContextManager("system prompt", "task", max_context_tokens=100, preserve_recent=2)
        # Fill with many messages to blow the budget
        for i in range(20):
            ctx.add_assistant(f"response {i} " + "x" * 200)
            ctx.add_tool_result("bash", "output " + "y" * 200)
        msgs = ctx.get_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "system prompt"

    def test_compression_preserves_original_task(self):
        ctx = ContextManager("sys", "original task", max_context_tokens=100, preserve_recent=2)
        for i in range(20):
            ctx.add_assistant(f"resp {i} " + "x" * 200)
            ctx.add_tool_result("bash", "out " + "y" * 200)
        msgs = ctx.get_messages()
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "original task"

    def test_compression_preserves_recent_messages(self):
        ctx = ContextManager("sys", "task", max_context_tokens=100, preserve_recent=4)
        for i in range(20):
            ctx.add_assistant(f"response {i}")
            ctx.add_tool_result("bash", f"output {i}")
        msgs = ctx.get_messages()
        # Last 4 messages should be the most recent ones
        recent = msgs[-4:]
        assert any("response 19" in m["content"] for m in recent)
        assert any("output 19" in m["content"] for m in recent)

    def test_no_compression_when_under_budget(self):
        ctx = ContextManager("sys", "task", max_context_tokens=100000)
        ctx.add_assistant("short")
        ctx.add_tool_result("bash", "short output")
        msgs = ctx.get_messages()
        assert len(msgs) == 4  # system + task + assistant + tool result

    def test_tool_output_summaries_are_shorter(self):
        ctx = ContextManager("sys", "task", max_context_tokens=100, preserve_recent=2)
        long_output = "very long output content\n" * 100
        for _ in range(10):
            ctx.add_assistant("thinking...")
            ctx.add_tool_result("bash", long_output)

        # Before compression
        before_tokens = ctx.token_estimate()

        # Trigger compression
        msgs = ctx.get_messages()

        # After compression, total should be smaller
        after_tokens = sum(estimate_tokens(m["content"]) for m in msgs)
        assert after_tokens < before_tokens

    def test_assistant_action_preserved_in_compression(self):
        ctx = ContextManager("sys", "task", max_context_tokens=100, preserve_recent=2)
        long_reasoning = "x" * 300 + '\nACTION: {"tool": "bash", "args": {"command": "ls"}}'
        ctx.add_assistant(long_reasoning)
        ctx.add_tool_result("bash", "file1\nfile2")
        # Add more to push over budget
        for _ in range(10):
            ctx.add_assistant("x" * 200)
            ctx.add_tool_result("bash", "y" * 200)
        msgs = ctx.get_messages()
        # The compressed assistant message should keep the ACTION line
        compressed_assistants = [m for m in msgs if m["role"] == "assistant" and "ACTION:" in m["content"]]
        assert len(compressed_assistants) >= 1
