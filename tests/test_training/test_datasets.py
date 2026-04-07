"""Tests for dataset generation."""

from pen_tester_agent.training.datasets import SECURITY_TASKS, _build_system_context, build_dataset


class TestSecurityTasks:
    def test_tasks_not_empty(self):
        assert len(SECURITY_TASKS) > 0

    def test_tasks_are_strings(self):
        for task in SECURITY_TASKS:
            assert isinstance(task, str)
            assert len(task) > 10

    def test_tasks_cover_categories(self):
        """Ensure tasks span multiple security categories."""
        text = " ".join(SECURITY_TASKS).lower()
        assert "nmap" in text
        assert "gobuster" in text or "ffuf" in text or "dirb" in text
        assert "read" in text
        assert "write" in text or "report" in text
        assert "subdomain" in text or "dns" in text


class TestBuildSystemContext:
    def test_system_context_contains_tools(self):
        context = _build_system_context()
        assert "bash" in context
        assert "read_file" in context
        assert "write_file" in context
        assert "ACTION" in context


class TestBuildDataset:
    def test_build_dataset_returns_correct_size(self):
        ds = build_dataset(size=10)
        assert len(ds) == 10

    def test_dataset_items_have_prompt(self):
        ds = build_dataset(size=5)
        for item in ds:
            assert "prompt" in item
            messages = item["prompt"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_dataset_cycles_tasks(self):
        ds = build_dataset(size=len(SECURITY_TASKS) + 5)
        # First and (len+1)th tasks should be the same
        assert ds[0]["prompt"][1]["content"] == ds[len(SECURITY_TASKS)]["prompt"][1]["content"]

    def test_returns_list(self):
        ds = build_dataset(size=3)
        assert isinstance(ds, list)
        assert isinstance(ds[0], dict)
