from pen_tester_agent.providers.ollama import OllamaProvider, _DEFAULT_NUM_CTX


def _patch_chat(monkeypatch, captured):
    def fake_chat(model, messages, options=None):
        captured["model"] = model
        captured["messages"] = messages
        captured["options"] = options
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(
        "pen_tester_agent.providers.ollama.ollama.chat", fake_chat
    )


def test_num_ctx_passed_as_option(monkeypatch):
    captured = {}
    _patch_chat(monkeypatch, captured)
    provider = OllamaProvider(model="m", num_ctx=24000)
    result = provider.chat([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert captured["options"] == {"num_ctx": 24000}


def test_default_num_ctx_avoids_truncation(monkeypatch):
    # A direct provider with no explicit num_ctx must still set a real window,
    # not fall back to Ollama's ~4K default.
    captured = {}
    _patch_chat(monkeypatch, captured)
    provider = OllamaProvider(model="m")
    provider.chat([{"role": "user", "content": "hi"}])
    assert captured["options"] == {"num_ctx": _DEFAULT_NUM_CTX}
    assert _DEFAULT_NUM_CTX > 4096


def test_explicit_none_num_ctx_defers_to_ollama(monkeypatch):
    captured = {}
    _patch_chat(monkeypatch, captured)
    provider = OllamaProvider(model="m", num_ctx=None)
    provider.chat([{"role": "user", "content": "hi"}])
    assert captured["options"] is None


def test_default_model_is_reasoning_variant():
    assert OllamaProvider().model == "qwen3.6:35b"
