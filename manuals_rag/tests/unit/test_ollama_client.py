from manuals_rag_common.ollama import (
    build_chat_payload,
    capture_ollama_usage,
    chat_json,
    extract_chat_content,
    parse_json_content,
    model_family,
    summarize_ollama_usage,
    supports_thinking_control,
)


def test_model_family_detects_qwen_and_gpt_oss():
    assert model_family("qwen3.5:4b") == "qwen"
    assert model_family("gpt-oss:20b") == "gpt_oss"
    assert model_family("something-else") == "other"


def test_qwen_payload_disables_thinking_and_uses_json_schema():
    payload = build_chat_payload(
        model="qwen3.5:4b",
        messages=[{"role": "system", "content": "Return JSON."}, {"role": "user", "content": "Classify this."}],
        json_schema={"type": "object"},
        think=False,
        num_predict=-1,
        num_ctx=8192,
    )
    assert payload["think"] is False
    assert payload["format"] == {"type": "object"}
    assert payload["messages"][0]["content"].endswith("/no_think")
    assert payload["options"]["presence_penalty"] == 1.5
    assert payload["options"]["num_predict"] == -1
    assert payload["options"]["num_ctx"] == 8192


def test_gpt_oss_payload_omits_think_control():
    payload = build_chat_payload(
        model="gpt-oss:20b",
        messages=[{"role": "system", "content": "Return JSON."}, {"role": "user", "content": "Classify this."}],
        json_schema={"type": "object"},
        think=False,
    )
    assert "think" not in payload
    assert "/no_think" not in payload["messages"][0]["content"]
    assert payload["format"] == {"type": "object"}


def test_extract_chat_content_reads_message_content():
    assert extract_chat_content({"message": {"content": '{"ok":true}'}}) == '{"ok":true}'


def test_extract_chat_content_strips_inline_thinking_markup():
    payload = {"message": {"content": "<think>hidden reasoning</think>\n{\"ok\":true}"}}
    assert extract_chat_content(payload) == '{"ok":true}'


def test_parse_json_content_tolerates_raw_control_characters_in_strings():
    assert parse_json_content('{"reason":"line one\nline two"}') == {
        "reason": "line one\nline two"
    }


def test_parse_json_content_accepts_model_prose_and_fence_wrappers():
    assert parse_json_content('Result follows:\n```json\n{"entities": []}\n```') == {
        "entities": []
    }


def test_supports_thinking_control_for_qwen_only():
    assert supports_thinking_control("qwen3.5:4b") is True
    assert supports_thinking_control("gpt-oss:20b") is False


def test_chat_json_warms_requested_model_before_chat(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            calls.append(("GET", path, None))
            if path == "/api/tags":
                return FakeResponse({"models": [{"name": "gpt-oss:20b"}]})
            if path == "/api/ps":
                return FakeResponse({"models": []})
            raise AssertionError(path)

        def post(self, path, json):
            calls.append(("POST", path, json))
            if path == "/api/generate":
                return FakeResponse({"response": "", "done": True})
            if path == "/api/chat":
                return FakeResponse({"message": {"content": '{"ok": true}'}})
            raise AssertionError(path)

    monkeypatch.setattr("manuals_rag_common.ollama.httpx.Client", FakeClient)

    parsed, raw = chat_json(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Hi"}],
        json_schema={"type": "object"},
        num_ctx=8192,
    )

    assert parsed == {"ok": True}
    assert raw == '{"ok": true}'
    call_paths = [entry[1] for entry in calls]
    assert call_paths[:3] == ["/api/tags", "/api/ps", "/api/generate"]
    assert "/api/chat" in call_paths
    assert calls[2][2]["model"] == "gpt-oss:20b"
    assert calls[2][2]["options"]["num_ctx"] == 8192
    assert next(c[2] for c in calls if c[1] == "/api/chat")["options"]["num_ctx"] == 8192


def test_chat_json_reloads_and_retries_after_chat_failure(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            calls.append(("GET", path, None))
            if path == "/api/tags":
                return FakeResponse({"models": [{"name": "gpt-oss:20b"}]})
            if path == "/api/ps":
                return FakeResponse({"models": []})
            raise AssertionError(path)

        def post(self, path, json):
            calls.append(("POST", path, json))
            if path == "/api/generate":
                return FakeResponse({"response": "", "done": True})
            if path == "/api/chat":
                self.chat_calls += 1
                if self.chat_calls == 1:
                    raise RuntimeError("first chat failed")
                return FakeResponse({"message": {"content": '{"ok": true}'}})
            raise AssertionError(path)

    monkeypatch.setattr("manuals_rag_common.ollama.httpx.Client", FakeClient)

    parsed, _raw = chat_json(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Hi"}],
        json_schema={"type": "object"},
    )

    assert parsed == {"ok": True}
    assert [entry[1] for entry in calls].count("/api/generate") == 2
    assert [entry[1] for entry in calls].count("/api/chat") == 2


def test_usage_capture_reports_actual_ollama_token_and_duration_counts(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            return FakeResponse({"models": [{"name": "gpt-oss:20b"}]})

        def post(self, path, json):
            assert path == "/api/chat"
            return FakeResponse(
                {
                    "model": "gpt-oss:20b",
                    "message": {"content": '{"ok": true}'},
                    "prompt_eval_count": 120,
                    "eval_count": 15,
                    "total_duration": 250_000_000,
                }
            )

    monkeypatch.setattr("manuals_rag_common.ollama.httpx.Client", FakeClient)
    with capture_ollama_usage() as events:
        chat_json(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": "Hi"}],
            json_schema={"type": "object"},
            purpose="unit_usage",
        )

    usage = summarize_ollama_usage(events)
    assert usage["model_calls"] == 1
    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 15
    assert usage["total_tokens"] == 135
    assert usage["total_duration_ms"] == 250.0
    assert usage["by_purpose"]["unit_usage"]["model_calls"] == 1


def test_inference_timeout_does_not_reload_and_duplicate_work(monkeypatch):
    import httpx
    import pytest
    import manuals_rag_common.ollama as module
    loads = []
    monkeypatch.setattr(module, 'ensure_model_loaded', lambda **kwargs: loads.append(kwargs))
    def timed_out(**kwargs):
        raise httpx.ReadTimeout('inference deadline')
    monkeypatch.setattr(module, '_post_chat', timed_out)
    with pytest.raises(httpx.ReadTimeout):
        module.chat_json(model='test',messages=[],json_schema={'type':'object'},timeout=1)
    assert len(loads) == 1
    assert not loads[0].get('force_reload')


def test_empty_structured_output_is_not_success_or_reload(monkeypatch):
    import pytest
    import manuals_rag_common.ollama as module

    loads = []
    monkeypatch.setattr(module, "ensure_model_loaded", lambda **kw: loads.append(kw))
    body = {"message": {"content": "", "thinking": "unfinished"}, "done_reason": "length"}
    monkeypatch.setattr(module, "_post_chat", lambda **kw: body)
    monkeypatch.setattr(module, "_post_chat_stream", lambda **kw: body)
    for call in (module.chat_json, module.chat_json_stream):
        with pytest.raises(ValueError, match="empty structured output"):
            call(model="qwen3.5:9b", messages=[], json_schema={"type": "object"})
    assert len(loads) == 2
    assert not any(item.get("force_reload") for item in loads)
