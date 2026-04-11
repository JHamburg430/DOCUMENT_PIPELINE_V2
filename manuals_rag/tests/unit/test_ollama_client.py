from manuals_rag_common.ollama import build_chat_payload, chat_json, extract_chat_content, model_family, supports_thinking_control


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
    )
    assert payload["think"] is False
    assert payload["format"] == {"type": "object"}
    assert payload["messages"][0]["content"].endswith("/no_think")
    assert payload["options"]["presence_penalty"] == 1.5
    assert payload["options"]["num_predict"] == -1


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
    )

    assert parsed == {"ok": True}
    assert raw == '{"ok": true}'
    call_paths = [entry[1] for entry in calls]
    assert call_paths[:3] == ["/api/tags", "/api/ps", "/api/generate"]
    assert "/api/chat" in call_paths
    assert calls[2][2]["model"] == "gpt-oss:20b"


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
