import io
import json
import types
import requests

def test_azure_transcribe_fallback(make_tool, monkeypatch):
    # Arrange: simulate Azure 404 for initial version then success on fallback
    creds = {
        "azure_endpoint_transcribe": "https://example.openai.azure.com",
        "azure_api_key_transcribe": "key",
        "azure_api_version_transcribe": "2024-12-01-preview",
        "azure_deployment_transcribe": "gpt-4o-transcribe",
    }
    tool = make_tool(creds)

    # Fake post with 404 on first call and 200 on fallback
    calls = {"n": 0}
    def fake_post(url, headers=None, data=None, files=None, timeout=None, stream=False):
        calls["n"] += 1
        class Resp:
            def __init__(self, status, text, json_obj=None):
                self.status_code = status
                self.text = text
                self._json = json_obj
            def json(self):
                if self._json is not None:
                    return self._json
                try:
                    return json.loads(self.text)
                except Exception:
                    return {"text": self.text}
            def iter_lines(self, decode_unicode=False):
                return iter(())
        if "api-version=2024-12-01-preview" in url:
            return Resp(404, "Resource not found")
        if "api-version=2024-02-15-preview" in url:
            return Resp(200, "hello world")
        return Resp(500, "unexpected")

    monkeypatch.setattr(requests, "post", fake_post)

    # Act: invoke with a small bytes content
    file_bytes = b"data"
    msgs = list(tool._invoke({
        "file": {"name": "a.wav", "type": "audio/wav", "content": file_bytes},
        "transcription_type": "transcribe",
        "model": "gpt-4o-transcribe",
        "response_format": "text",
        "stream": False,
    }))

    # Assert: result comes from fallback (hello world)
    assert any(m.type == "json" and m.data.get("result", {}).get("text") == "hello world" for m in msgs)


def test_azure_transcribe_streaming(make_tool, monkeypatch):
    creds = {
        "azure_endpoint_transcribe": "https://example.openai.azure.com",
        "azure_api_key_transcribe": "key",
        "azure_api_version_transcribe": "2024-02-15-preview",
        "azure_deployment_transcribe": "gpt-4o-transcribe",
    }
    tool = make_tool(creds)

    def fake_post(url, headers=None, data=None, files=None, timeout=None, stream=False):
        class Resp:
            status_code = 200
            def iter_lines(self, decode_unicode=False):
                return iter([
                    b"data: {\"type\": \"transcript.text.delta\", \"delta\": \"Hello \"}",
                    b"data: {\"type\": \"transcript.text.delta\", \"delta\": \"world\"}",
                    b"data: [DONE]",
                ])
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        return Resp()
    monkeypatch.setattr(requests, "post", fake_post)

    msgs = list(tool._invoke({
        "file": {"name": "a.wav", "type": "audio/wav", "content": b"x"},
        "transcription_type": "transcribe",
        "model": "gpt-4o-transcribe",
        "response_format": "text",
        "stream": True,
    }))

    # Should yield text chunks and a final JSON
    assert any(m.type == "text" and m.text == "Hello " for m in msgs)
    assert any(m.type == "text" and m.text == "world" for m in msgs)
    assert any(m.type == "json" and m.data.get("result", {}).get("text") == "Hello world" for m in msgs)
