import json
import requests


def test_whisper_transcribe_verbose_json(make_tool, monkeypatch):
    creds = {
        "azure_endpoint_whisper": "https://example.openai.azure.com",
        "azure_api_key_whisper": "key",
        "azure_api_version_whisper": "2024-02-01",
        "azure_deployment_whisper": "whisper-1",
    }
    tool = make_tool(creds)

    def fake_post(url, headers=None, data=None, files=None, timeout=None, stream=False):
        class Resp:
            status_code = 200
            def json(self):
                return {
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"},
                        {"id": 1, "start": 1.0, "end": 2.0, "text": "world"},
                    ]
                }
            text = "{}"
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)

    msgs = list(tool._invoke({
        "file": {"name": "a.wav", "type": "audio/wav", "content": b"x"},
        "transcription_type": "transcribe",
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": "segment",
        "stream": False,
    }))

    # Expect verbose_json payload
    assert any(m.type == "json" and "segments" in m.data.get("result", {}) for m in msgs)


def test_whisper_translate_fallback(make_tool, monkeypatch):
    creds = {
        "azure_endpoint_whisper": "https://example.openai.azure.com",
        "azure_api_key_whisper": "key",
        "azure_api_version_whisper": "2024-02-01",
        "azure_deployment_whisper": "whisper-1",
    }
    tool = make_tool(creds)

    # First translations call returns Chinese text; fallback should call transcriptions with translate=true
    calls = {"transcriptions": 0}
    def fake_post(url, headers=None, data=None, files=None, timeout=None, stream=False):
        class Resp:
            def __init__(self, status=200, text="", json_obj=None):
                self.status_code = status
                self.text = text
                self._json = json_obj
            def json(self):
                return self._json if self._json is not None else {"text": self.text}
        if "/audio/translations" in url:
            # Return non-English text
            return Resp(200, json_obj={"text": "你好世界"})
        if "/audio/transcriptions" in url:
            calls["transcriptions"] += 1
            return Resp(200, json_obj={"text": "Hello world"})
        return Resp(500, "unexpected")

    monkeypatch.setattr(requests, "post", fake_post)

    msgs = list(tool._invoke({
        "file": {"name": "a.wav", "type": "audio/wav", "content": b"x"},
        "transcription_type": "translate",
        "model": "whisper-1",
        "response_format": "text",
        "stream": False,
    }))

    # Ensure fallback applied and returned English
    assert calls["transcriptions"] >= 1
    assert any(m.type == "text" and m.text == "Hello world" for m in msgs)
