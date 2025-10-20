#!/usr/bin/env python3
import pathlib
import sys
import json
import types
import os

# Allow running from repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Lightweight mock of dify_plugin so we can run outside Dify and on Python <3.10
class _MockMsg:
    def __init__(self, kind: str, payload):
        self.type = kind
        if kind == "text":
            self.text = payload
        else:
            self.data = payload
    def to_dict(self):
        if self.type == "text":
            return {"type": "text", "text": getattr(self, "text", "")}
        return {"type": "json", "data": getattr(self, "data", {})}

class _MockTool:
    def __init__(self):
        self.runtime = types.SimpleNamespace(credentials={})
    def create_text_message(self, text: str):
        return _MockMsg("text", text)
    def create_json_message(self, data: dict):
        return _MockMsg("json", data)

mock_module = types.ModuleType("dify_plugin")
setattr(mock_module, "Tool", _MockTool)
entities_module = types.ModuleType("dify_plugin.entities")
entities_tool_module = types.ModuleType("dify_plugin.entities.tool")
setattr(entities_tool_module, "ToolInvokeMessage", _MockMsg)
sys.modules["dify_plugin"] = mock_module
sys.modules["dify_plugin.entities"] = entities_module
sys.modules["dify_plugin.entities.tool"] = entities_tool_module

from tools.openai_audio import OpenaiAudioTool  # type: ignore


def run(tool_params: dict, creds: dict, mock: bool = False):
    tool = OpenaiAudioTool()
    tool.runtime = types.SimpleNamespace(credentials=creds)

    # Optional mock HTTP layer
    if mock:
        import requests
        class _Resp:
            def __init__(self, status=200, text="mock transcript", json_obj=None, stream=False):
                self.status_code = status
                self.text = text
                self._json = json_obj if json_obj is not None else {"text": text}
                self._stream = stream
            def json(self):
                return self._json
            def iter_lines(self):
                if not self._stream:
                    return iter(())
                lines = [
                    b"data: {\"type\": \"transcript.text.delta\", \"delta\": \"Hello \"}",
                    b"data: {\"type\": \"transcript.text.delta\", \"delta\": \"world\"}",
                    b"data: [DONE]",
                ]
                return iter(lines)
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        def _fake_post(url, headers=None, data=None, files=None, stream=False):
            if stream:
                return _Resp(stream=True)
            return _Resp()
        requests.post = _fake_post  # type: ignore
        def _fake_get(url, headers=None, timeout=None):
            class _G:
                status_code = 200
                def json(self):
                    return {"data": [{"name": creds.get("azure_deployment", "gpt-4o-transcribe")}]} 
            return _G()
        requests.get = _fake_get  # type: ignore

    outputs = []
    for msg in tool._invoke(tool_params):
        # msg is _MockMsg in mock mode
        d = msg.to_dict() if hasattr(msg, "to_dict") else str(msg)
        print(json.dumps(d))
        outputs.append(d)
    return outputs


def load_env_credentials():
    env_path = REPO_ROOT / ".env"
    creds = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    mapped = {}
    if "OPENAI_API_KEY" in creds:
        mapped["api_key"] = creds["OPENAI_API_KEY"]
    if "AZURE_OPENAI_API_KEY" in creds:
        mapped["azure_api_key"] = creds["AZURE_OPENAI_API_KEY"]
    if "AZURE_OPENAI_ENDPOINT" in creds:
        mapped["azure_endpoint"] = creds["AZURE_OPENAI_ENDPOINT"]
    if "AZURE_OPENAI_DEPLOYMENT" in creds:
        mapped["azure_deployment"] = creds["AZURE_OPENAI_DEPLOYMENT"]
    if "AZURE_OPENAI_API_VERSION" in creds:
        mapped["azure_api_version"] = creds["AZURE_OPENAI_API_VERSION"]
    return mapped


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Local harness for openai_audio tool")
    p.add_argument("audio", help="Path to audio file (mp3/mp4/wav/...) to transcribe")
    p.add_argument("--provider", choices=["openai", "azure"], default="openai")
    p.add_argument("--model", default="gpt-4o-transcribe")
    p.add_argument("--translate", action="store_true", help="Use translation instead of transcription")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--response-format", default="text", choices=["text","json","verbose_json","srt","vtt"]) 
    p.add_argument("--language", default="")
    p.add_argument("--timestamp-granularities", default="none", choices=["none","segment","word","segment_and_word"], help="Whisper-only")
    p.add_argument("--azure-deployment", default=None)
    p.add_argument("--mock", action="store_true", help="Mock HTTP calls (no network)")
    args = p.parse_args()

    creds = load_env_credentials()
    if args.provider == "openai" and "api_key" not in creds and not args.mock:
        print("Missing OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)
    if args.provider == "azure":
        if "azure_endpoint" not in creds and not args.mock:
            print("Missing AZURE_OPENAI_ENDPOINT in .env", file=sys.stderr)
            sys.exit(1)
        if "azure_api_key" not in creds and "api_key" not in creds and not args.mock:
            print("Missing AZURE_OPENAI_API_KEY or OPENAI_API_KEY in .env", file=sys.stderr)
            sys.exit(1)
        if args.azure_deployment:
            creds["azure_deployment"] = args.azure_deployment

    audio_path = pathlib.Path(args.audio)
    data = audio_path.read_bytes()

    # Guess MIME from extension
    ext = audio_path.suffix.lower()
    mime = "application/octet-stream"
    if ext in [".mp3"]:
        mime = "audio/mpeg"
    elif ext in [".m4a", ".mp4"]:
        mime = "audio/mp4"
    elif ext in [".wav"]:
        mime = "audio/wav"
    elif ext in [".aiff", ".aif"]:
        mime = "audio/aiff"

    params = {
        "file": {"name": audio_path.name, "type": mime, "content": data},
        "transcription_type": "translate" if args.translate else "transcribe",
        "model": args.model,
        "response_format": args.response_format,
        "language": args.language,
        "timestamp_granularities": args.timestamp_granularities,
        "stream": args.stream,
        "output_format": "default",
    }

    if args.azure_deployment:
        params["azure_deployment"] = args.azure_deployment

    run(params, creds, mock=args.mock)
