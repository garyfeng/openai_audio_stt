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
        def _fake_post(url, headers=None, data=None, files=None, stream=False, timeout=None):
            if stream:
                return _Resp(stream=True)
            return _Resp()
        requests.post = _fake_post  # type: ignore
        def _fake_get(url, headers=None, timeout=None):
            class _G:
                status_code = 200
                def json(self):
                    dep = (
                        creds.get("azure_deployment_transcribe")
                        or creds.get("azure_deployment_gpt4o")
                        or creds.get("azure_deployment")
                        or "gpt-4o-transcribe"
                    )
                    return {"data": [{"name": dep}]} 
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
    # Load from .env file (if present)
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    # Merge in process environment (so export VAR=... also works)
    for k, v in os.environ.items():
        if k not in creds and ("AZURE" in k or "OPENAI" in k):
            creds[k] = v
    mapped = {}
    # OpenAI
    if "OPENAI_API_KEY" in creds:
        mapped["api_key"] = creds["OPENAI_API_KEY"]
    # Azure TRANSCRIBE resource
    # Map to new transcribe-specific keys
    if "AZURE_OPENAI_TRANSCRIBE_API_KEY" in creds:
        mapped["azure_api_key_transcribe"] = creds["AZURE_OPENAI_TRANSCRIBE_API_KEY"]
    if "AZURE_OPENAI_TRANSCRIBE_ENDPOINT" in creds:
        mapped["azure_endpoint_transcribe"] = creds["AZURE_OPENAI_TRANSCRIBE_ENDPOINT"]
    if "AZURE_OPENAI_TRANSCRIBE_API_VERSION" in creds:
        mapped["azure_api_version_transcribe"] = creds["AZURE_OPENAI_TRANSCRIBE_API_VERSION"]
    if "AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT" in creds:
        mapped["azure_deployment_transcribe"] = creds["AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT"]

    # Azure WHISPER resource
    if "AZURE_OPENAI_WHISPER_API_KEY" in creds:
        mapped["azure_api_key_whisper"] = creds["AZURE_OPENAI_WHISPER_API_KEY"]
    if "AZURE_OPENAI_WHISPER_ENDPOINT" in creds:
        mapped["azure_endpoint_whisper"] = creds["AZURE_OPENAI_WHISPER_ENDPOINT"]
    if "AZURE_OPENAI_WHISPER_API_VERSION" in creds:
        mapped["azure_api_version_whisper"] = creds["AZURE_OPENAI_WHISPER_API_VERSION"]
    if "AZURE_OPENAI_WHISPER_DEPLOYMENT" in creds:
        mapped["azure_deployment_whisper"] = creds["AZURE_OPENAI_WHISPER_DEPLOYMENT"]

    # Azure generic fallback (if not provided)
    for key_name in ["AZURE_OPENAI_API_KEY","AZURE_OPENAI_KEY","AZURE_API_KEY"]:
        if "azure_api_key" not in mapped and key_name in creds:
            mapped["azure_api_key"] = creds[key_name]
            break
    for ep_name in ["AZURE_OPENAI_ENDPOINT","AZURE_ENDPOINT","AZURE_OPENAI_RESOURCE_ENDPOINT"]:
        if "azure_endpoint" not in mapped and ep_name in creds:
            mapped["azure_endpoint"] = creds[ep_name]
            break
    for ver_name in ["AZURE_OPENAI_API_VERSION","AZURE_API_VERSION"]:
        if "azure_api_version" not in mapped and ver_name in creds:
            mapped["azure_api_version"] = creds[ver_name]
            break
    # Deployment names (multiple conventions)
    # Generic/legacy
    generic = creds.get("AZURE_OPENAI_DEPLOYMENT") or creds.get("AZURE_DEPLOYMENT")
    # GPT-4o/transcribe variants
    gpt4o = creds.get("AZURE_OPENAI_GPT4O_DEPLOYMENT") or \
            creds.get("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT") or \
            creds.get("AZURE_OPENAI_DEPLOYMENT_GPT4O") or \
            creds.get("AZURE_OPENAI_DEPLOYMENT_TRANSCRIBE") or \
            creds.get("AZURE_GPT4O_DEPLOYMENT") or \
            creds.get("AZURE_TRANSCRIBE_DEPLOYMENT") or \
            creds.get("AZURE_GPT4O_TRANSCRIBE_DEPLOYMENT") or \
            creds.get("AZURE_AUDIO_TRANSCRIBE_DEPLOYMENT")
    # Whisper variants
    whisper = creds.get("AZURE_OPENAI_WHISPER_DEPLOYMENT") or \
              creds.get("AZURE_OPENAI_DEPLOYMENT_WHISPER") or \
              creds.get("AZURE_WHISPER_DEPLOYMENT") or \
              creds.get("AZURE_AUDIO_WHISPER_DEPLOYMENT")
    # Heuristic fallback: scan for any AZURE*DEPLOYMENT values
    if not gpt4o or not whisper:
        for k, v in creds.items():
            lk = k.lower()
            if "deployment" in lk and "azure" in lk:
                vv = (v or "").lower()
                if (not whisper) and ("whisper" in vv):
                    whisper = v
                elif (not gpt4o) and ("4o" in vv or "transcribe" in vv or "gpt-4o" in vv):
                    gpt4o = v
                elif not generic:
                    generic = v
    # Store separately to decide based on model later
    if gpt4o:
        mapped["azure_deployment_gpt4o"] = gpt4o
    if whisper:
        mapped["azure_deployment_whisper"] = whisper
    # Fallback generic (if only one provided)
    if generic and "azure_deployment_gpt4o" not in mapped and "azure_deployment_whisper" not in mapped:
        mapped["azure_deployment"] = generic
    # Final generic detection for endpoint/key if still missing
    if "azure_endpoint" not in mapped:
        for k, v in creds.items():
            lk = k.lower()
            if "azure" in lk and "endpoint" in lk and v.startswith("http"):
                mapped["azure_endpoint"] = v
                break
    if "azure_api_key" not in mapped:
        for k, v in creds.items():
            lk = k.lower()
            if "azure" in lk and "key" in lk and len(v) > 10:
                mapped["azure_api_key"] = v
                break
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
