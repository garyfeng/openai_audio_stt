#!/usr/bin/env python3
import pathlib
import sys
import json
from types import SimpleNamespace

# Allow running from repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dify_plugin.entities.tool import ToolInvokeMessage  # type: ignore
from tools.openai_audio import OpenaiAudioTool  # type: ignore

class FakeRuntime:
    def __init__(self, credentials: dict):
        self.credentials = credentials


def run(tool_params: dict, creds: dict):
    tool = OpenaiAudioTool()
    tool.runtime = FakeRuntime(credentials=creds)
    outputs = []
    for msg in tool._invoke(tool_params):
        if isinstance(msg, ToolInvokeMessage) and msg.type == "text":
            print(msg.text, end="", flush=True)
        else:
            try:
                print(json.dumps(msg.to_dict()))
            except Exception:
                print(str(msg))
        outputs.append(msg)
    print()
    return outputs


def load_env_credentials():
    # Load minimal creds from .env if available
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
    # Map common keys
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
    p.add_argument("--azure-deployment", default=None)
    args = p.parse_args()

    creds = load_env_credentials()
    if args.provider == "openai" and "api_key" not in creds:
        print("Missing OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)
    if args.provider == "azure":
        if "azure_endpoint" not in creds:
            print("Missing AZURE_OPENAI_ENDPOINT in .env", file=sys.stderr)
            sys.exit(1)
        if "azure_api_key" not in creds and "api_key" not in creds:
            print("Missing AZURE_OPENAI_API_KEY or OPENAI_API_KEY in .env", file=sys.stderr)
            sys.exit(1)
        if args.azure_deployment:
            creds["azure_deployment"] = args.azure_deployment

    audio_path = pathlib.Path(args.audio)
    data = audio_path.read_bytes()

    params = {
        "file": {"name": audio_path.name, "type": "application/octet-stream", "content": data},
        "transcription_type": "translate" if args.translate else "transcribe",
        "model": args.model,
        "response_format": args.response_format,
        "language": args.language,
        "stream": args.stream,
        "output_format": "default",
    }

    if args.azure_deployment:
        params["azure_deployment"] = args.azure_deployment

    run(params, creds)
