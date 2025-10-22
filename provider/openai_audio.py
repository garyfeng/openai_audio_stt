from typing import Any
import requests

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class OpenaiAudioProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            # Normalize endpoints
            def _norm(ep: str | None) -> str | None:
                if not ep: return ep
                ep = ep.strip()
                while ep.endswith('/'):
                    ep = ep[:-1]
                if '://' in ep:
                    proto, rest = ep.split('://', 1)
                    rest = rest.replace('//', '/')
                    ep = f"{proto}://{rest}"
                return ep
            
            # Validate Azure Transcribe (GPT-4o) if configured
            azure_endpoint = _norm(credentials.get("azure_endpoint")) or _norm(credentials.get("azure_openai_transcribe_endpoint"))
            azure_api_key = credentials.get("azure_api_key") or credentials.get("azure_openai_transcribe_api_key") or credentials.get("api_key")
            azure_api_version = credentials.get("azure_api_version", credentials.get("azure_openai_transcribe_api_version", "2024-12-01-preview"))
            azure_deployment_gpt4o = credentials.get("azure_deployment_gpt4o") or credentials.get("azure_openai_transcribe_deployment") or credentials.get("azure_deployment")
            
            if azure_endpoint:
                if not azure_api_key:
                    raise ValueError("Azure Transcribe API key is required when azure_endpoint is set")
                headers = {"api-key": azure_api_key}
                url = f"{azure_endpoint}/openai/deployments?api-version={azure_api_version}"
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    msg = f"Azure Transcribe validation failed ({resp.status_code})"
                    try:
                        data = resp.json()
                        if "error" in data:
                            m = data["error"].get("message") or str(data["error"]) 
                            msg = f"Azure Transcribe validation failed: {m}"
                    except Exception:
                        pass
                    raise ValueError(msg)
                if azure_deployment_gpt4o:
                    try:
                        data = resp.json()
                        deployments = data.get("data") or data.get("value") or []
                        names = {d.get("name") for d in deployments if isinstance(d, dict)}
                        if azure_deployment_gpt4o not in names:
                            raise ValueError(f"Azure GPT-4o deployment '{azure_deployment_gpt4o}' not found. Available: {sorted(list(names))}")
                    except Exception:
                        pass
            
            # Validate Azure Whisper if configured
            whisper_endpoint = _norm(credentials.get("azure_endpoint_whisper")) or _norm(credentials.get("azure_openai_whisper_endpoint"))
            whisper_api_key = credentials.get("azure_api_key_whisper") or credentials.get("azure_openai_whisper_api_key") or credentials.get("api_key")
            whisper_api_version = credentials.get("azure_api_version_whisper", credentials.get("azure_openai_whisper_api_version", "2024-02-01"))
            whisper_deployment = credentials.get("azure_deployment_whisper") or credentials.get("azure_openai_whisper_deployment")
            
            if whisper_endpoint:
                if not whisper_api_key:
                    raise ValueError("Azure Whisper API key is required when azure_endpoint_whisper is set")
                headers = {"api-key": whisper_api_key}
                url = f"{whisper_endpoint}/openai/deployments?api-version={whisper_api_version}"
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    msg = f"Azure Whisper validation failed ({resp.status_code})"
                    try:
                        data = resp.json()
                        if "error" in data:
                            m = data["error"].get("message") or str(data["error"]) 
                            msg = f"Azure Whisper validation failed: {m}"
                    except Exception:
                        pass
                    raise ValueError(msg)
                if whisper_deployment:
                    try:
                        data = resp.json()
                        deployments = data.get("data") or data.get("value") or []
                        names = {d.get("name") for d in deployments if isinstance(d, dict)}
                        if whisper_deployment not in names:
                            raise ValueError(f"Azure Whisper deployment '{whisper_deployment}' not found. Available: {sorted(list(names))}")
                    except Exception:
                        pass
            
            # Fallback: OpenAI key validation (if Azure not set at all)
            if not azure_endpoint and not whisper_endpoint:
                api_key = credentials.get("api_key")
                if api_key:
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=15)
                    if response.status_code != 200:
                        error_message = f"API key validation failed with status code: {response.status_code}"
                        try:
                            error_data = response.json()
                            if "error" in error_data and "message" in error_data["error"]:
                                error_message = f"API key validation failed: {error_data['error']['message']}"
                        except Exception:
                            pass
                        raise ValueError(error_message)
                # If neither Azure nor OpenAI is provided, allow save; tool will error at invoke time if needed
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
