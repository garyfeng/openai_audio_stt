from typing import Any
import requests

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class OpenaiAudioProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            azure_endpoint = credentials.get("azure_endpoint")
            if azure_endpoint:
                api_key = credentials.get("azure_api_key") or credentials.get("api_key")
                if not api_key:
                    raise ValueError("Azure API key is required when azure_endpoint is set")
                version = credentials.get("azure_api_version", "2024-12-01-preview")
                headers = {"api-key": api_key}
                url = f"{azure_endpoint.rstrip('/')}/openai/deployments?api-version={version}"
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    error_message = f"Azure validation failed ({response.status_code})"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            msg = error_data["error"].get("message") or str(error_data["error"])
                            if msg:
                                error_message = f"Azure validation failed: {msg}"
                    except Exception:
                        pass
                    raise ValueError(error_message)
                deployment = credentials.get("azure_deployment")
                if deployment:
                    try:
                        data = response.json()
                        # Azure may return 'data' or 'value'
                        deployments = data.get("data") or data.get("value") or []
                        names = {d.get("name") for d in deployments if isinstance(d, dict)}
                        if deployment not in names:
                            raise ValueError(f"Azure deployment '{deployment}' not found. Available: {sorted(list(names))}")
                    except Exception:
                        pass
                return
            
            api_key = credentials.get("api_key")
            if not api_key:
                raise ValueError("API key is required")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                error_message = f"API key validation failed with status code: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data and "message" in error_data["error"]:
                        error_message = f"API key validation failed: {error_data['error']['message']}"
                except Exception:
                    pass
                raise ValueError(error_message)
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
