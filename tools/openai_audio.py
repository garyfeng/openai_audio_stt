from collections.abc import Generator
# ruff: noqa

from typing import Any, Optional
import requests
import json
import tempfile
import pathlib
import os

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class OpenaiAudioTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        # Read a sane HTTP timeout from environment (connect, read)
        DEFAULT_TIMEOUT = int(os.getenv("MAX_REQUEST_TIMEOUT", "120"))
        HTTP_TIMEOUT = (10, DEFAULT_TIMEOUT)

        # Credentials
        api_key = self.runtime.credentials.get("api_key")
        azure_endpoint_transcribe = self.runtime.credentials.get("azure_endpoint_transcribe") or self.runtime.credentials.get("azure_endpoint")
        azure_endpoint = azure_endpoint_transcribe
        if azure_endpoint:
            # Normalize endpoint: remove trailing slashes and collapse duplicate slashes
            azure_endpoint = azure_endpoint.strip()
            while azure_endpoint.endswith('/'):
                azure_endpoint = azure_endpoint[:-1]
            proto_sep = '://'
            if proto_sep in azure_endpoint:
                proto, rest = azure_endpoint.split(proto_sep, 1)
                rest = rest.replace('//', '/')
                azure_endpoint = f"{proto}{proto_sep}{rest}"
        # Multi-resource support: allow separate whisper credentials
        azure_api_key_whisper = self.runtime.credentials.get("azure_api_key_whisper")
        azure_endpoint_whisper = self.runtime.credentials.get("azure_endpoint_whisper")
        azure_api_key = self.runtime.credentials.get("azure_api_key_transcribe") or self.runtime.credentials.get("azure_api_key") or api_key
        azure_api_version = self.runtime.credentials.get("azure_api_version_transcribe") or self.runtime.credentials.get("azure_api_version") or "2024-12-01-preview"
        azure_api_version_whisper = self.runtime.credentials.get("azure_api_version_whisper") or "2024-02-01"
        
        if not api_key and not (azure_endpoint or azure_endpoint_whisper):
            raise Exception("API key not found in credentials")
        
        # Parameters
        file_data = tool_parameters.get("file")
        transcription_type = tool_parameters.get("transcription_type", "transcribe")
        model = tool_parameters.get("model", "gpt-4o-transcribe")
        response_format = tool_parameters.get("response_format", "text")
        prompt = tool_parameters.get("prompt", "")
        language = tool_parameters.get("language", "")
        timestamp_granularities = tool_parameters.get("timestamp_granularities", "none")
        stream = tool_parameters.get("stream", False)
        output_format = tool_parameters.get("output_format", "default")
        azure_deployment_override = tool_parameters.get("azure_deployment")
        
        # Determine endpoint & model rules
        is_azure = bool(azure_endpoint)
        is_azure_whisper = bool(azure_endpoint_whisper)
        # Choose deployment intelligently if Azure
        azure_deployment_transcribe = self.runtime.credentials.get("azure_deployment_transcribe")
        azure_deployment_whisper = self.runtime.credentials.get("azure_deployment_whisper")
        selected_deployment = None
        
        # Enforce Whisper for translation
        if transcription_type == "translate":
            # For OpenAI, translation only supports whisper-1
            model = "whisper-1"
            # Force Whisper resource and deployment for Azure translate if available
            if azure_endpoint_whisper:
                azure_endpoint = azure_endpoint_whisper
            if azure_api_key_whisper:
                azure_api_key = azure_api_key_whisper
            # Use whisper API version if provided
            if azure_api_version_whisper:
                azure_api_version = azure_api_version_whisper
            # Require whisper deployment
            selected_deployment = azure_deployment_whisper
            if not selected_deployment:
                raise Exception("Translation requires an Azure Whisper deployment (azure_deployment_whisper)")
            # Recompute Azure effective flag after endpoint override
            is_azure = bool(azure_endpoint)
        
        # If not translation, select deployment normally
        if is_azure and transcription_type != "translate":
            if azure_deployment_override:
                selected_deployment = azure_deployment_override
            else:
                # If model hints whisper, pick whisper deployment; else pick transcribe
                if model == "whisper-1" and azure_deployment_whisper:
                    selected_deployment = azure_deployment_whisper
                    # Switch to whisper resource/key/version if provided
                    if azure_endpoint_whisper:
                        azure_endpoint = azure_endpoint_whisper
                    if azure_api_key_whisper:
                        azure_api_key = azure_api_key_whisper
                    if azure_api_version_whisper:
                        azure_api_version = azure_api_version_whisper
                else:
                    selected_deployment = (
                        azure_deployment_transcribe
                        or self.runtime.credentials.get("azure_deployment_gpt4o")  # legacy alias
                        or self.runtime.credentials.get("azure_deployment")        # legacy generic
                    )
            if not selected_deployment:
                raise Exception("Azure deployment name is required (provide azure_deployment_transcribe or set whisper/transcribe deployment in credentials)")
        
        # Build endpoint with API version; Whisper can have a different api-version
        def _build_azure_url(path_kind: str, version_override: Optional[str] = None) -> str:
            # Decide endpoint & key based on model (whisper can be a separate resource)
            endpoint_to_use = azure_endpoint
            ver = version_override or azure_api_version
            # Normalize endpoint again (defensive)
            endpoint = (endpoint_to_use or "").strip().rstrip('/')
            return f"{endpoint}/openai/deployments/{selected_deployment}/audio/{path_kind}?api-version={ver}"
        
        # Determine path kind and endpoint
        path_kind = "translations" if transcription_type == "translate" else "transcriptions"
        
        if transcription_type == "translate":
            if not is_azure:
                # OpenAI translate supports only whisper-1
                api_endpoint = "https://api.openai.com/v1/audio/translations"
            else:
                api_endpoint = _build_azure_url(path_kind)
        else:
            if not is_azure:
                api_endpoint = "https://api.openai.com/v1/audio/transcriptions"
            else:
                api_endpoint = _build_azure_url(path_kind)
        
        # Enforce format constraints: Whisper-only advanced formats
        if model != "whisper-1":
            if response_format in ["verbose_json", "srt", "vtt"]:
                response_format = "text"
            if timestamp_granularities != "none":
                timestamp_granularities = "none"
        
        # Streaming support: only for GPT-4o models
        if stream:
            if not is_azure:
                if not model.startswith("gpt-4o"):
                    stream = False
            else:
                # For Azure, rely on deployment being GPT-4o to stream; if user selected Whisper translate, disable stream
                if transcription_type == "translate":
                    stream = False
        
        if not file_data:
            raise Exception("No audio file provided")
            
        try:
            if isinstance(file_data, dict):
                file_content = file_data.get("content")
                file_name = file_data.get("name", "audio_file")
                file_type = file_data.get("type", "")
            elif hasattr(file_data, "read"):
                file_content = file_data.read()
                file_name = getattr(file_data, "name", "audio_file")
                file_type = ""
            elif str(type(file_data)).find("dify_plugin.file.file.File") >= 0:
                original_filename = ""
                file_extension = ""
                
                if hasattr(file_data, "filename") and file_data.filename:
                    original_filename = file_data.filename
                    
                if hasattr(file_data, "extension") and file_data.extension:
                    file_extension = file_data.extension
                    
                if hasattr(file_data, "url"):
                    try:
                        file_response = requests.get(file_data.url, timeout=HTTP_TIMEOUT)
                        if file_response.status_code == 200:
                            # Basic size guard (25MB) if content-length present
                            cl = file_response.headers.get("Content-Length")
                            if cl and int(cl) > 25 * 1024 * 1024:
                                raise Exception("Audio file too large (>25MB)")
                            file_content = file_response.content
                        else:
                            raise Exception(f"Failed to download file from URL: {file_response.status_code}")
                    except Exception as download_error:
                        raise Exception(f"Error downloading file from URL: {str(download_error)}")
                elif hasattr(file_data, "content"):
                    file_content = file_data.content
                elif hasattr(file_data, "read") and callable(file_data.read):
                    file_content = file_data.read()
                else:
                    raise Exception("Dify File object does not have accessible content")
                
                if original_filename:
                    file_name = original_filename
                elif file_extension:
                    file_name = f"audio_file{file_extension}"
                else:
                    file_name = "audio_file.mp4"
                
                if hasattr(file_data, "type"):
                    file_type = file_data.type
                elif hasattr(file_data, "mime_type"):
                    file_type = file_data.mime_type
                else:
                    file_type = ""
            else:
                raise Exception(f"Unsupported file data type: {type(file_data)}")
            
            if not file_content:
                raise Exception("Empty file content")
                
            file_ext = ".mp4"
            if file_name and '.' in file_name:
                file_ext = file_name[file_name.rindex('.'):]
            
            temp_file_path = None
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(file_content.encode() if isinstance(file_content, str) else file_content)
                    temp_file_path = temp_file.name
                
                # Build headers & data depending on provider
                if is_azure:
                    # Choose appropriate key for whisper vs gpt-4o
                    use_key = azure_api_key
                    if model == "whisper-1" and azure_api_key_whisper:
                        use_key = azure_api_key_whisper
                    headers = {"api-key": use_key}
                    request_data = {"response_format": response_format}
                else:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    request_data = {"model": model, "response_format": response_format}
                
                if prompt:
                    request_data["prompt"] = prompt
                
                if language:
                    request_data["language"] = language
                
                # Whisper-only timestamp granularities (works for OpenAI and Azure Whisper deployments)
                if timestamp_granularities != "none" and (
                    (not is_azure and model == "whisper-1") or (is_azure and transcription_type != "translate")
                ):
                    # For Whisper timestamps, response_format must be verbose_json
                    request_data["response_format"] = "verbose_json"
                    if timestamp_granularities == "segment":
                        request_data["timestamp_granularities"] = ["segment"]
                    elif timestamp_granularities == "word":
                        request_data["timestamp_granularities"] = ["word"]
                    elif timestamp_granularities == "segment_and_word":
                        request_data["timestamp_granularities"] = ["segment", "word"]
                
                if stream:
                    request_data["stream"] = True
                
                # Helper to post with possible Azure fallback on 404 Resource not found
                def _post_with_optional_fallback(url: str) -> requests.Response:
                    with open(temp_file_path, "rb") as f:
                        files = {"file": (file_name, f, file_type)}
                        resp = requests.post(url, headers=headers, data=request_data, files=files, timeout=HTTP_TIMEOUT, stream=stream)
                    if is_azure and resp.status_code == 404:
                        # Try fallback versions for Transcribe when Azure returns 404, regardless of initial version
                        if transcription_type != "translate":
                            fallback_candidates = [
                                "2024-02-15-preview",
                                "2024-12-01-preview",
                            ]
                            for fv in fallback_candidates:
                                if fv == azure_api_version:
                                    continue
                                fallback_url = _build_azure_url(path_kind, version_override=fv)
                                with open(temp_file_path, "rb") as f2:
                                    files2 = {"file": (file_name, f2, file_type)}
                                    r2 = requests.post(fallback_url, headers=headers, data=request_data, files=files2, timeout=HTTP_TIMEOUT, stream=stream)
                                    if r2.status_code == 200:
                                        resp = r2
                                        break
                    return resp
                
                # Execute request
                response = _post_with_optional_fallback(api_endpoint)
                
                if response.status_code != 200:
                    # Improve error reporting
                    err_text = response.text
                    try:
                        j = response.json()
                        msg = j.get("error", {}).get("message") or j.get("message")
                        if msg:
                            err_text = msg
                    except Exception:
                        pass
                    raise Exception(f"Error {response.status_code}: {err_text}")
                    
                if stream:
                    buffer = ""
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                data = line_text[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(data)
                                    if 'type' in json_data and json_data['type'] == 'transcript.text.delta':
                                        if 'delta' in json_data:
                                            text_chunk = json_data['delta']
                                            buffer += text_chunk
                                            yield self.create_text_message(text_chunk)
                                    elif 'type' in json_data and json_data['type'] == 'transcript.text.done':
                                        if 'text' in json_data:
                                            buffer = json_data['text']
                                            yield self.create_text_message(buffer)
                                    elif 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        if 'text' in delta:
                                            text_chunk = delta['text']
                                            buffer += text_chunk
                                            yield self.create_text_message(text_chunk)
                                except json.JSONDecodeError:
                                    pass
                    if buffer and output_format in ["default", "json_only"]:
                        yield self.create_json_message({"result": {"text": buffer}})
                else:
                    if response_format in ["json", "verbose_json"]:
                        result = response.json()
                    else:
                        # Try to parse JSON for 'text' even when response_format==text (translations return JSON)
                        try:
                            j = response.json()
                            if isinstance(j, dict) and "text" in j:
                                result = {"text": j["text"]}
                            else:
                                result = j
                        except Exception:
                            result = {"text": response.text}

                    # Azure Whisper translation fallback: if translate returns non-English text, try transcriptions with translate=true
                    def _looks_non_english(txt: str) -> bool:
                        if not txt:
                            return False
                        total = len(txt)
                        non_ascii = sum(1 for ch in txt if ord(ch) > 127)
                        # If more than 20% of chars are non-ASCII, likely not English
                        return (non_ascii / max(total, 1)) > 0.2

                    if transcription_type == "translate" and is_azure:
                        # Extract text from result to assess language
                        text_out = result.get("text") if isinstance(result, dict) else str(result)
                        if _looks_non_english(str(text_out)):
                            # Fallback to transcriptions with translate flag
                            fallback_url = _build_azure_url("transcriptions")
                            request_data_fallback = dict(request_data)
                            request_data_fallback.pop("stream", None)
                            request_data_fallback["translate"] = True
                            with open(temp_file_path, "rb") as f3:
                                files3 = {"file": (file_name, f3, file_type)}
                                r3 = requests.post(
                                    fallback_url,
                                    headers=headers,
                                    data=request_data_fallback,
                                    files=files3,
                                    timeout=HTTP_TIMEOUT,
                                    stream=False,
                                )
                            if r3.status_code == 200:
                                try:
                                    j2 = r3.json()
                                    if isinstance(j2, dict) and "text" in j2:
                                        result = {"text": j2["text"]}
                                    else:
                                        result = j2
                                except Exception:
                                    result = {"text": r3.text}

                    if output_format == "json_only":
                        yield self.create_json_message({"result": result})
                    elif output_format == "text_only":
                        if isinstance(result, dict) and "text" in result:
                            yield self.create_text_message(result["text"])
                        else:
                            yield self.create_text_message(str(result))
                    else:
                        yield self.create_json_message({"result": result})
                        if isinstance(result, dict) and "text" in result:
                            yield self.create_text_message(result["text"])
                        else:
                            yield self.create_text_message(str(result))
            finally:
                if temp_file_path:
                    try:
                        pathlib.Path(temp_file_path).unlink()
                    except Exception:
                        pass
            
        except Exception as e:
            raise Exception(f"Exception while processing audio: {str(e)}")
