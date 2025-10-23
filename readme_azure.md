# Azure OpenAI Technical Guide for OpenAI Audio STT Plugin

This document explains Azure-specific behaviors, configuration nuances, and the runtime/provider fallbacks implemented in this plugin. Azure OpenAI sometimes exposes different API-version support between control-plane endpoints (e.g., deployments listing) and data-plane endpoints (e.g., audio transcriptions). These differences can be confusing when configuring and running the plugin in Dify.

## Overview

This plugin supports two Azure OpenAI resources that you may configure separately:
- GPT‑4o Transcribe resource (high-quality transcription; streaming supported)
- Whisper resource (legacy transcription; advanced formats like verbose_json/timestamps; translation to English)

Key behaviors:
- Transcribe (GPT‑4o): streaming enabled; formats limited to text/json
- Transcribe (Whisper): streaming disabled; verbose_json/srt/vtt/timestamps supported
- Translate (Whisper only): returns English; streaming disabled; advanced formats not supported

## Azure API version inconsistencies

In practice, Azure OpenAI endpoints may behave differently depending on the API version.
- Deployment listing (`GET {endpoint}/openai/deployments?api-version=...`) may only work with older versions (e.g., `2023-03-15-preview`) on some resources.
- Audio data-plane endpoints may require newer versions (e.g., `2024-02-15-preview` or `2024-12-01-preview` for GPT‑4o, and `2024-02-01` for Whisper).

### Dify UI validation vs runtime

- Dify’s Plugin configuration UI validates credentials by calling the deployments listing endpoint. Some resources only accept `2023-03-15-preview` here. As a result, you may need to set:
  - Azure Transcribe API Version = `2023-03-15-preview` to save the configuration.
- However, invoking audio transcriptions with that version often fails with 404 in runtime.

To bridge this:
- Provider validation (listing) fallbacks:
  - Whisper listing: If `2024-02-01` returns 404, the plugin will try `2024-02-15-preview` and then `2023-03-15-preview` to allow saving.
- Runtime (audio) fallbacks:
  - GPT‑4o Transcribe: If a transcriptions call returns 404, the plugin automatically retries with `2024-02-15-preview` and `2024-12-01-preview` (first 200 wins).

Recommendation:
- If your resource supports newer versions, set Azure Transcribe API Version to `2024-02-15-preview` (or `2024-12-01-preview`) directly in Dify to avoid runtime fallbacks.
- For Whisper, use `2024-02-01`.

## Resource selection and routing

We separate credentials for Transcribe and Whisper resources:
- Transcribe:
  - `azure_endpoint_transcribe`, `azure_api_key_transcribe`, `azure_api_version_transcribe`, `azure_deployment_transcribe`
- Whisper:
  - `azure_endpoint_whisper`, `azure_api_key_whisper`, `azure_api_version_whisper` (default `2024-02-01`), `azure_deployment_whisper`

Routing rules:
- If `transcription_type=translate`:
  - Force `model=whisper-1`
  - Use Whisper resource endpoint/key/version
  - Require a Whisper deployment name
- If `transcription_type=transcribe`:
  - `model=gpt-4o-transcribe` → Transcribe resource
  - `model=whisper-1` → Whisper resource

Notes:
- Endpoints are normalized (strip trailing slashes; collapse duplicate `//`).
- You can set a per-call override via tool parameter `azure_deployment`.

## Formats, streaming, and timestamps

- GPT‑4o Transcribe:
  - Streaming supported (`stream=true`)
  - Formats: `text` or `json` (advanced formats disabled automatically)
- Whisper Transcribe:
  - Streaming disabled
  - Formats: `text`, `json`, `verbose_json`, `srt`, `vtt`
  - Timestamps: `segment`, `word`, `segment_and_word` (forces `verbose_json`)
- Whisper Translate:
  - Streaming disabled
  - Returns English
  - Advanced formats/timestamps not available on translations endpoint

## Translation fallback (Whisper)

Some Whisper resources may return same-language text from the translations endpoint. The plugin implements a compatibility fallback:
- If `transcription_type=translate` and the first response looks non-English (e.g., >20% characters are non-ASCII), the plugin retries using the Whisper transcriptions endpoint with `translate=true`.
- Caveats:
  - This fallback is heuristic and may not trigger in all scenarios.
  - The fallback does not enable advanced formats/timestamps—it remains a translation flow.

If translation remains non-English for your resource, consider:
- Verifying your Whisper resource API version (`2024-02-01` recommended)
- Testing with `curl` to both translations and transcriptions endpoints
- Optionally forcing the fallback always (we can add a provider flag if needed)

## Error handling and diagnostics

- On non-200 responses, the plugin surfaces status code and message from Azure/OpenAI.
- For Azure 404s during transcribe, runtime fallbacks are attempted automatically.
- Common 404 causes:
  - API-version mismatch (listing vs audio)
  - Deployment name mismatch (use exact Azure deployment name, not model name)
  - Translate called on GPT‑4o deployment (must be Whisper)

### Curl commands for validation

- List deployments (Transcribe):
  ```bash
  curl -H "api-key: $AZURE_API_KEY" \
       "$AZURE_ENDPOINT/openai/deployments?api-version=2024-12-01-preview"
  # If 404, try:
  curl -H "api-key: $AZURE_API_KEY" \
       "$AZURE_ENDPOINT/openai/deployments?api-version=2024-02-15-preview"
  curl -H "api-key: $AZURE_API_KEY" \
       "$AZURE_ENDPOINT/openai/deployments?api-version=2023-03-15-preview"
  ```

- GPT‑4o Transcribe (data-plane):
  ```bash
  curl -X POST \
    -H "api-key: $AZURE_API_KEY" \
    -F "file=@sample.mp3" \
    -F "response_format=text" \
    "$AZURE_ENDPOINT/openai/deployments/$AZURE_DEPLOYMENT/audio/transcriptions?api-version=2024-02-15-preview"
  ```

- Whisper Transcribe (verbose_json + timestamps):
  ```bash
  curl -X POST \
    -H "api-key: $AZURE_WHISPER_API_KEY" \
    -F "file=@sample.mp3" \
    -F "response_format=verbose_json" \
    -F "timestamp_granularities=segment" \
    "$AZURE_WHISPER_ENDPOINT/openai/deployments/$AZURE_WHISPER_DEPLOYMENT/audio/transcriptions?api-version=2024-02-01"
  ```

- Whisper Translate (endpoint):
  ```bash
  curl -X POST \
    -H "api-key: $AZURE_WHISPER_API_KEY" \
    -F "file=@sample.mp3" \
    "$AZURE_WHISPER_ENDPOINT/openai/deployments/$AZURE_WHISPER_DEPLOYMENT/audio/translations?api-version=2024-02-01"
  ```

- Whisper Translate fallback (via transcriptions with translate=true):
  ```bash
  curl -X POST \
    -H "api-key: $AZURE_WHISPER_API_KEY" \
    -F "file=@sample.mp3" \
    -F "translate=true" \
    "$AZURE_WHISPER_ENDPOINT/openai/deployments/$AZURE_WHISPER_DEPLOYMENT/audio/transcriptions?api-version=2024-02-01"
  ```

## Dify configuration tips

- If the UI only saves with `2023-03-15-preview` for Transcribe API Version, that’s OK. The runtime will auto-retry with newer versions on 404.
- Ensure deployment names match exactly those in Azure.
- Separate resources are recommended: one for GPT‑4o Transcribe and one for Whisper.
- Whisper translate requires the Whisper resource to be configured.

## Reliability & security

- Timeouts: requests use connect/read timeouts (default 10s connect, 120s read; configurable via `MAX_REQUEST_TIMEOUT`).
- File size: basic guard for 25MB via Content-Length if available; consider adjusting in your environment.
- Temporary files: stored in the OS temp directory; removed in a `finally` block.
- Potential SSRF: if untrusted URLs were ever passed, consider restricting to Dify’s file service domains; the plugin currently uses timeouts and size caps, but not host allow-lists.

## Known limitations

- Translation fallback is heuristic and may not work for all languages/resources.
- Advanced formats/timestamps are not available on the translations endpoint.
- Some Azure resources may continue to require older API versions for listing even when audio endpoints support newer versions.

## Troubleshooting checklist

1. Verify deployments via `curl` for both listing and data-plane audio endpoints.
2. Confirm exact deployment names (case-sensitive).
3. For GPT‑4o Transcribe, prefer `2024-02-15-preview` in runtime.
4. For Whisper, use `2024-02-01`.
5. If Dify UI rejects newer versions, save with `2023-03-15-preview` and rely on runtime fallback.
6. Use non-English audio when testing translate; English input will return same text.
7. If translate still returns non-English, consider enabling a provider flag to always use transcriptions+translate fallback (contact maintainer).

---

If you run into Azure-specific errors that aren’t addressed here, please open an issue with: endpoint, api-version, deployment name (redact keys), and the exact error message. We’ll expand the fallback matrix as needed.
