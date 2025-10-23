# Changelog

## Unreleased

### Added
- Azure-specific technical guide (readme_azure.md) detailing API-version differences, fallbacks, and routing behaviors.
- CI workflow (GitHub Actions) to run pytest on Python 3.10.
- pytest tests:
  - Azure GPT‑4o transcribe: 404 runtime fallback on api-version, streaming SSE parsing.
  - Azure Whisper: verbose_json + timestamps, translate fallback via transcriptions translate=true.

### Fixed
- Whisper transcribe routing: when model==whisper-1, switch to Whisper endpoint/key/version, enabling verbose_json/timestamps.
- Dify tool YAML file parameter: use `form: form` to allow user uploads.
- Runtime resilience: Azure transcribe 404 fallback tries supported api-versions automatically.
- Provider validation: Whisper deployments listing fallback across 2024-02-01, 2024-02-15-preview, 2023-03-15-preview.
- Temp file handling: context-managed file handles and cleanup via finally; added HTTP timeouts.

### Known Issues
- Whisper translation may still return source language on some resources; a heuristic fallback retries via transcriptions with translate=true but may not cover all cases.
- SSRF protections for file download are limited to timeouts and size guards; host allow-list not yet implemented.
