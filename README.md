## OpenAI Audio STT

**Author:** lysonober
**Version:** 0.0.4
**Type:** Tool

### Description

The OpenAI Audio tool is a powerful speech-to-text conversion solution that leverages OpenAI's Audio API to transform audio content into accurate text transcriptions and translations. This tool supports multiple audio formats (mp3, mp4, mpeg, mpga, m4a, wav, webm) and can process files up to 25MB in size. It offers both transcription (keeping the original language) and translation (converting to English) capabilities across a wide range of languages.

The tool integrates three powerful models: GPT-4o Transcribe for high-quality transcription, GPT-4o Mini Transcribe for faster processing, and Whisper-1 for legacy support with additional formatting options. Advanced features include streaming output for real-time transcription with GPT-4o models, timestamp generation at segment or word level with Whisper-1, and multiple output formats including plain text, JSON, SRT, and VTT subtitles.

### Use Cases

1️⃣ Today, for **content creators and video producers**,
2️⃣ when **working with hours of interview footage or multilingual content**,
3️⃣ they are forced to **spend excessive time manually transcribing audio or hiring expensive transcription services**,
4️⃣ therefore, the customer needs a way to **quickly and accurately convert speech to text while preserving timestamps and supporting multiple languages**.

---

1️⃣ Today, for **accessibility specialists and educational institutions**,
2️⃣ when **creating accessible content for diverse audiences with hearing impairments**,
3️⃣ they are forced to **navigate complex subtitle creation tools or outsource caption generation**,
4️⃣ therefore, the customer needs a way to **efficiently generate accurate subtitles in various formats (SRT, VTT) with precise timestamps**.

### Parameters

| Parameter               | Type    | Required | Description                                                                                                                                                                                                                      |
| ----------------------- | ------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| file                    | file    | Yes      | The audio file to transcribe. Supports mp3, mp4, mpeg, mpga, m4a, wav, and webm formats with a maximum size of 25MB.                                                                                                             |
| transcription_type      | select  | No       | Determines whether to transcribe the audio in its original language ("transcribe") or translate it to English ("translate"). Note that translation is only available with the Whisper-1 model and will disable streaming output. |
| model                   | select  | No       | The AI model to use for processing. Options include GPT-4o Transcribe (high quality), GPT-4o Mini Transcribe (faster), and Whisper-1 (legacy with more format options). Default is GPT-4o Transcribe.                            |
| response_format         | select  | No       | The format of the transcript output. Options include text, JSON, verbose JSON (Whisper-1 only), SRT subtitles (Whisper-1 only), and VTT subtitles (Whisper-1 only). Default is text.                                             |
| prompt                  | string  | No       | Optional guidance for the model's transcription. Useful for improving accuracy with uncommon words, acronyms, or specific terminology by providing context.                                                                      |
| language                | string  | No       | ISO-639-1 language code (e.g., 'en', 'zh', 'ja') to help improve accuracy if the audio language is known. This helps the model focus on the specific language patterns.                                                          |
| timestamp_granularities | select  | No       | Adds timestamps to the transcript at segment or word level. Only available with the Whisper-1 model and requires verbose_json response format. Options are none, segment, or word.                                               |
| stream                  | boolean | No       | Enables streaming output where transcription results are delivered as they're generated. This feature is only available with GPT-4o Transcribe and GPT-4o Mini Transcribe models. Default is true.                               |
| output_format           | select  | No       | Controls how the plugin formats its output in Dify. Options include Default (JSON + Text), JSON Only, or Text Only. This affects how the results are presented to the user in the interface.                                     |

### Parameter Interactions: What Happens When You Change Settings

The OpenAI Audio tool has several settings that affect each other. Understanding these relationships will help you get the results you want:

#### Automatic Setting Adjustments

1. **When You Choose Translation Mode**

   - **What happens:** If you select "Translate to English" as your transcription type, the tool will automatically switch to the Whisper-1 model, even if you selected a different model.
2. **When You Choose Special Output Formats**

   - **What happens:** If you select formats like "Verbose JSON", "SRT Subtitles", or "VTT Subtitles" but are using a GPT-4o model, the tool will automatically switch back to plain text format.
   - **Why it's designed this way:** The newer GPT-4o models focus on speed and accuracy for basic transcription, while Whisper-1 offers more formatting options.
3. **When You Request Timestamps**

   - **What happens:** If you ask for word or segment timestamps but are using a GPT-4o model, the timestamp feature will be turned off.
4. **When You Enable Streaming**

   - **What happens:** If you turn on streaming (getting results in real-time) but are using the Whisper-1 model, streaming will be automatically disabled.
5. **When You Enable Timestamps with Whisper-1**

   - **What happens:** If you turn on timestamps and are using the Whisper-1 model, the output format will automatically switch to "Verbose JSON".
   - **Why it's designed this way:** Timestamps contain extra information that doesn't fit in simple text formats, so the tool uses a format that can include all the details.

#### Summary of Automatic Adjustments

| If you set                                | And              | Then automatically                | Reason                                    |
| ----------------------------------------- | ---------------- | --------------------------------- | ----------------------------------------- |
| `transcription_type: translate`         | any model        | `model: whisper-1`              | Translation only works with Whisper-1     |
| `response_format: verbose_json/srt/vtt` | not Whisper-1    | `response_format: text`         | Advanced formats only work with Whisper-1 |
| `timestamp_granularities: segment/word` | not Whisper-1    | `timestamp_granularities: none` | Timestamps only work with Whisper-1       |
| `timestamp_granularities: segment/word` | Whisper-1        | `response_format: verbose_json` | Timestamps require verbose JSON format    |
| `stream: true`                          | not GPT-4o model | `stream: false`                 | Streaming only works with GPT-4o models   |

### Technical Details

The OpenAI Audio tool communicates with OpenAI's Audio API endpoints:

- Transcription: `https://api.openai.com/v1/audio/transcriptions`
- Translation: `https://api.openai.com/v1/audio/translations`

The tool handles various file input methods, creates temporary files for processing, and manages the API communication including streaming responses. It automatically applies appropriate parameter validation and model compatibility checks to ensure optimal results.

### Output Examples

**Text Output:**

```
Hello, this is a sample transcription of spoken audio content that demonstrates the accuracy of the OpenAI Audio tool.
```

**JSON Output (simplified):**

```json
{
  "result": {
    "text": "Hello, this is a sample transcription of spoken audio content that demonstrates the accuracy of the OpenAI Audio tool."
  }
}
```

### Support

If you have any questions, please contact me at: lysonober@gmail.com

Follow me on X (Twitter): https://x.com/lyson_ober

### Privacy

Please refer to the [PRIVACY.md](./PRIVACY.md) file for information about how your data is handled when using this plugin. This plugin does not collect any data directly, but your audio is processed through OpenAI's services subject to their privacy policies.

## Azure OpenAI Support (GPT-4o Transcribe + Whisper)

This plugin now supports Azure OpenAI deployments for GPT-4o Transcribe and Whisper. Set provider credentials accordingly:

- `azure_endpoint`: e.g., `https://<your-resource>.openai.azure.com`
- `azure_api_key`: Azure OpenAI API key (or reuse `api_key`)
- `azure_deployment`: Your deployment name in Azure (e.g., `gpt-4o-transcribe` or `whisper-1`)
- `azure_api_version`: default `2024-12-01-preview`

When Azure credentials are present, the plugin automatically uses Azure endpoints.

### Curl example (Azure GPT-4o Transcribe)

```
curl -X POST \
  -H "api-key: $AZURE_API_KEY" \
  -F "file=@sample.mp3" \
  -F "response_format=text" \
  "$AZURE_ENDPOINT/openai/deployments/$AZURE_DEPLOYMENT/audio/transcriptions?api-version=2024-12-01-preview"
```

### Whisper on Azure
- Transcribe supports `verbose_json`, `srt`, `vtt` and timestamp granularities (segment/word).
- Translate uses the translations endpoint via the Whisper deployment.

### Streaming
- GPT-4o Transcribe supports streaming via SSE. Whisper translate does not stream.

### Local Testing Outside Dify
See `scripts/test_harness.py` for a quick way to call the tool directly.
