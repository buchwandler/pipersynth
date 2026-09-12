# ONNX providers

ONNX Runtime is optional and imported only when a session is created or providers are queried.

The default provider is exactly `CPUExecutionProvider`. Passing `providers` preserves the requested order. Provider options can be supplied through `PipelineConfig.provider_options` or `PiperVoice.load(provider_options=...)`. A requested unavailable provider raises an actionable error instead of silently selecting another provider.

Use `available_providers()` to inspect the installed runtime. For tests, inject a `session_factory`; this does not require ONNX Runtime.

CPU and GPU extras are mutually exclusive installation choices. PiperSynth does not include `piper-tts`.
