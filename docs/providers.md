# ONNX providers

OnnxVoice owns ONNX Runtime loading and provider selection. PiperSynth imports no ONNX Runtime module directly. Runtime dependencies are loaded only when an OnnxVoice runtime is opened or providers are queried.

The default provider remains exactly `CPUExecutionProvider`. Passing `providers` preserves the requested order. `ProviderConfig` remains a PiperSynth compatibility type and is translated to OnnxVoice provider and provider-options arguments. A requested unavailable provider raises an actionable error instead of silently selecting another provider.

Use `available_providers()` to inspect the installed runtime. For deterministic tests, inject a runtime with an `infer()` method or use `PiperVoice.load(..., session_factory=...)` as a compatibility seam. Neither approach requires a real model runtime.

Install one runtime extra:

```bash
pip install "pipersynth[cpu]"
pip install "pipersynth[gpu]"
```

The catalog and managed voice store are provided by OnnxVoice. PiperSynth does not include `piper-tts` or the former Piper-specific catalog package.
