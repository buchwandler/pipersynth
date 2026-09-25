# ONNX providers

OnnxVoice owns ONNX Runtime loading and provider selection. PiperSynth does not import ONNX Runtime directly. Providers are opened when a `PiperVoice` is loaded or when `available_providers()` is queried.

The default execution provider is `CPUExecutionProvider`. Pass an ordered provider sequence to either local or catalog voice loading:

```python
from pipersynth import PiperVoice, ProviderConfig

providers = (
    ProviderConfig("CUDAExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider",
)
with PiperVoice.from_pretrained(
    "en_US-lessac-medium",
    providers=providers,
) as voice:
    result = voice.synthesize_text("Prepared speech.", language="en-us")
```

`provider_options` applies common options to provider names that do not provide their own options:

```python
with PiperVoice.from_local(
    "voice.onnx",
    providers=("CUDAExecutionProvider",),
    provider_options={"device_id": 0},
) as voice:
    ...
```

Inspect installed runtime providers with `available_providers()`. Install `pipersynth[cpu]` or `pipersynth[gpu]` to provide an ONNX Runtime backend. An unavailable requested provider raises an error; PiperSynth does not silently select a different provider.
