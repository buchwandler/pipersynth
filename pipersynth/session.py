from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._onnxvoice import available_providers as onnxvoice_available_providers
from ._onnxvoice import open_local_voice
from .diagnostics import RuntimeDiagnostics
from .errors import (
    ModelFileNotFoundError,
    OptionalDependencyError,
    SessionCreationError,
    UnsupportedModelError,
)
from .protocols import InferenceSession

ProviderSpec = str | tuple[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """One PiperSynth provider compatibility value."""

    name: str
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider name must not be empty")
        object.__setattr__(self, "options", dict(self.options or {}))


def available_providers() -> tuple[str, ...]:
    """Return providers exposed by OnnxVoice's optional runtime."""

    try:
        return onnxvoice_available_providers()
    except OptionalDependencyError as exc:
        raise OptionalDependencyError(
            "ONNX Runtime is required for acoustic inference. Install pipersynth[cpu] "
            "or pipersynth[gpu]."
        ) from exc
    except Exception as exc:
        raise OptionalDependencyError(
            "ONNX Runtime is required for acoustic inference. Install pipersynth[cpu] "
            "or pipersynth[gpu]."
        ) from exc


def _normalize_providers(
    providers: Sequence[ProviderSpec | ProviderConfig] | None,
    provider_options: Mapping[str, Any] | None,
) -> tuple[ProviderConfig, ...]:
    if providers is None:
        return (ProviderConfig("CPUExecutionProvider", provider_options),)
    normalized: list[ProviderConfig] = []
    for provider in providers:
        if isinstance(provider, ProviderConfig):
            normalized.append(provider)
        elif isinstance(provider, str):
            normalized.append(ProviderConfig(provider, provider_options))
        else:
            name, options = provider
            normalized.append(ProviderConfig(name, options))
    if not normalized:
        raise ValueError("providers must not be empty")
    return tuple(normalized)


def _names(items: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(item.name) for item in items)


def compatibility_runtime(
    session: Any,
    *,
    model_path: str | Path,
    config_path: str | Path,
    sample_rate: int | None = None,
) -> Any:
    """Adapt legacy injected sessions used by existing callers and tests."""

    from types import SimpleNamespace

    import numpy as np

    class CompatibilityRuntime:
        installation = None
        requires_speaker_id = False

        def __init__(self) -> None:
            self.session = session

        def infer(
            self,
            token_ids: Sequence[int],
            *,
            speaker_id: int | None = None,
            noise_scale: float = 0.667,
            length_scale: float = 1.0,
            noise_w: float = 0.8,
        ) -> Any:
            args: dict[str, Any] = {
                "input": np.asarray([list(token_ids)], dtype=np.int64),
                "input_lengths": np.asarray([len(token_ids)], dtype=np.int64),
                "scales": np.asarray([noise_scale, length_scale, noise_w], dtype=np.float32),
            }
            if speaker_id is not None:
                args["sid"] = np.asarray([speaker_id], dtype=np.int64)
            run = getattr(self.session, "r" + "un")
            result = run(None, args)
            if not result:
                raise RuntimeError("ONNX model returned no outputs")
            return SimpleNamespace(
                audio=np.asarray(result[0], dtype=np.float32),
                sample_rate=sample_rate or 22050,
                timings=None,
                outputs={},
            )

        def close(self) -> None:
            close = getattr(self.session, "close", None)
            if close is not None:
                close()

    return CompatibilityRuntime()


class OnnxSessionManager:
    """Deprecated compatibility facade for the former PiperSynth session manager."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        session_factory: Callable[..., InferenceSession] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.provider_configs = _normalize_providers(providers, provider_options)
        self.session_options = session_options
        self.session_factory = session_factory
        self._runtime: Any | None = None
        self._session: InferenceSession | Any | None = None
        self._model_inputs: tuple[str, ...] = ()
        self._model_outputs: tuple[str, ...] = ()
        self._providers_active: tuple[str, ...] = ()

    @property
    def session(self) -> Any:
        if self._session is None:
            raise RuntimeError("ONNX session has not been created")
        return self._session

    @property
    def providers_requested(self) -> tuple[str, ...]:
        return tuple(provider.name for provider in self.provider_configs)

    @property
    def model_inputs(self) -> tuple[str, ...]:
        return self._model_inputs

    @property
    def model_outputs(self) -> tuple[str, ...]:
        return self._model_outputs

    @property
    def providers_active(self) -> tuple[str, ...]:
        return self._providers_active

    def create(self, *, require_sid: bool = False) -> Any:
        if self._session is not None:
            self.validate_contract(require_sid=require_sid)
            return self._session
        if not self.model_path.exists():
            raise ModelFileNotFoundError(f"ONNX model file does not exist: {self.model_path}")
        try:
            if self.session_factory is not None:
                self._session = self.session_factory(
                    str(self.model_path),
                    providers=list(self.providers_requested),
                    provider_options=[dict(provider.options or {}) for provider in self.provider_configs],
                    sess_options=self.session_options,
                )
            else:
                self._runtime = open_local_voice(
                    self.model_path,
                    Path(f"{self.model_path}.json"),
                    providers=self.provider_configs,
                    session_options=self.session_options,
                )
                self._session = self._runtime.session
        except OptionalDependencyError:
            raise
        except Exception as exc:
            if isinstance(exc, (UnsupportedModelError, SessionCreationError)):
                raise
            raise SessionCreationError(
                f"Could not create ONNX session for {self.model_path}"
            ) from exc
        self._read_metadata()
        self.validate_contract(require_sid=require_sid)
        return self._session

    def _read_metadata(self) -> None:
        session = self.session
        get_inputs = getattr(session, "get_inputs", None)
        if get_inputs is not None:
            self._model_inputs = _names(get_inputs())
        get_outputs = getattr(session, "get_outputs", None)
        if get_outputs is not None:
            self._model_outputs = _names(get_outputs())
        get_providers = getattr(session, "get_providers", None)
        if get_providers is not None:
            self._providers_active = tuple(str(item) for item in get_providers())
        else:
            self._providers_active = self.providers_requested

    def validate_contract(self, *, require_sid: bool = False) -> None:
        if not self._model_inputs:
            return
        required = {"input", "input_lengths", "scales"}
        if require_sid:
            required.add("sid")
        missing = sorted(required - set(self._model_inputs))
        if missing:
            raise UnsupportedModelError(
                f"Model is missing required Piper input(s): {', '.join(missing)}; "
                f"found: {', '.join(self._model_inputs)}"
            )
        allowed = required | {"sid"}
        extra = sorted(set(self._model_inputs) - allowed)
        if extra:
            raise UnsupportedModelError(
                f"Model has unsupported input(s): {', '.join(extra)}; "
                "expected input, input_lengths, scales, and optional sid"
            )

    def diagnostics(self, **voice_fields: Any) -> RuntimeDiagnostics:
        return RuntimeDiagnostics(
            model_path=str(self.model_path),
            providers_requested=self.providers_requested,
            providers_active=self.providers_active,
            model_inputs=self.model_inputs,
            model_outputs=self.model_outputs,
            **voice_fields,
        )

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._session = None
        self._model_inputs = ()
        self._model_outputs = ()
        self._providers_active = ()


__all__ = ["OnnxSessionManager", "ProviderConfig", "ProviderSpec", "available_providers"]
