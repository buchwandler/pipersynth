from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """One ONNX Runtime execution provider and its options."""

    name: str
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider name must not be empty")
        object.__setattr__(self, "options", dict(self.options or {}))


def available_providers() -> tuple[str, ...]:
    """Return providers exposed by the installed ONNX Runtime package."""

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "ONNX Runtime is required for acoustic inference. Install pipersynth[cpu] "
            "or pipersynth[gpu]."
        ) from exc
    return tuple(ort.get_available_providers())


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


class OnnxSessionManager:
    """Own creation, inspection, and release of one ONNX Runtime session."""

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
        self._session: InferenceSession | None = None
        self._model_inputs: tuple[str, ...] = ()
        self._model_outputs: tuple[str, ...] = ()
        self._providers_active: tuple[str, ...] = ()

    @property
    def session(self) -> InferenceSession:
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

    def create(self, *, require_sid: bool = False) -> InferenceSession:
        if self._session is not None:
            self.validate_contract(require_sid=require_sid)
            return self._session
        if not self.model_path.exists():
            raise ModelFileNotFoundError(f"ONNX model file does not exist: {self.model_path}")
        factory = self.session_factory
        if factory is None:
            try:
                import onnxruntime as ort
            except ModuleNotFoundError as exc:
                raise OptionalDependencyError(
                    "ONNX Runtime is required for acoustic inference. Install pipersynth[cpu] "
                    "or pipersynth[gpu]."
                ) from exc
            available = tuple(ort.get_available_providers())
            unavailable = [name for name in self.providers_requested if name not in available]
            if unavailable:
                raise SessionCreationError(
                    f"Requested ONNX Runtime provider(s) are unavailable: {', '.join(unavailable)}; "
                    f"available providers: {', '.join(available)}"
                )
            factory = ort.InferenceSession
        kwargs: dict[str, Any] = {"providers": list(self.providers_requested)}
        options = [dict(provider.options or {}) for provider in self.provider_configs]
        if any(options):
            kwargs["provider_options"] = options
        if self.session_options is not None:
            kwargs["sess_options"] = self.session_options
        try:
            self._session = factory(str(self.model_path), **kwargs)
        except Exception as exc:
            raise SessionCreationError(f"Could not create ONNX session for {self.model_path}") from exc
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
        self._session = None
        self._model_inputs = ()
        self._model_outputs = ()
        self._providers_active = ()
