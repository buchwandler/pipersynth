from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .asset_progress import AssetProgressEvent, AssetProgressPhase
from .errors import (
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    CatalogUnavailableError,
    ModelInferenceError,
    OptionalDependencyError,
    SessionCreationError,
    UnsupportedModelError,
    VoiceNotFoundError,
)


@dataclass(frozen=True, slots=True)
class ResolvedPiperVoice:
    """PiperSynth's read-only view of an OnnxVoice installation."""

    ref: str | None
    voice_id: str | None
    model_path: Path
    config_path: Path
    model_card_path: Path | None
    language_code: str | None
    source_revision: str | None
    metadata: Mapping[str, Any]
    installation: Any | None = None


def _onnxvoice() -> Any:
    try:
        import onnxvoice
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "OnnxVoice is required for Piper model loading. Install pipersynth[cpu] or "
            "pipersynth[gpu]."
        ) from exc
    return onnxvoice


def normalize_piper_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref.strip():
        raise VoiceNotFoundError("Piper voice reference must not be empty")
    value = ref.strip()
    if ":" not in value:
        return f"piper:{value}"
    system, item_id = value.split(":", 1)
    if system.casefold() != "piper" or not item_id:
        raise VoiceNotFoundError(f"Not a Piper voice reference: {ref!r}")
    return f"piper:{item_id}"


def normalize_provider_request(
    providers: Sequence[Any] | str | None,
    provider_options: Mapping[str, Any] | None,
) -> tuple[str | Sequence[str] | None, Any | None]:
    """Translate PiperSynth provider compatibility values to OnnxVoice values."""

    if providers is None:
        return "CPUExecutionProvider", provider_options
    if isinstance(providers, str):
        return providers, provider_options

    names: list[str] = []
    options: list[dict[str, Any]] = []
    has_explicit_options = False
    for provider in providers:
        if hasattr(provider, "name"):
            name = str(provider.name)
            value = dict(getattr(provider, "options", {}) or {})
            has_explicit_options = has_explicit_options or bool(value)
        elif isinstance(provider, tuple):
            name, raw_options = provider
            value = dict(raw_options or {})
            has_explicit_options = True
        else:
            name = str(provider)
            value = dict(provider_options or {})
        names.append(name)
        options.append(value)
    if not names:
        raise ValueError("providers must not be empty")
    return names, options if has_explicit_options else provider_options


def adapt_progress(
    callback: Callable[[AssetProgressEvent], None] | None,
) -> Callable[[Any], None] | None:
    if callback is None:
        return None

    def emit(event: Any) -> None:
        phase = str(getattr(event, "phase", "asset"))
        phase_map = {
            "catalog_started": "catalog-refresh",
            "catalog_cached": "catalog-load",
            "catalog_completed": "catalog-load",
            "install_started": "voice-resolve",
            "install_completed": "download-complete",
            "download_started": "download-start",
            "download_progress": "download-start",
            "download_completed": "download-complete",
            "artifact_cached": "cache-hit",
        }
        raw_ref = getattr(event, "ref", None)
        voice_id = (
            raw_ref.split(":", 1)[1] if isinstance(raw_ref, str) and ":" in raw_ref else raw_ref
        )
        callback(
            AssetProgressEvent(
                cast(AssetProgressPhase, phase_map.get(phase, "voice-resolve")),
                voice_id=voice_id,
                message=getattr(event, "message", None),
            )
        )

    return emit


def _map_error(exc: Exception, *, operation: str) -> Exception:
    name = type(exc).__name__
    if name in {"OptionalDependencyError", "ImportError"}:
        return OptionalDependencyError(str(exc))
    if name in {"AssetNotFoundError", "NotInstalledError"}:
        return VoiceNotFoundError(str(exc))
    if name in {"OfflineError"}:
        from .errors import OfflineAssetError

        return OfflineAssetError(str(exc))
    if name in {"IntegrityError", "ManifestError", "UnsafePathError", "LockError"}:
        return AssetCacheError(str(exc))
    if name in {"CatalogError"}:
        return CatalogUnavailableError(str(exc))
    if name in {"RuntimeContractError", "CapabilityError", "UnsupportedSystemError"}:
        return UnsupportedModelError(str(exc))
    if operation == "infer":
        return ModelInferenceError("ONNX model inference failed")
    if operation == "open":
        return SessionCreationError("Could not open Piper ONNX runtime")
    return AssetDownloadError(str(exc)) if operation == "install" else AssetError(str(exc))


def _call(operation: str, function: Callable[[], Any]) -> Any:
    try:
        return function()
    except (FileNotFoundError, ValueError, TypeError):
        raise
    except Exception as exc:
        mapped = _map_error(exc, operation=operation)
        raise mapped from exc


def open_local_voice(
    model_path: str | Path,
    config_path: str | Path,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    return _call(
        "open",
        lambda: module.open_local(
            system="piper",
            model=model_path,
            config=config_path,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )


def install_pretrained_voice(
    ref: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: Callable[[AssetProgressEvent], None] | None = None,
) -> ResolvedPiperVoice:
    normalized = normalize_piper_ref(ref)
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir, offline=bool(offline))
    installation = _call(
        "install",
        lambda: manager.install(
            normalized,
            refresh=refresh_catalog,
            force=force_download,
            progress=adapt_progress(progress),
        ),
    )
    return installation_to_voice_info(installation, ref=normalized)


def open_installed_voice(
    resolved: ResolvedPiperVoice,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    manager = module.OnnxVoice()
    return _call(
        "open",
        lambda: manager.open(
            resolved.installation,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )


def installation_to_voice_info(installation: Any, *, ref: str | None = None) -> ResolvedPiperVoice:
    try:
        model = installation.artifact("model").path
        config = installation.artifact("config").path
    except (KeyError, AttributeError) as exc:
        raise UnsupportedModelError(
            "Piper installation must contain model and config artifacts"
        ) from exc
    try:
        model_card = installation.artifact("model_card").path
    except KeyError:
        model_card = None
    raw_metadata = dict(getattr(installation, "metadata", {}) or {})
    language = raw_metadata.get("language") or {}
    language_code = language.get("code") if isinstance(language, Mapping) else None
    return ResolvedPiperVoice(
        ref=ref or getattr(installation, "ref", None),
        voice_id=getattr(installation, "id", None),
        model_path=Path(model),
        config_path=Path(config),
        model_card_path=Path(model_card) if model_card is not None else None,
        language_code=str(language_code) if language_code is not None else None,
        source_revision=(
            str(raw_metadata["source_revision"])
            if raw_metadata.get("source_revision") is not None
            else None
        ),
        metadata=raw_metadata,
        installation=installation,
    )


def runtime_diagnostics(runtime: Any) -> dict[str, Any]:
    installation = getattr(runtime, "installation", None)
    model_path = None
    config_path = None
    if installation is not None:
        try:
            model_path = str(installation.artifact("model").path)
        except (KeyError, AttributeError):
            pass
        try:
            config_path = str(installation.artifact("config").path)
        except (KeyError, AttributeError):
            pass
    session = getattr(runtime, "session", None)
    return {
        "model_path": model_path,
        "config_path": config_path,
        "providers_requested": tuple(getattr(session, "provider_request", ()) or ()),
        "providers_active": tuple(getattr(session, "resolved_providers", ()) or ()),
        "model_inputs": tuple(getattr(session, "input_names", ()) or ()),
        "model_outputs": tuple(getattr(session, "output_names", ()) or ()),
    }


def summarize_array(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    return {"shape": [int(size) for size in array.shape], "dtype": str(array.dtype)}


def summarize_inference(result: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    timing = getattr(result, "timings", None)
    timings = summarize_array(timing) if timing is not None else None
    outputs = {
        str(name): summarize_array(value)
        for name, value in dict(getattr(result, "outputs", {}) or {}).items()
    }
    return timings, outputs


def available_providers() -> tuple[str, ...]:
    return tuple(_call("open", lambda: _onnxvoice().available_providers()))


__all__ = [
    "ResolvedPiperVoice",
    "adapt_progress",
    "available_providers",
    "installation_to_voice_info",
    "install_pretrained_voice",
    "normalize_piper_ref",
    "normalize_provider_request",
    "open_installed_voice",
    "open_local_voice",
    "runtime_diagnostics",
    "summarize_inference",
]
