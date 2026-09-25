from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from piperg2p import (
    OverrideSpan,
    TokenAnnotation,
    VoiceConfig,
    get_g2p,
)

from ._onnxvoice import (
    ResolvedPiperVoice,
    install_pretrained_voice,
    open_installed_voice,
    open_local_voice,
    runtime_diagnostics,
    summarize_inference,
)
from .asset_progress import AssetProgressEvent
from .audio import finish_audio, prepare_audio
from .diagnostics import RuntimeDiagnostics
from .errors import (
    ConfigFileNotFoundError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidSpeakerError,
    InvalidSynthesisConfigError,
    ModelFileNotFoundError,
    ModelInferenceError,
    SynthesisInputTooLongError,
    VoiceClosedError,
)
from .session import ProviderConfig, ProviderSpec
from .types import (
    LinguisticToken,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
)
from .voice_level import (
    VoiceCalibrationKey,
    VoiceLevelApplication,
    apply_voice_level_calibration,
)

try:
    from ._version import __version__ as _PIPERSYNTH_VERSION
except ImportError:
    _PIPERSYNTH_VERSION = "unknown"

_TRANSIENT_G2P_OPTIONS = frozenset(
    {
        "cache_dir",
        "cache_path",
        "download_dir",
        "progress",
        "progress_callback",
        "force_download",
        "offline",
        "refresh_catalog",
    }
)
_UNSTABLE_OPTION = object()


def _stable_option_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else _UNSTABLE_OPTION
    if isinstance(value, Mapping):
        return {
            key: stable_value
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if isinstance(key, str)
            and key.casefold() not in _TRANSIENT_G2P_OPTIONS
            and (stable_value := _stable_option_value(item)) is not _UNSTABLE_OPTION
        }
    if isinstance(value, (tuple, list)):
        items = [_stable_option_value(item) for item in value]
        return items if all(item is not _UNSTABLE_OPTION for item in items) else _UNSTABLE_OPTION
    return _UNSTABLE_OPTION


def _stable_g2p_options(options: Mapping[str, Any]) -> dict[str, Any]:
    stable = _stable_option_value(options)
    return stable if isinstance(stable, dict) else {}


def _g2p_version() -> str:
    try:
        return distribution_version("piperg2p")
    except PackageNotFoundError:
        return "unknown"


def _reduce_waveform(value: Any) -> np.ndarray:
    audio = np.asarray(value)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    else:
        non_singleton = [size for size in audio.shape if size != 1]
        if len(non_singleton) > 1:
            raise ModelInferenceError(
                f"model returned ambiguous waveform shape {audio.shape}; expected [T], [1, T], or [1, 1, T]"
            )
        audio = np.squeeze(audio)
        if audio.ndim == 0:
            audio = audio.reshape(1)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ModelInferenceError("model returned non-finite audio")
    return audio


@dataclass(frozen=True, slots=True)
class PiperInference:
    audio: np.ndarray
    sample_rate: int
    timing_summary: Mapping[str, Any] | None = None
    output_summary: Mapping[str, Any] = field(default_factory=dict)


class PiperVoice:
    """Piper policy using PiperG2P around an OnnxVoice runtime."""

    def __init__(
        self,
        runtime: Any,
        config: VoiceConfig,
        *,
        g2p_factory: Callable[..., Any] = get_g2p,
        g2p_options: Mapping[str, Any] | None = None,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        installation: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self._g2p_factory = g2p_factory
        self._g2p_options = dict(g2p_options or {})
        self.model_path = Path(model_path) if model_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self.installation = installation
        self._closed = False
        self._last_g2p_diagnostics: Any | None = None

        self._last_voice_level_application: VoiceLevelApplication | None = None

    @classmethod
    def _from_resolved(
        cls,
        resolved: ResolvedPiperVoice,
        *,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        g2p_factory: Callable[..., Any] = get_g2p,
        g2p_options: Mapping[str, Any] | None = None,
    ) -> PiperVoice:
        config = VoiceConfig.from_json(resolved.config_path)
        runtime = open_installed_voice(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(
            runtime,
            config,
            g2p_factory=g2p_factory,
            g2p_options=g2p_options,
            model_path=resolved.model_path,
            config_path=resolved.config_path,
            installation=resolved.installation,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Any,
        *,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        g2p_factory: Callable[..., Any] = get_g2p,
        g2p_options: Mapping[str, Any] | None = None,
    ) -> PiperVoice:
        """Load a managed VoiceBundle without discarding its catalog identity."""
        if getattr(bundle, "installation", None) is None:
            return cls.from_local(
                bundle.model_path,
                bundle.config_path,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
                g2p_factory=g2p_factory,
                g2p_options=g2p_options,
            )
        from ._onnxvoice import installation_to_voice_info

        return cls._from_resolved(
            installation_to_voice_info(bundle.installation),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            g2p_factory=g2p_factory,
            g2p_options=g2p_options,
        )

    @classmethod
    def from_local(
        cls,
        model_path: str | Path,
        config_path: str | Path | None = None,
        *,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        g2p_factory: Callable[..., Any] = get_g2p,
        g2p_options: Mapping[str, Any] | None = None,
    ) -> PiperVoice:
        """Load an explicit local Piper model through OnnxVoice."""

        model = Path(model_path)
        if not model.exists():
            raise ModelFileNotFoundError(f"ONNX model file does not exist: {model}")
        config_file = Path(config_path) if config_path is not None else Path(f"{model}.json")
        if not config_file.exists():
            raise ConfigFileNotFoundError(f"Voice config file does not exist: {config_file}")
        config = VoiceConfig.from_json(config_file)
        runtime = open_local_voice(
            model,
            config_file,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(
            runtime,
            config,
            g2p_factory=g2p_factory,
            g2p_options=g2p_options,
            model_path=model,
            config_path=config_file,
        )

    @classmethod
    def from_pretrained(
        cls,
        voice: str,
        *,
        cache_dir: str | Path | None = None,
        offline: bool | None = None,
        refresh_catalog: bool = False,
        force_download: bool = False,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        g2p_factory: Callable[..., Any] = get_g2p,
        g2p_options: Mapping[str, Any] | None = None,
        progress: Callable[[AssetProgressEvent], None] | None = None,
    ) -> PiperVoice:
        resolved = install_pretrained_voice(
            voice,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            progress=progress,
        )
        return cls._from_resolved(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            g2p_factory=g2p_factory,
            g2p_options=g2p_options,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def diagnostics(self) -> RuntimeDiagnostics:
        self._ensure_open()
        runtime_fields = runtime_diagnostics(self.runtime)
        g2p_diagnostics = self._last_g2p_diagnostics
        frontend_name = getattr(g2p_diagnostics, "backend", None)
        speaker_names = tuple(getattr(self.config, "speaker_id_map", {}).keys())
        return RuntimeDiagnostics(
            model_path=runtime_fields.get("model_path")
            or (str(self.model_path) if self.model_path is not None else None),
            config_path=runtime_fields.get("config_path")
            or (str(self.config_path) if self.config_path is not None else None),
            sample_rate=self.config.sample_rate,
            num_symbols=self.config.num_symbols,
            num_speakers=self.config.num_speakers,
            speaker_names=speaker_names,
            phoneme_type=self.config.phoneme_type.value,
            espeak_voice=self.config.espeak_voice,
            providers_requested=tuple(runtime_fields.get("providers_requested", ())),
            providers_active=tuple(runtime_fields.get("providers_active", ())),
            model_inputs=tuple(runtime_fields.get("model_inputs", ())),
            model_outputs=tuple(runtime_fields.get("model_outputs", ())),
            frontend=frontend_name,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise VoiceClosedError("PiperVoice is closed")

    def resolve_speaker_id(self, value: int | str | None) -> int | None:
        """Resolve a direct speaker ID, configured speaker name, or default."""

        self._ensure_open()
        if value is None:
            return self.config.default_speaker_id if self.config.num_speakers > 1 else None
        if isinstance(value, bool):
            raise InvalidSpeakerError("speaker ID must be an integer, name, or None")
        if isinstance(value, str):
            try:
                value = self.config.speaker_id_map[value]
            except KeyError as exc:
                names = ", ".join(sorted(self.config.speaker_id_map)) or "none"
                raise InvalidSpeakerError(
                    f"unknown speaker {value!r}; known speakers: {names}"
                ) from exc
        if not isinstance(value, int):
            raise InvalidSpeakerError("speaker ID must be an integer, name, or None")
        if self.config.num_speakers == 1:
            if value != 0:
                raise InvalidSpeakerError("single-speaker voices only accept speaker ID 0")
            return None
        if value < 0 or value >= self.config.num_speakers:
            raise InvalidSpeakerError(
                f"speaker ID {value} is outside 0..{self.config.num_speakers - 1}"
            )
        return value

    def calibration_key(self, speaker: int | str | None = None) -> VoiceCalibrationKey | None:
        """Return the exact managed Piper identity used for calibration lookup."""
        self._ensure_open()
        if self.installation is None:
            return None
        model_id = getattr(self.installation, "id", None)
        metadata = getattr(self.installation, "metadata", {}) or {}
        quality = metadata.get("quality") if isinstance(metadata, Mapping) else None
        if not isinstance(model_id, str) or not model_id:
            return None
        if not isinstance(quality, str) or not quality:
            return None
        speaker_id = self.resolve_speaker_id(speaker)
        return VoiceCalibrationKey("piper", model_id, quality, f"speaker-{speaker_id or 0}")

    @property
    def last_voice_level_application(self) -> VoiceLevelApplication | None:
        return self._last_voice_level_application

    def _resolved_scales(self, syn: SynthesisConfig) -> tuple[float, float, float]:
        return (
            float(self.config.noise_scale if syn.noise_scale is None else syn.noise_scale),
            float(self.config.length_scale if syn.length_scale is None else syn.length_scale),
            float(self.config.noise_w_scale if syn.noise_w_scale is None else syn.noise_w_scale),
        )

    def _validated_ids(self, phoneme_ids: Sequence[int]) -> list[int]:
        if isinstance(phoneme_ids, (str, bytes, bytearray)):
            raise ValueError("phoneme_ids must be a sequence of integers")
        try:
            values = list(phoneme_ids)
        except TypeError as exc:
            raise ValueError("phoneme_ids must be a sequence of integers") from exc
        for identifier in values:
            if isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)):
                raise ValueError("phoneme IDs must be integers")
            if identifier < 0 or identifier >= self.config.num_symbols:
                raise ValueError(
                    f"phoneme ID {identifier} is outside 0..{self.config.num_symbols - 1}"
                )
        return [int(identifier) for identifier in values]

    def _infer_ids(
        self,
        phoneme_ids: Sequence[int],
        syn_config: SynthesisConfig | None = None,
        *,
        speaker: int | str | None = None,
    ) -> PiperInference:
        self._ensure_open()
        ids_values = self._validated_ids(phoneme_ids)
        if not ids_values:
            return PiperInference(np.zeros(0, dtype=np.float32), self.config.sample_rate)
        syn = syn_config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(speaker)
        noise_scale, length_scale, noise_w = self._resolved_scales(syn)
        try:
            result = self.runtime.infer(
                ids_values,
                speaker_id=speaker_id,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_w=noise_w,
            )
        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            raise ModelInferenceError("ONNX model inference failed") from exc
        if result is None or not hasattr(result, "audio"):
            raise ModelInferenceError("ONNX model returned no audio output")
        sample_rate = int(getattr(result, "sample_rate", 0))
        if sample_rate != self.config.sample_rate:
            raise ModelInferenceError(
                f"model sample rate {sample_rate} does not match voice config {self.config.sample_rate}"
            )
        audio = _reduce_waveform(result.audio)
        timing_summary, output_summary = summarize_inference(result)
        return PiperInference(audio, sample_rate, timing_summary, output_summary)

    def postprocess_inference(
        self,
        audio: np.ndarray,
        syn_config: SynthesisConfig,
        *,
        speaker: int | str | None = None,
    ) -> np.ndarray:
        """Apply engine-local normalization, voice leveling, and output gain."""
        prepared = prepare_audio(audio, normalize=syn_config.normalize_audio)
        key = self.calibration_key(speaker)
        calibrated, application = apply_voice_level_calibration(
            prepared, syn_config.voice_level, key
        )
        self._last_voice_level_application = application
        return finish_audio(calibrated, output_gain=syn_config.output_gain)

    def synthesize_ids(
        self,
        phoneme_ids: Sequence[int],
        config: SynthesisConfig | None = None,
        *,
        speaker: int | str | None = None,
    ) -> np.ndarray:
        """Run acoustic inference from already encoded phoneme IDs."""
        synthesis_config = config or SynthesisConfig()
        inference = self._infer_ids(phoneme_ids, synthesis_config, speaker=speaker)
        return self.postprocess_inference(inference.audio, synthesis_config, speaker=speaker)

    def _phonemize_request(self, request: SynthesisRequest) -> Any:
        self._ensure_open()
        if not isinstance(request, SynthesisRequest):
            raise InvalidRequestError("request must be a SynthesisRequest")
        model_language = self.config.espeak_voice
        phoneme_type = getattr(self.config.phoneme_type, "value", self.config.phoneme_type)
        if phoneme_type == "espeak" and model_language:
            requested = request.language.casefold().replace("_", "-")
            active = model_language.casefold().replace("_", "-")
            if requested != active:
                raise InvalidLanguageError(
                    f"language {request.language!r} is incompatible with active Piper model language "
                    f"{model_language!r}"
                )
        overrides = tuple(
            OverrideSpan(
                override.start,
                override.end,
                {
                    **({"ph": override.phonemes} if override.phonemes is not None else {}),
                    **({"lang": override.language} if override.language is not None else {}),
                    **({"stress": int(override.stress)} if override.stress is not None else {}),
                },
            )
            for override in request.pronunciation_overrides
        )
        annotations = tuple(
            TokenAnnotation(
                start=token.start,
                end=token.end,
                text=token.text,
                pos=token.pos,
                tag=token.tag,
                lemma=token.lemma,
                language=token.language,
                morph=token.morph,
            )
            for token in request.tokens
        )
        try:
            g2p = self._g2p_factory(request.language, config=self.config, **self._g2p_options)
            result = g2p.phonemize_prepared(
                request.text,
                overrides=overrides or None,
                annotations=annotations or None,
            )
        except Exception as exc:
            raise InvalidRequestError("PiperG2P could not process the synthesis request") from exc
        self._last_g2p_diagnostics = result.diagnostics
        return result

    def _known_max_phonemes(self) -> int | None:
        metadata = getattr(self.installation, "metadata", None)
        candidates = [
            getattr(self.runtime, "max_phonemes", None),
            getattr(self.config, "max_phonemes", None),
            metadata.get("max_phonemes") if isinstance(metadata, Mapping) else None,
        ]
        for value in candidates:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    def _model_id(self) -> str | None:
        model_id = getattr(self.installation, "id", None)
        if isinstance(model_id, str) and model_id:
            return model_id
        return str(self.model_path) if self.model_path is not None else None

    def _synthesis_identity(
        self,
        request: SynthesisRequest,
        speaker_id: int | None,
        config: SynthesisConfig,
        application: VoiceLevelApplication,
    ) -> dict[str, Any]:
        model_metadata = getattr(self.installation, "metadata", None)
        model_metadata = model_metadata if isinstance(model_metadata, Mapping) else {}
        model_id = getattr(self.installation, "id", None)
        model_id = model_id if isinstance(model_id, str) and model_id else None
        model_revision = model_metadata.get("source_revision")
        quality = model_metadata.get("quality")
        noise_scale, length_scale, noise_w_scale = self._resolved_scales(config)
        frontend = getattr(self._last_g2p_diagnostics, "backend", None)
        if frontend is not None and not isinstance(frontend, str):
            frontend = getattr(frontend, "value", None)
        calibration_key = application.calibration_key
        return {
            "pipersynth_version": _PIPERSYNTH_VERSION,
            "model_id": model_id,
            "model_revision": model_revision if isinstance(model_revision, str) else None,
            "quality": quality if isinstance(quality, str) else None,
            "speaker_id": speaker_id,
            "language": request.language,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w_scale": noise_w_scale,
            "normalize_audio": config.normalize_audio,
            "output_gain": config.output_gain,
            "voice_level": {
                "mode": application.mode,
                "source": application.source,
                "gain_db": application.gain_db,
                "calibration_key": str(calibration_key) if calibration_key else None,
                "catalog_revision": application.catalog_revision,
            },
            "frontend": {
                "backend": frontend,
                "g2p_version": _g2p_version(),
                "options": _stable_g2p_options(self._g2p_options),
            },
        }

    def synthesize(
        self,
        request: SynthesisRequest | SynthesisSegment,
        *,
        config: SynthesisConfig | None = None,
    ) -> SynthesisResult:
        """Synthesize one already-shaped request without choosing text boundaries."""
        self._ensure_open()
        if isinstance(request, SynthesisSegment):
            request = SynthesisRequest(
                id=request.id,
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                tokens=request.annotations,
                pronunciation_overrides=request.pronunciation_overrides,
            )
        if not isinstance(request, SynthesisRequest):
            raise InvalidRequestError("request must be a SynthesisRequest")
        if config is not None and not isinstance(config, SynthesisConfig):
            raise InvalidSynthesisConfigError("config must be a SynthesisConfig")
        synthesis_config = config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(request.speaker)
        effective_speaker_id = (
            speaker_id if speaker_id is not None else (self.config.default_speaker_id or 0)
        )
        phonemized = self._phonemize_request(request)
        sentences = tuple(phonemized.sentences)
        phonemes = tuple(phoneme for sentence in sentences for phoneme in sentence.phonemes)
        phoneme_ids = tuple(identifier for sentence in sentences for identifier in sentence.ids)
        max_phonemes = self._known_max_phonemes()
        if max_phonemes is not None and len(phoneme_ids) > max_phonemes:
            raise SynthesisInputTooLongError(
                text_length=len(request.text),
                phoneme_count=len(phoneme_ids),
                max_phonemes=max_phonemes,
                model_id=self._model_id(),
            )
        inference = self._infer_ids(phoneme_ids, synthesis_config, speaker=speaker_id)
        audio = self.postprocess_inference(inference.audio, synthesis_config, speaker=speaker_id)
        warnings = tuple(
            dict.fromkeys(
                (
                    *phonemized.warnings,
                    *(warning for sentence in sentences for warning in sentence.warnings),
                )
            )
        )
        metadata: dict[str, Any] = {
            "speaker_id": effective_speaker_id,
            "phonemes": phonemes,
            "phoneme_ids": phoneme_ids,
        }
        if phonemized.diagnostics is not None:
            metadata["frontend_diagnostics"] = phonemized.diagnostics
        if inference.timing_summary is not None:
            metadata["inference_timing"] = inference.timing_summary
        if inference.output_summary:
            metadata["inference_output"] = inference.output_summary
        if max_phonemes is not None:
            metadata["input_capacity"] = {
                "phoneme_count": len(phoneme_ids),
                "max_phonemes": max_phonemes,
            }
        application = self.last_voice_level_application
        if application is not None:
            calibration_key = application.calibration_key
            metadata["voice_level"] = {
                "mode": application.mode,
                "applied": application.applied,
                "gain_db": application.gain_db,
                "source": application.source,
                "calibration_key": str(calibration_key) if calibration_key else None,
                "reason": application.reason,
                "catalog_revision": application.catalog_revision,
            }
            identity = self._synthesis_identity(
                request, effective_speaker_id, synthesis_config, application
            )
            identity_payload = json.dumps(
                {"text": request.text, "identity": identity},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            metadata["synthesis_identity"] = identity
            metadata["synthesis_hash"] = hashlib.sha256(
                identity_payload.encode("utf-8")
            ).hexdigest()
        return SynthesisResult(
            id=request.id,
            audio=audio,
            sample_rate=inference.sample_rate,
            text=request.text,
            language=request.language,
            warnings=warnings,
            word_timings=(),
            metadata=metadata,
        )

    def synthesize_text(
        self,
        prepared_text: str,
        *,
        language: str,
        id: str | None = None,
        speaker: int | str | None = None,
        tokens: tuple[LinguisticToken, ...] = (),
        pronunciation_overrides: tuple[PronunciationOverride, ...] = (),
        config: SynthesisConfig | None = None,
    ) -> SynthesisResult:
        """Synthesize one prepared request; text is never split by this helper."""
        request = SynthesisRequest(
            id=id if id is not None else uuid4().hex,
            text=prepared_text,
            language=language,
            speaker=speaker,
            tokens=tokens,
            pronunciation_overrides=pronunciation_overrides,
        )
        return self.synthesize(request, config=config)

    def warmup(self) -> None:
        self._ensure_open()
        _ = getattr(self.runtime, "session", None)

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()
        self._closed = True

    def __enter__(self) -> PiperVoice:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
