from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from piperg2p import OverrideSpan, PiperFrontend, TokenAnnotation, VoiceConfig, get_g2p

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
    InvalidSpeakerError,
    InvalidSynthesisConfigError,
    ModelFileNotFoundError,
    ModelInferenceError,
    VoiceClosedError,
)
from .session import ProviderConfig, ProviderSpec
from .types import (
    RenderedChunk,
    RenderedSegment,
    SynthesisConfig,
    SynthesisSegment,
)
from .voice_level import (
    VoiceCalibrationKey,
    VoiceLevelApplication,
    apply_voice_level_calibration,
)


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
    """Piper policy and frontend wrapped around an OnnxVoice runtime."""

    def __init__(
        self,
        runtime: Any,
        config: VoiceConfig,
        frontend: PiperFrontend,
        *,
        owns_frontend: bool = False,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        installation: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.frontend = frontend
        self._owns_frontend = owns_frontend
        self.model_path = Path(model_path) if model_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self.installation = installation
        self._closed = False

        self._last_voice_level_application: VoiceLevelApplication | None = None

    @classmethod
    def _from_resolved(
        cls,
        resolved: ResolvedPiperVoice,
        *,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        frontend_options: Mapping[str, Any] | None = None,
        frontend: PiperFrontend | None = None,
    ) -> PiperVoice:
        config = VoiceConfig.from_json(resolved.config_path)
        owns_frontend = frontend is None
        if frontend is None:
            frontend = PiperFrontend(config, **dict(frontend_options or {}))
        try:
            runtime = open_installed_voice(
                resolved,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
            )
        except Exception:
            if owns_frontend:
                frontend.close()
            raise
        return cls(
            runtime,
            config,
            frontend,
            owns_frontend=owns_frontend,
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
        frontend_options: Mapping[str, Any] | None = None,
    ) -> PiperVoice:
        """Load a managed VoiceBundle without discarding its catalog identity."""
        if getattr(bundle, "installation", None) is None:
            return cls.from_local(
                bundle.model_path,
                bundle.config_path,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
                frontend_options=frontend_options,
            )
        from ._onnxvoice import installation_to_voice_info

        return cls._from_resolved(
            installation_to_voice_info(bundle.installation),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            frontend_options=frontend_options,
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
        frontend: PiperFrontend | None = None,
        frontend_options: Mapping[str, Any] | None = None,
    ) -> PiperVoice:
        """Load an explicit local Piper model through OnnxVoice."""

        model = Path(model_path)
        if not model.exists():
            raise ModelFileNotFoundError(f"ONNX model file does not exist: {model}")
        config_file = Path(config_path) if config_path is not None else Path(f"{model}.json")
        if not config_file.exists():
            raise ConfigFileNotFoundError(f"Voice config file does not exist: {config_file}")
        config = VoiceConfig.from_json(config_file)
        owns_frontend = frontend is None
        if frontend is None:
            frontend = PiperFrontend(config, **dict(frontend_options or {}))
        try:
            runtime = open_local_voice(
                model,
                config_file,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
            )
        except Exception:
            if owns_frontend:
                frontend.close()
            raise
        return cls(
            runtime,
            config,
            frontend,
            owns_frontend=owns_frontend,
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
        frontend_options: Mapping[str, Any] | None = None,
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
            frontend_options=frontend_options,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def diagnostics(self) -> RuntimeDiagnostics:
        self._ensure_open()
        runtime_fields = runtime_diagnostics(self.runtime)
        frontend_diagnostics = getattr(self.frontend, "diagnostics", None)
        frontend_name = getattr(frontend_diagnostics, "backend", None)
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

    def _phonemize_segment(self, segment: SynthesisSegment) -> Any:
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("segment must be a SynthesisSegment")
        model_language = self.config.espeak_voice
        phoneme_type = getattr(self.config.phoneme_type, "value", self.config.phoneme_type)
        if phoneme_type == "espeak" and model_language:
            requested = segment.language.casefold().replace("_", "-")
            active = model_language.casefold().replace("_", "-")
            if requested != active:
                raise InvalidSynthesisConfigError(
                    f"language {segment.language!r} is incompatible with active Piper model language "
                    f"{model_language!r}; use a source-aligned language override for supported spans"
                )
        g2p = get_g2p(segment.language, config=self.config)
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
            for override in segment.pronunciation_overrides
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
            for token in segment.annotations
        )
        return g2p.phonemize_prepared(
            segment.text,
            overrides=overrides or None,
            annotations=annotations or None,
        )

    def _iter_rendered_chunks(
        self,
        segment: SynthesisSegment,
        phonemized: Any,
        synthesis_config: SynthesisConfig,
        speaker_id: int | None,
    ) -> Iterator[RenderedChunk]:
        chunk_index = 0
        request_warnings = tuple(phonemized.warnings)
        for sentence in phonemized.sentences:
            if not sentence.ids:
                continue
            inference = self._infer_ids(sentence.ids, synthesis_config, speaker=speaker_id)
            audio = self.postprocess_inference(
                inference.audio, synthesis_config, speaker=speaker_id
            )
            metadata: dict[str, Any] = {}
            if inference.timing_summary is not None:
                metadata["inference_timing"] = inference.timing_summary
            if inference.output_summary:
                metadata["inference_output"] = inference.output_summary
            if phonemized.diagnostics is not None:
                metadata["frontend_diagnostics"] = phonemized.diagnostics
            application = self.last_voice_level_application
            if application is not None:
                metadata["voice_level"] = {
                    "applied": application.applied,
                    "gain_db": application.gain_db,
                    "source": application.source,
                    "key": str(application.key) if application.key else None,
                }
            warnings = tuple(
                dict.fromkeys((*sentence.warnings, *(request_warnings if chunk_index == 0 else ())))
            )
            yield RenderedChunk(
                index=chunk_index,
                audio=audio,
                sample_rate=inference.sample_rate,
                segment_id=segment.id,
                phonemes=tuple(sentence.phonemes),
                phoneme_ids=tuple(sentence.ids),
                warnings=warnings,
                metadata=metadata,
            )
            chunk_index += 1

    def iter_chunks(
        self,
        segment: SynthesisSegment,
        *,
        config: SynthesisConfig | None = None,
    ) -> Iterator[RenderedChunk]:
        """Yield one inferred chunk per PiperG2P sentence group in a request."""
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("segment must be a SynthesisSegment")
        synthesis_config = config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(segment.speaker)
        phonemized = self._phonemize_segment(segment)
        yield from self._iter_rendered_chunks(segment, phonemized, synthesis_config, speaker_id)

    def synthesize(
        self,
        segment: SynthesisSegment,
        *,
        config: SynthesisConfig | None = None,
    ) -> RenderedSegment:
        """Render one independent prepared-text speech request."""
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("segment must be a SynthesisSegment")
        synthesis_config = config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(segment.speaker)
        phonemized = self._phonemize_segment(segment)
        chunks = tuple(
            self._iter_rendered_chunks(segment, phonemized, synthesis_config, speaker_id)
        )
        audio = (
            np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        phonemes = tuple(symbol for chunk in chunks for symbol in chunk.phonemes)
        phoneme_ids = tuple(identifier for chunk in chunks for identifier in chunk.phoneme_ids)
        warnings = tuple(
            dict.fromkeys(
                (*phonemized.warnings, *(warning for chunk in chunks for warning in chunk.warnings))
            )
        )
        metadata: dict[str, Any] = {}
        if phonemized.diagnostics is not None:
            metadata["frontend_diagnostics"] = phonemized.diagnostics
        if chunks and "voice_level" in chunks[-1].metadata:
            metadata["voice_level"] = chunks[-1].metadata["voice_level"]
        return RenderedSegment(
            id=segment.id,
            audio=audio,
            sample_rate=self.config.sample_rate,
            text=segment.text,
            language=segment.language,
            speaker_id=speaker_id,
            phonemes=phonemes,
            phoneme_ids=phoneme_ids,
            warnings=warnings,
            chunks=chunks,
            diagnostics=self.diagnostics,
            metadata=metadata,
        )

    def synthesize_text(
        self,
        prepared_text: str,
        *,
        language: str,
        id: str | None = None,
        speaker: int | str | None = None,
        config: SynthesisConfig | None = None,
    ) -> RenderedSegment:
        """Synthesize already-prepared speakable text."""
        segment = SynthesisSegment(
            id=id if id is not None else uuid4().hex,
            text=prepared_text,
            language=language,
            speaker=speaker,
        )
        return self.synthesize(segment, config=config)

    def warmup(self) -> None:
        self._ensure_open()
        _ = getattr(self.runtime, "session", None)

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_frontend:
            self.frontend.close()
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()
        self._closed = True

    def __enter__(self) -> PiperVoice:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
