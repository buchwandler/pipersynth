from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from piperg2p import (
    OverrideSpan,
    RawPhonemeSegment,
    TokenAnnotation,
    VoiceConfig,
    get_g2p,
    parse_raw_blocks,
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
    InvalidSpeakerError,
    InvalidSynthesisConfigError,
    ModelFileNotFoundError,
    ModelInferenceError,
    VoiceClosedError,
)
from .session import ProviderConfig, ProviderSpec
from .types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedChunk,
    RenderedSegment,
    SynthesisConfig,
    SynthesisSegment,
    TextChunkingConfig,
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


@dataclass(frozen=True, slots=True)
class _TextPart:
    segment: SynthesisSegment
    metadata: Mapping[str, Any]


def _remap_segment(segment: SynthesisSegment, start: int, end: int) -> SynthesisSegment:
    overrides = tuple(
        PronunciationOverride(
            override.start - start,
            override.end - start,
            override.phonemes,
            override.language,
            override.stress,
        )
        for override in segment.pronunciation_overrides
        if start <= override.start and override.end <= end
    )
    annotations = tuple(
        LinguisticToken(
            token.start - start,
            token.end - start,
            token.text,
            token.pos,
            token.tag,
            token.lemma,
            token.language,
            token.morph,
        )
        for token in segment.annotations
        if start <= token.start and token.end <= end
    )
    return SynthesisSegment(
        id=segment.id,
        text=segment.text[start:end],
        language=segment.language,
        speaker=segment.speaker,
        pronunciation_overrides=overrides,
        annotations=annotations,
    )


def _request_text_parts(
    segment: SynthesisSegment, chunking: TextChunkingConfig | None
) -> tuple[_TextPart, ...]:
    chunking = chunking if chunking is not None else TextChunkingConfig()
    if not isinstance(chunking, TextChunkingConfig):
        raise TypeError("chunking must be a TextChunkingConfig")
    if chunking.mode == "none":
        return (
            _TextPart(
                segment,
                {
                    "split_mode": "none",
                    "char_start": 0,
                    "char_end": len(segment.text),
                },
            ),
        )

    from phrasplit import split_with_offsets_with_diagnostics

    result = split_with_offsets_with_diagnostics(
        segment.text,
        mode="sentence",
        use_spacy=False,
        language=segment.language,
        max_chars=chunking.max_chars,
    )
    spans = tuple(result.segments)
    backend = f"phrasplit-{result.diagnostics.backend}"
    split_diagnostics = {
        "backend": result.diagnostics.backend,
        "language": result.diagnostics.language,
        "analysis_source": result.diagnostics.analysis_source,
        "model_owned_by_caller": result.diagnostics.model_owned_by_caller,
    }
    protected_ranges = [
        (override.start, override.end) for override in segment.pronunciation_overrides
    ]
    protected_ranges.extend(
        (annotation.start, annotation.end) for annotation in segment.annotations
    )
    protected_ranges.extend(
        (raw.source_start, raw.source_end)
        for raw in parse_raw_blocks(segment.text)
        if isinstance(raw, RawPhonemeSegment) and raw.source_end is not None
    )
    if not spans:
        return (
            _TextPart(
                segment,
                {
                    "split_mode": "sentence",
                    "split_backend": backend,
                    "split_diagnostics": split_diagnostics,
                    "char_start": 0,
                    "char_end": len(segment.text),
                    "split_id": None,
                },
            ),
        )

    candidate_ends = [span.char_end for span in spans]
    candidate_ends[-1] = len(segment.text)
    part_ends = [
        boundary
        for boundary in candidate_ends
        if not any(start < boundary < end for start, end in protected_ranges)
    ]
    parts: list[_TextPart] = []
    part_start = 0
    for part_end in part_ends:
        if part_end <= part_start:
            continue
        split_ids = tuple(
            span.id for span in spans if span.char_end > part_start and span.char_start < part_end
        )
        part_segment = _remap_segment(segment, part_start, part_end)
        parts.append(
            _TextPart(
                part_segment,
                {
                    "split_mode": "sentence",
                    "split_backend": backend,
                    "split_diagnostics": split_diagnostics,
                    "char_start": part_start,
                    "char_end": part_end,
                    "split_id": "+".join(split_ids) if split_ids else None,
                },
            )
        )
        part_start = part_end
    if part_start < len(segment.text):
        parts.append(
            _TextPart(
                _remap_segment(segment, part_start, len(segment.text)),
                {
                    "split_mode": "sentence",
                    "split_backend": backend,
                    "split_diagnostics": split_diagnostics,
                    "char_start": part_start,
                    "char_end": len(segment.text),
                    "split_id": None,
                },
            )
        )
    return tuple(parts)


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
        g2p = self._g2p_factory(segment.language, config=self.config, **self._g2p_options)
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
        result = g2p.phonemize_prepared(
            segment.text,
            overrides=overrides or None,
            annotations=annotations or None,
        )
        self._last_g2p_diagnostics = result.diagnostics
        return result

    def _iter_rendered_chunks(
        self,
        segment: SynthesisSegment,
        phonemized: Any,
        synthesis_config: SynthesisConfig,
        speaker_id: int | None,
        *,
        chunk_index_start: int = 0,
        request_warnings: Sequence[str] | None = None,
        text_chunk_metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[RenderedChunk]:
        chunk_index = chunk_index_start
        warnings_for_request = (
            tuple(phonemized.warnings) if request_warnings is None else tuple(request_warnings)
        )
        is_first_chunk = True
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
            if text_chunk_metadata is not None:
                metadata["text_chunk"] = dict(text_chunk_metadata)
            application = self.last_voice_level_application
            if application is not None:
                metadata["voice_level"] = {
                    "applied": application.applied,
                    "gain_db": application.gain_db,
                    "source": application.source,
                    "key": str(application.key) if application.key else None,
                }
            chunk_warnings = tuple(
                dict.fromkeys(
                    (*sentence.warnings, *(warnings_for_request if is_first_chunk else ()))
                )
            )
            yield RenderedChunk(
                index=chunk_index,
                audio=audio,
                sample_rate=inference.sample_rate,
                segment_id=segment.id,
                phonemes=tuple(sentence.phonemes),
                phoneme_ids=tuple(sentence.ids),
                warnings=chunk_warnings,
                metadata=metadata,
            )
            chunk_index += 1
            is_first_chunk = False

    def _iter_phonemized_parts(
        self, parts: Sequence[_TextPart]
    ) -> Iterator[tuple[_TextPart, Any, tuple[str, ...]]]:
        seen_warnings: set[str] = set()
        for part in parts:
            phonemized = self._phonemize_segment(part.segment)
            new_warnings = tuple(
                warning for warning in phonemized.warnings if warning not in seen_warnings
            )
            seen_warnings.update(phonemized.warnings)
            yield part, phonemized, new_warnings

    def iter_chunks(
        self,
        segment: SynthesisSegment,
        *,
        config: SynthesisConfig | None = None,
        chunking: TextChunkingConfig | None = None,
    ) -> Iterator[RenderedChunk]:
        """Yield inferred Piper sentence groups in source order."""
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("segment must be a SynthesisSegment")
        synthesis_config = config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(segment.speaker)
        parts = _request_text_parts(segment, chunking)
        chunk_index = 0
        for part, phonemized, request_warnings in self._iter_phonemized_parts(parts):
            for chunk in self._iter_rendered_chunks(
                part.segment,
                phonemized,
                synthesis_config,
                speaker_id,
                chunk_index_start=chunk_index,
                request_warnings=request_warnings,
                text_chunk_metadata=part.metadata,
            ):
                yield chunk
                chunk_index = chunk.index + 1

    def synthesize(
        self,
        segment: SynthesisSegment,
        *,
        config: SynthesisConfig | None = None,
        chunking: TextChunkingConfig | None = None,
    ) -> RenderedSegment:
        """Render one independent prepared-text speech request."""
        self._ensure_open()
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("segment must be a SynthesisSegment")
        synthesis_config = config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(segment.speaker)
        text_parts = _request_text_parts(segment, chunking)
        rendered_chunks: list[RenderedChunk] = []
        all_warnings: list[str] = []
        chunk_index = 0
        for part, phonemized, new_warnings in self._iter_phonemized_parts(text_parts):
            all_warnings.extend(new_warnings)
            for chunk in self._iter_rendered_chunks(
                part.segment,
                phonemized,
                synthesis_config,
                speaker_id,
                chunk_index_start=chunk_index,
                request_warnings=new_warnings,
                text_chunk_metadata=part.metadata,
            ):
                rendered_chunks.append(chunk)
                all_warnings.extend(chunk.warnings)
                chunk_index = chunk.index + 1
        chunks = tuple(rendered_chunks)
        audio = (
            np.concatenate([chunk.audio for chunk in chunks]).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        phonemes = tuple(symbol for chunk in chunks for symbol in chunk.phonemes)
        phoneme_ids = tuple(identifier for chunk in chunks for identifier in chunk.phoneme_ids)
        warnings = tuple(dict.fromkeys(all_warnings))
        metadata: dict[str, Any] = {
            "text_chunking": {
                "mode": (chunking or TextChunkingConfig()).mode,
                "backend": text_parts[0].metadata.get("split_backend"),
                "diagnostics": text_parts[0].metadata.get("split_diagnostics"),
                "input_chars": len(segment.text),
                "text_parts": len(text_parts),
                "rendered_chunks": len(chunks),
                "max_chars": (chunking or TextChunkingConfig()).max_chars,
            }
        }
        if self._last_g2p_diagnostics is not None:
            metadata["frontend_diagnostics"] = self._last_g2p_diagnostics
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
        chunking: TextChunkingConfig | None = None,
    ) -> RenderedSegment:
        """Synthesize already-prepared speakable text."""
        segment = SynthesisSegment(
            id=id if id is not None else uuid4().hex,
            text=prepared_text,
            language=language,
            speaker=speaker,
        )
        return self.synthesize(segment, config=config, chunking=chunking)

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
