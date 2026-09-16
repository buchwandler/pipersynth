from __future__ import annotations

import wave
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from piperg2p import PiperFrontend, VoiceConfig

from .audio import postprocess_audio, silence_samples
from .diagnostics import RuntimeDiagnostics
from .errors import (
    ConfigFileNotFoundError,
    InvalidSpeakerError,
    ModelFileNotFoundError,
    ModelInferenceError,
    VoiceClosedError,
)
from .session import OnnxSessionManager, ProviderConfig, ProviderSpec
from .types import AudioChunk, SynthesisConfig

SessionFactory = Callable[..., Any]


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
    return np.asarray(audio, dtype=np.float32)


class PiperVoice:
    """Independent ONNX synthesis runtime for a Piper-compatible voice model."""

    def __init__(
        self,
        session: Any,
        config: VoiceConfig,
        frontend: PiperFrontend,
        *,
        owns_frontend: bool = False,
        session_manager: OnnxSessionManager | None = None,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.session = session
        self.config = config
        self.frontend = frontend
        self._owns_frontend = owns_frontend
        self._session_manager = session_manager
        self.model_path = Path(model_path) if model_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self._closed = False

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        config_path: str | Path | None = None,
        *,
        providers: Sequence[ProviderSpec | ProviderConfig] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        session_factory: SessionFactory | None = None,
        frontend: PiperFrontend | None = None,
        frontend_options: Mapping[str, Any] | None = None,
    ) -> PiperVoice:
        """Load an ONNX voice and its companion ``.onnx.json`` configuration."""

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
        manager = OnnxSessionManager(
            model,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            session_factory=session_factory,
        )
        try:
            session = manager.create(require_sid=config.num_speakers > 1)
        except Exception:
            if owns_frontend:
                frontend.close()
            raise
        return cls(
            session,
            config,
            frontend,
            owns_frontend=owns_frontend,
            session_manager=manager,
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
        progress: Callable[..., Any] | None = None,
    ) -> PiperVoice:
        from .asset_manager import VoiceAssetManager

        manager = VoiceAssetManager(cache_dir, offline=offline, progress=progress)
        bundle = manager.resolve_voice(
            voice, refresh_catalog=refresh_catalog, force_download=force_download
        )
        return cls.load(
            bundle.model_path,
            bundle.config_path,
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
        frontend_diagnostics = getattr(self.frontend, "diagnostics", None)
        frontend_name = getattr(frontend_diagnostics, "backend", None)
        speaker_names = tuple(getattr(self.config, "speaker_id_map", {}).keys())
        fields = {
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "sample_rate": self.config.sample_rate,
            "num_symbols": self.config.num_symbols,
            "num_speakers": self.config.num_speakers,
            "speaker_names": speaker_names,
            "phoneme_type": self.config.phoneme_type.value,
            "espeak_voice": self.config.espeak_voice,
            "frontend": frontend_name,
        }
        if self._session_manager is not None:
            return self._session_manager.diagnostics(**fields)
        return RuntimeDiagnostics(
            model_path=str(self.model_path) if self.model_path is not None else None,
            **fields,
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

    def _resolved_scales(self, syn: SynthesisConfig) -> np.ndarray:
        return np.asarray(
            [
                self.config.noise_scale if syn.noise_scale is None else syn.noise_scale,
                self.config.length_scale if syn.length_scale is None else syn.length_scale,
                self.config.noise_w_scale
                if syn.resolved_noise_w_scale is None
                else syn.resolved_noise_w_scale,
            ],
            dtype=np.float32,
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

    def synthesize_ids(
        self,
        phoneme_ids: Sequence[int],
        syn_config: SynthesisConfig | None = None,
    ) -> np.ndarray:
        """Run acoustic inference from already encoded phoneme IDs."""

        self._ensure_open()
        ids_values = self._validated_ids(phoneme_ids)
        if not ids_values:
            return np.zeros(0, dtype=np.float32)
        syn = syn_config or SynthesisConfig()
        speaker_id = self.resolve_speaker_id(syn.speaker_id)
        ids = np.asarray([ids_values], dtype=np.int64)
        args: dict[str, np.ndarray] = {
            "input": ids,
            "input_lengths": np.asarray([ids.shape[1]], dtype=np.int64),
            "scales": self._resolved_scales(syn),
        }
        if self.config.num_speakers > 1:
            assert speaker_id is not None
            args["sid"] = np.asarray([speaker_id], dtype=np.int64)
        try:
            result = self.session.run(None, args)
        except Exception as exc:
            raise ModelInferenceError("ONNX model inference failed") from exc
        if not result:
            raise ModelInferenceError("ONNX model returned no outputs")
        audio = _reduce_waveform(result[0])
        return postprocess_audio(audio, normalize=syn.normalize_audio, volume=syn.volume)

    def synthesize(
        self,
        text: str,
        syn_config: SynthesisConfig | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield one audio chunk per sentence returned by ``piperg2p``."""

        self._ensure_open()
        result = self.frontend.phonemize_prepared(text)
        for sentence in result.sentences:
            if not sentence.ids:
                continue
            audio = self.synthesize_ids(sentence.ids, syn_config)
            metadata: dict[str, Any] = {}
            if result.diagnostics is not None:
                metadata["frontend_diagnostics"] = result.diagnostics
            yield AudioChunk(
                sample_rate=self.config.sample_rate,
                audio_float_array=audio,
                phonemes=tuple(sentence.phonemes),
                phoneme_ids=tuple(sentence.ids),
                warnings=tuple(sentence.warnings),
                metadata=metadata,
            )

    def synthesize_array(
        self,
        text: str,
        syn_config: SynthesisConfig | None = None,
        *,
        sentence_silence: float = 0.0,
    ) -> np.ndarray:
        """Synthesize text and concatenate all sentence chunks."""

        self._ensure_open()
        silence_count = silence_samples(self.config.sample_rate, sentence_silence)
        chunks = list(self.synthesize(text, syn_config))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if silence_count == 0 or len(chunks) == 1:
            return np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(
                np.float32, copy=False
            )
        silence = np.zeros(silence_count, dtype=np.float32)
        parts: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            if index:
                parts.append(silence)
            parts.append(chunk.audio_float_array)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def synthesize_wav(
        self,
        text: str,
        wav_file: str | Path | BinaryIO,
        syn_config: SynthesisConfig | None = None,
        *,
        sentence_silence: float = 0.0,
        set_wav_format: bool = True,
    ) -> str | Path | BinaryIO:
        """Stream synthesized chunks to a mono 16-bit PCM WAV target."""

        self._ensure_open()
        silence_count = silence_samples(self.config.sample_rate, sentence_silence)
        wav_target = str(wav_file) if isinstance(wav_file, Path) else wav_file
        with wave.open(wav_target, "wb") as handle:
            if set_wav_format:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self.config.sample_rate)
            silence = np.zeros(silence_count, dtype=np.float32)
            wrote_chunk = False
            for chunk in self.synthesize(text, syn_config):
                if wrote_chunk and silence_count:
                    handle.writeframes(silence.astype(np.int16).tobytes())
                handle.writeframes(chunk.audio_int16_bytes)
                wrote_chunk = True
        return wav_file

    def save_wav(
        self,
        path: str | Path,
        text: str,
        syn_config: SynthesisConfig | None = None,
        *,
        sentence_silence: float = 0.0,
    ) -> Path:
        """Synthesize text and save mono 16-bit PCM WAV."""

        target = Path(path)
        self.synthesize_wav(text, target, syn_config, sentence_silence=sentence_silence)
        return target

    def warmup(self) -> None:
        self._ensure_open()
        if self._session_manager is not None:
            self._session_manager.create(require_sid=self.config.num_speakers > 1)

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_frontend:
            self.frontend.close()
        if self._session_manager is not None:
            self._session_manager.close()
        self.session = None
        self._closed = True

    def __enter__(self) -> PiperVoice:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
