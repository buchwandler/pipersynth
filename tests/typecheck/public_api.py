"""Type-check the public PiperSynth API surface as a consumer sees it."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, assert_type

import pipersynth
from pipersynth import (
    AssetError,
    AssetProgressCallback,
    AssetProgressEvent,
    AudioChunk,
    AudioResult,
    AudioUnitDescriptor,
    AudioUnitResult,
    CacheInfo,
    ConsoleAssetProgress,
    GenerationConfig,
    IdentityTextPreparer,
    LinguisticsConfig,
    PauseConfig,
    PipelineConfig,
    PiperPipeline,
    PiperSynthError,
    PiperVoice,
    PlannerConfig,
    PlanRenderingError,
    PreparedAudioUnits,
    PreparedTextResult,
    ProviderConfig,
    RuntimeDiagnostics,
    SpokenformTextPreparer,
    SSMDConfig,
    SynthesisConfig,
    TimingDiagnostics,
    UtterancePlan,
    UtterancePlanner,
    VoiceAssetManager,
    VoiceBundle,
    VoiceConfig,
    VoiceMetadata,
    available_providers,
    build_pipeline,
    list_cached_voices,
    list_voices,
    load_catalog_voice,
    synthesize,
    synthesize_to_wav,
)

assert_type(pipersynth.__version__, str)

# Public classes and exceptions are the exported type objects.
assert_type(PiperPipeline, type[PiperPipeline])
assert_type(PreparedAudioUnits, type[PreparedAudioUnits])
assert_type(PiperVoice, type[PiperVoice])
assert_type(PipelineConfig, type[PipelineConfig])
assert_type(GenerationConfig, type[GenerationConfig])
assert_type(SynthesisConfig, type[SynthesisConfig])
assert_type(AudioResult, type[AudioResult])
assert_type(AudioChunk, type[AudioChunk])
assert_type(AudioUnitResult, type[AudioUnitResult])
assert_type(AudioUnitDescriptor, type[AudioUnitDescriptor])
assert_type(VoiceAssetManager, type[VoiceAssetManager])
assert_type(VoiceBundle, type[VoiceBundle])
assert_type(VoiceMetadata, type[VoiceMetadata])
assert_type(CacheInfo, type[CacheInfo])
assert_type(ProviderConfig, type[ProviderConfig])
assert_type(RuntimeDiagnostics, type[RuntimeDiagnostics])
assert_type(TimingDiagnostics, type[TimingDiagnostics])
assert_type(PreparedTextResult, type[PreparedTextResult])
assert_type(IdentityTextPreparer, type[IdentityTextPreparer])
assert_type(SpokenformTextPreparer, type[SpokenformTextPreparer])
assert_type(ConsoleAssetProgress, type[ConsoleAssetProgress])
assert_type(AssetProgressEvent, type[AssetProgressEvent])
# UtterPlan re-exports are checked as values; they may not ship type information.
assert callable(LinguisticsConfig)
assert callable(PauseConfig)
assert callable(PlannerConfig)
assert callable(SSMDConfig)
assert callable(UtterancePlan)
assert callable(UtterancePlanner)
assert_type(PiperSynthError, type[PiperSynthError])
assert_type(AssetError, type[AssetError])
assert_type(PlanRenderingError, type[PlanRenderingError])
assert callable(VoiceConfig)

# Configuration values accept the documented field types.
generation = GenerationConfig(speaker="alice", length_scale=1.0)
assert_type(generation, GenerationConfig)
config = PipelineConfig(model_path=Path("voice.onnx"), generation=generation, language="en")
assert_type(config, PipelineConfig)
synthesis = SynthesisConfig(speaker_id=0, length_scale=1.0)
assert_type(synthesis, SynthesisConfig)
provider = ProviderConfig("CPUExecutionProvider")
assert_type(provider, ProviderConfig)


def _progress(event: AssetProgressEvent) -> None:
    pass


callback: AssetProgressCallback = _progress
assert_type(callback, Callable[[AssetProgressEvent], None])


def _check_asset_functions() -> None:
    assert_type(available_providers(), tuple[str, ...])
    assert_type(list_voices(language="en"), tuple[VoiceMetadata, ...])
    assert_type(list_cached_voices(), tuple[VoiceBundle, ...])
    assert_type(
        load_catalog_voice("en_US-amy-low", cache_dir="cache", catalog_path="voices.json"),
        VoiceBundle,
    )


def _check_pipeline_surface(pipeline: PiperPipeline, plan: UtterancePlan) -> None:
    assert_type(pipeline.config, PipelineConfig)
    assert_type(pipeline.voice, PiperVoice)
    assert_type(pipeline.plan("hello"), UtterancePlan)
    assert_type(pipeline.prepare_plan(plan), PreparedAudioUnits)
    assert_type(pipeline.render_plan(plan), AudioResult)
    assert_type(pipeline.prepare_units("hello"), PreparedAudioUnits)
    assert_type(next(pipeline.iter_units("hello")), AudioUnitResult)
    assert_type(next(pipeline.iter_pcm("hello")), bytes)
    assert_type(pipeline.run("hello"), AudioResult)
    assert_type(PiperPipeline.from_pretrained("en_US-amy-low"), PiperPipeline)
    assert_type(build_pipeline(config), PiperPipeline)


def _check_prepared_units_surface(prepared: PreparedAudioUnits) -> None:
    assert_type(prepared.render(), Iterator[AudioUnitResult])


def _check_convenience_surface() -> None:
    assert_type(synthesize("hello", voice="en_US-amy-low"), Any)
    assert_type(synthesize_to_wav("hello", "out.wav", voice="en_US-amy-low"), Path)
