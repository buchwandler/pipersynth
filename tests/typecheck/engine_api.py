from pathlib import Path
from typing import assert_type

from pipersynth import (
    LinguisticToken,
    PiperVoice,
    PronunciationOverride,
    RenderedChunk,
    SynthesisConfig,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    VoiceLevelConfig,
    synthesize,
    synthesize_to_wav,
)

request = SynthesisRequest(
    id="line-1",
    text="Hello.",
    language="en-us",
    pronunciation_overrides=(PronunciationOverride(0, 5, phonemes="həˈloʊ"),),
    tokens=(LinguisticToken(0, 5, text="Hello", morph="Number=Sing"),),
)
segment = SynthesisSegment(
    id="line-2",
    text="Hello.",
    language="en-us",
    annotations=(LinguisticToken(0, 5, text="Hello", morph="Number=Sing"),),
)
config = SynthesisConfig(voice_level=VoiceLevelConfig(mode="calibrated"))
assert_type(request, SynthesisRequest)
assert_type(segment, SynthesisSegment)
assert_type(config, SynthesisConfig)
assert_type(RenderedChunk, type[RenderedChunk])
assert_type(PiperVoice, type[PiperVoice])


def convenience_render() -> SynthesisResult:
    return assert_type(
        synthesize("Hello.", voice="en_US-lessac-medium", language="en-us"),
        SynthesisResult,
    )


def convenience_save() -> Path:
    return assert_type(
        synthesize_to_wav("Hello.", "hello.wav", voice="en_US-lessac-medium", language="en-us"),
        Path,
    )


def render(voice: PiperVoice, request: SynthesisRequest) -> SynthesisResult:
    result = voice.synthesize(request, config=config)
    return assert_type(result, SynthesisResult)


def legacy_segment_render(voice: PiperVoice, request: SynthesisSegment) -> SynthesisResult:
    return assert_type(voice.synthesize(request), SynthesisResult)
