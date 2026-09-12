from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol

from .errors import OptionalDependencyError, TextPreparationError

_RAW_BLOCK_RE = re.compile(r"\[\[.*?\]\]", re.DOTALL)


def normalize_catalog_language_for_spokenform(code: str) -> str:
    """Convert a catalog locale such as ``en_US`` to a spokenform language."""

    normalized = code.replace("-", "_").split("_", 1)[0].strip().lower()
    if not normalized:
        raise ValueError("catalog language code must not be empty")
    return normalized

@dataclass(frozen=True, slots=True)
class PreparedTextResult:
    """Text preparation output with source provenance and diagnostics."""

    source_text: str
    spoken_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def prepared_text(self) -> str:
        return self.spoken_text


class TextPreparer(Protocol):
    def prepare(self, text: str, *, language: str | None) -> PreparedTextResult: ...


class IdentityTextPreparer:
    """Pass text through without changing its meaning or contents."""

    def prepare(self, text: str, *, language: str | None = None) -> PreparedTextResult:
        return PreparedTextResult(text, text, {"mode": "identity", "language": language})


class SpokenformTextPreparer:
    """Adapt spokenform's language-specific normalizer to PiperSynth."""

    def __init__(self, prepare_fn: Callable[..., Any] | None = None) -> None:
        self._prepare_fn = prepare_fn

    def prepare(self, text: str, *, language: str | None) -> PreparedTextResult:
        if not language:
            raise TextPreparationError("language is required for spokenform preparation")
        prepare_fn = self._prepare_fn
        if prepare_fn is None:
            try:
                from spokenform import PreparationConfig, prepare_language
            except ModuleNotFoundError as exc:
                raise OptionalDependencyError(
                    "spokenform preparation requires the optional dependency. "
                    "Install pipersynth[spokenform]."
                ) from exc
            prepare_fn = prepare_language
            config = PreparationConfig.for_speech(language)
        else:
            config = None
        protected_spans = tuple(
            SimpleNamespace(start=match.start(), end=match.end(), kind="piper_raw_phoneme", source="pipersynth")
            for match in _RAW_BLOCK_RE.finditer(text)
        )
        try:
            kwargs: dict[str, Any] = {"language": language, "protected_spans": protected_spans}
            if config is not None:
                kwargs["config"] = config
            prepared = prepare_fn(text, **kwargs)
        except Exception as exc:
            if isinstance(exc, (TextPreparationError, OptionalDependencyError)):
                raise
            raise TextPreparationError("spokenform could not prepare the supplied text") from exc
        spoken_text = getattr(prepared, "spoken_text", None)
        if not isinstance(spoken_text, str):
            raise TextPreparationError("spokenform returned an invalid prepared text result")
        warnings = tuple(getattr(prepared, "warnings", ()))
        metadata: dict[str, Any] = {"mode": "spokenform", "language": language}
        to_dict = getattr(prepared, "to_dict", None)
        if to_dict is not None:
            metadata["spokenform"] = to_dict()
        return PreparedTextResult(text, spoken_text, metadata, warnings)
