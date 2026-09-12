from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np


class InferenceInput(Protocol):
    name: str


class InferenceOutput(Protocol):
    name: str


class InferenceSession(Protocol):
    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]) -> Sequence[Any]: ...

    def get_inputs(self) -> Sequence[InferenceInput]: ...

    def get_outputs(self) -> Sequence[InferenceOutput]: ...

    def get_providers(self) -> Sequence[str]: ...


class SessionFactory(Protocol):
    def __call__(self, model_path: str, **kwargs: Any) -> InferenceSession: ...
