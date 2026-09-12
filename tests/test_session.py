import sys
from pathlib import Path

import pytest

from pipersynth.errors import OptionalDependencyError, UnsupportedModelError
from pipersynth.session import OnnxSessionManager, available_providers


class Item:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, inputs=("input", "input_lengths", "scales")) -> None:
        self.inputs = [Item(name) for name in inputs]
        self.outputs = [Item("audio")]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "voice.onnx"
    path.write_bytes(b"model")
    return path


def test_explicit_providers_and_options_are_preserved(tmp_path: Path) -> None:
    calls = []

    def factory(path, **kwargs):
        calls.append((path, kwargs))
        return FakeSession()

    manager = OnnxSessionManager(
        model_file(tmp_path),
        providers=["CUDAExecutionProvider", ("CPUExecutionProvider", {"arena_extend_strategy": "kNextPowerOfTwo"})],
        session_options="options",
        session_factory=factory,
    )
    manager.create()
    assert calls == [
        (
            str(model_file(tmp_path)),
            {
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [{}, {"arena_extend_strategy": "kNextPowerOfTwo"}],
                "sess_options": "options",
            },
        )
    ]
    assert manager.providers_requested == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert manager.model_inputs == ("input", "input_lengths", "scales")


def test_default_provider_is_cpu(tmp_path: Path) -> None:
    manager = OnnxSessionManager(model_file(tmp_path), session_factory=lambda path, **kwargs: FakeSession())
    assert manager.providers_requested == ("CPUExecutionProvider",)


def test_required_inputs_and_extra_inputs_are_rejected(tmp_path: Path) -> None:
    missing = OnnxSessionManager(
        model_file(tmp_path),
        session_factory=lambda path, **kwargs: FakeSession(("input",)),
    )
    with pytest.raises(UnsupportedModelError, match="input_lengths"):
        missing.create()

    extra = OnnxSessionManager(
        model_file(tmp_path),
        session_factory=lambda path, **kwargs: FakeSession(("input", "input_lengths", "scales", "extra")),
    )
    with pytest.raises(UnsupportedModelError, match="extra"):
        extra.create()


def test_sid_is_required_for_multispeaker_contract(tmp_path: Path) -> None:
    manager = OnnxSessionManager(
        model_file(tmp_path),
        session_factory=lambda path, **kwargs: FakeSession(),
    )
    with pytest.raises(UnsupportedModelError, match="sid"):
        manager.create(require_sid=True)


def test_manager_reuses_and_closes_session(tmp_path: Path) -> None:
    sessions = []

    def factory(path, **kwargs):
        session = FakeSession()
        sessions.append(session)
        return session

    manager = OnnxSessionManager(model_file(tmp_path), session_factory=factory)
    assert manager.create() is manager.create()
    assert len(sessions) == 1
    manager.close()
    assert manager.providers_active == ()


def test_missing_runtime_error_is_actionable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(OptionalDependencyError, match=r"pipersynth\[(cpu|gpu)\]"):
        available_providers()
