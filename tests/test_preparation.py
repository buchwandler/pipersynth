import sys
from types import SimpleNamespace

import pytest

from pipersynth.errors import OptionalDependencyError, TextPreparationError
from pipersynth.preparation import IdentityTextPreparer, SpokenformTextPreparer


def test_identity_preparation_preserves_source_and_text() -> None:
    result = IdentityTextPreparer().prepare("Hello", language="en")
    assert result.source_text == "Hello"
    assert result.prepared_text == "Hello"
    assert result.metadata["mode"] == "identity"


def test_spokenform_adapter_preserves_raw_phoneme_blocks() -> None:
    seen = []

    def fake_prepare(text, **kwargs):
        seen.extend(kwargs["protected_spans"])
        return SimpleNamespace(spoken_text="spoken " + text, warnings=("warning",), to_dict=lambda: {"ok": True})

    result = SpokenformTextPreparer(fake_prepare).prepare("Say [[ h e l l o ]]", language="en")
    assert result.source_text == "Say [[ h e l l o ]]"
    assert result.spoken_text == "spoken Say [[ h e l l o ]]"
    assert seen[0].start == 4
    assert seen[0].end == 19
    assert result.warnings == ("warning",)
    assert result.metadata["spokenform"] == {"ok": True}


def test_spokenform_requires_language() -> None:
    with pytest.raises(TextPreparationError, match="language"):
        SpokenformTextPreparer(lambda text, **kwargs: None).prepare("text", language=None)


def test_missing_spokenform_dependency_is_actionable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "spokenform", None)
    with pytest.raises(OptionalDependencyError, match=r"pipersynth\[spokenform\]"):
        SpokenformTextPreparer().prepare("text", language="en")
