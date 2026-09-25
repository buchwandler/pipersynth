from __future__ import annotations

from examples.run_all import example_paths


def test_long_text_example_is_registered_with_the_runner() -> None:
    assert any(path.name == "long_text.py" for path in example_paths())
