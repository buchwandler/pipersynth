import subprocess
import sys


def test_package_import_does_not_import_onnxruntime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pipersynth; assert 'onnxruntime' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_package_uses_utterplan_identity() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, pipersynth, utterplan; "
                "assert importlib.util.find_spec('ttsplan') is None; "
                "assert pipersynth.UtterancePlan is utterplan.UtterancePlan; "
                "assert pipersynth.UtterancePlanner is utterplan.UtterancePlanner; "
                "assert not hasattr(pipersynth, 'TTS' + 'Plan'); "
                "assert not hasattr(pipersynth, 'TTS' + 'Planner')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_piper_runtime_imports_in_project_sources() -> None:
    for path in ("pipersynth", "tests"):
        output = subprocess.run(
            ["grep", "-RE", r"(from|import)[[:space:]]+piper([[:space:]]|$)", path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert output.returncode == 1, output.stdout
