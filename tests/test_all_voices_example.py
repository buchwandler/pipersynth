from __future__ import annotations

import json
from pathlib import Path

import pytest

import examples.all_voices as example
from pipersynth.assets import VoiceMetadata


def _voices() -> tuple[VoiceMetadata, ...]:
    return (
        VoiceMetadata(
            id="en_US-test-medium",
            name="English Test",
            language_code="en_US",
            language_family="en",
            region="US",
            quality="medium",
            num_speakers=2,
            speaker_id_map={"alice": 0, "bob": 1},
            aliases=(),
            source_revision="revision",
        ),
        VoiceMetadata(
            id="bg_BG-test-medium",
            name="Bulgarian Test",
            language_code="bg_BG",
            language_family="bg",
            region="BG",
            quality="medium",
            num_speakers=1,
            speaker_id_map={},
            aliases=(),
            source_revision="revision",
        ),
    )


def _probe(locale: str) -> example.SpokenformProbe:
    if locale == "en_US":
        return example.SpokenformProbe("supported", "en-us", None)
    return example.SpokenformProbe("unsupported", None, f"unsupported {locale}")


def test_build_catalog_preserves_every_catalog_voice() -> None:
    catalog = example.build_catalog(_voices())

    assert [item.model_id for item in catalog.voices] == [
        "bg_BG-test-medium",
        "en_US-test-medium",
    ]
    assert len(catalog.voices) == 2


def test_build_catalog_expands_every_numeric_speaker() -> None:
    catalog = example.build_catalog(_voices())

    assert [(item.model_id, item.speaker_id) for item in catalog.identities] == [
        ("bg_BG-test-medium", 0),
        ("en_US-test-medium", 0),
        ("en_US-test-medium", 1),
    ]
    assert catalog.identities[-1].speaker_name == "bob"


def test_single_speaker_uses_speaker_zero() -> None:
    catalog = example.build_catalog(_voices())

    assert catalog.identities[0].speaker_id == 0
    assert catalog.identities[0].speaker_name is None


def test_calibration_keys_match_piper_voice_identity_format() -> None:
    catalog = example.build_catalog(_voices())

    assert catalog.identities[-1].calibration_key == "piper:en_US-test-medium:medium:speaker-1"


def test_language_summary_counts_catalog_voices_and_speaker_identities() -> None:
    catalog = example.with_language_coverage(example.build_catalog(_voices()), _probe)

    assert [
        (item.locale, item.catalog_voice_count, item.speaker_identity_count)
        for item in catalog.languages
    ] == [
        ("bg_BG", 1, 1),
        ("en_US", 1, 2),
    ]
    assert catalog.languages[0].spokenform_status == "unsupported"


def test_language_summary_is_deterministically_sorted() -> None:
    catalog = example.with_language_coverage(example.build_catalog(_voices()), _probe)

    assert [item.locale for item in catalog.languages] == ["bg_BG", "en_US"]


def test_supported_and_unsupported_probe_results_are_recorded() -> None:
    assert example.probe_spokenform_locale("definitely-not-a-locale").status in {
        "supported",
        "unsupported",
        "unavailable",
    }
    catalog = example.with_language_coverage(example.build_catalog(_voices()), _probe)
    assert catalog.languages[1].normalized_language == "en-us"
    assert catalog.languages[0].error == "unsupported bg_BG"


def test_missing_spokenform_does_not_break_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(_: str) -> example.SpokenformProbe:
        return example.SpokenformProbe("unavailable", None, "missing")

    catalog = example.with_language_coverage(example.build_catalog(_voices()), missing_import)

    assert all(item.spokenform_status == "unavailable" for item in catalog.languages)


def test_inventory_outputs_contain_all_catalog_voices_and_sorted_missing_text(
    tmp_path: Path,
) -> None:
    catalog = example.with_language_coverage(example.build_catalog(_voices()), _probe)

    outputs = example.write_inventory_outputs(catalog, tmp_path)

    inventory = json.loads(outputs["inventory"].read_text())
    languages = json.loads(outputs["languages"].read_text())
    assert len(inventory["voices"]) == 2
    assert len(inventory["identities"]) == 3
    assert [item["locale"] for item in languages["languages"]] == ["bg_BG", "en_US"]
    assert outputs["missing_text"].read_text() == "bg_BG\n"


def test_list_only_prints_languages_before_voice_table_and_never_constructs_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeManager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_voices(self, **_kwargs: object) -> tuple[VoiceMetadata, ...]:
            return _voices()

        def cache_info(self) -> object:
            return type("Cache", (), {"directory": Path("/cache")})()

    class SentinelPipeline:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("pipeline must not be constructed")

    monkeypatch.setattr(example, "VoiceAssetManager", FakeManager)
    monkeypatch.setattr(example, "PiperPipeline", SentinelPipeline)
    monkeypatch.setattr(
        example,
        "with_language_coverage",
        lambda catalog: example.ShowcaseCatalog(
            catalog.voices, catalog.identities, example.build_language_coverage(catalog, _probe)
        ),
    )
    monkeypatch.setattr(example, "artefact_dir", lambda: tmp_path)

    assert example.main(["--list-only"]) == 0
    output = capsys.readouterr().out
    assert output.index("Piper locales") < output.index("No.  Voice/model")
    assert (tmp_path / "all_voices_inventory.json").exists()


def test_normal_mode_stops_before_download_when_languages_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeManager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_voices(self, **_kwargs: object) -> tuple[VoiceMetadata, ...]:
            return _voices()

        def cache_info(self) -> object:
            return type("Cache", (), {"directory": Path("/cache")})()

    monkeypatch.setattr(example, "VoiceAssetManager", FakeManager)
    monkeypatch.setattr(
        example, "PiperPipeline", lambda *_args, **_kwargs: pytest.fail("download started")
    )
    monkeypatch.setattr(
        example,
        "with_language_coverage",
        lambda catalog: example.ShowcaseCatalog(
            catalog.voices, catalog.identities, example.build_language_coverage(catalog, _probe)
        ),
    )
    monkeypatch.setattr(example, "artefact_dir", lambda: tmp_path)

    assert example.main([]) == 2
