from pipersynth.asset_progress import AssetProgressEvent, ConsoleAssetProgress


def test_progress_event_is_stable_and_console_reporter_is_opt_in(capsys) -> None:
    event = AssetProgressEvent("cache-hit", "voice", "Using cache")
    ConsoleAssetProgress()(event)
    assert capsys.readouterr().out == "cache-hit [voice]: Using cache\n"
