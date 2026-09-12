from __future__ import annotations

from pathlib import Path

import pipersynth.__main__ as cli


def test_new_parser_has_catalog_commands() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["voices", "list", "--language", "en"])
    assert args.command == "voices"
    assert args.voice_command == "list"
    args = parser.parse_args(["speak", "--voice", "voice", "Hello"])
    assert args.voice == "voice"


def test_speak_delegates_to_convenience(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []

    def fake_synthesize(text, output, **kwargs):
        calls.append((text, output, kwargs))
        return Path(output)

    monkeypatch.setattr(cli, "synthesize_to_wav", fake_synthesize)
    cli.main(["speak", "--voice", "voice", "Hello", "-o", str(tmp_path / "out.wav")])
    assert calls[0][0] == "Hello"
    assert calls[0][2]["voice"] == "voice"
    assert str(tmp_path / "out.wav") in capsys.readouterr().out
