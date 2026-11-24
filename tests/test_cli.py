"""Unit tests for the nameage CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from nameage.cli import build_parser, main
from nameage.data import death_probs_filename


def _seed_data_dir(data_dir: Path) -> None:
    names_dir = data_dir / "namesbystate"
    names_dir.mkdir(parents=True)
    (names_dir / "CA.TXT").write_text(
        "CA,M,2025,John,100\nCA,M,2024,Paul,80\n", encoding="utf-8"
    )
    csv_path = data_dir / death_probs_filename("M", 2026)
    csv_path.write_text("Title\nYear,0,1,2\n2025,0.01,0.01,0.01\n", encoding="utf-8")


def test_parser_accepts_lowercase_sex_and_validates_region() -> None:
    args = build_parser().parse_args(["john", "--sex", "m", "--region", "ca"])
    assert args.sex == "M"
    assert args.region == "CA"

    with pytest.raises(SystemExit):
        build_parser().parse_args(["john", "--sex", "M", "--region", "California"])


def test_cli_batch_continues_past_missing_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "ds"
    out_dir = tmp_path / "out"
    _seed_data_dir(data_dir)

    rc = main(
        [
            "john",
            "NoSuchName",
            "Paul",
            "--sex",
            "m",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "NoSuchName" in captured.err
    assert (out_dir / "nameage-us-john.png").is_file()
    assert (out_dir / "nameage-us-paul.png").is_file()
    assert "John (US, M, 2025)" in captured.out
    assert "Paul (US, M, 2025)" in captured.out


def test_cli_reports_clean_error_on_missing_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fail_403(*args: object, **kwargs: object) -> None:
        raise RuntimeError("SSA returned HTTP 403")

    monkeypatch.setattr("nameage.data._download_atomic", _fail_403)
    rc = main(["John", "--sex", "M", "--data-dir", str(tmp_path / "empty")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "nameage: error: SSA returned HTTP 403" in captured.err
