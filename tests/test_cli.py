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


def test_cli_min_age_and_future_year_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "ds"
    out_dir = tmp_path / "out"
    names_dir = data_dir / "namesbystate"
    names_dir.mkdir(parents=True)
    (names_dir / "CA.TXT").write_text(
        "CA,F,2024,Lucy,500\nCA,F,2000,Lucy,200\n", encoding="utf-8"
    )
    csv_path = data_dir / death_probs_filename("F", 2026)
    ages = ",".join(str(a) for a in range(35))
    probs = ",".join("0.0" for _ in range(35))
    csv_path.write_text(
        f"Title\nYear,{ages}\n2024,{probs}\n2026,{probs}\n", encoding="utf-8"
    )

    rc = main(
        [
            "Lucy",
            "--sex",
            "F",
            "--year",
            "2026",
            "--min-age",
            "18",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert (
        "warning: reference year 2026 is after latest birth year 2024" in captured.err
    )
    assert "median age 26 (IQR 26–26, mean 26.0)" in captured.out

    rc_max = main(
        [
            "Lucy",
            "--sex",
            "F",
            "--year",
            "2024",
            "--max-age",
            "10",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
        ]
    )
    out_max = capsys.readouterr().out
    assert rc_max == 0
    assert "median age 0 (IQR 0–0, mean 0.0), expected living 500" in out_max


def test_cli_prefers_national_births_for_us_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "ds"
    out_dir = tmp_path / "out"
    _seed_data_dir(data_dir)
    nat_dir = data_dir / "names"
    nat_dir.mkdir()
    (nat_dir / "yob2025.txt").write_text("John,M,999\n", encoding="utf-8")

    rc_us = main(
        [
            "John",
            "--sex",
            "M",
            "--region",
            "US",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
        ]
    )
    out_us = capsys.readouterr().out
    assert rc_us == 0
    assert "expected living 994" in out_us  # 999 * (1 - 0.5 * 0.01) = 994.005

    rc_ca = main(
        [
            "John",
            "--sex",
            "M",
            "--region",
            "CA",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
        ]
    )
    out_ca = capsys.readouterr().out
    assert rc_ca == 0
    assert "expected living 100" in out_ca  # 100 * 0.995 = 99.5 -> rounds to 100
