"""Unit tests for SSA data loading, caching, and extraction safety."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nameage.data import (
    _extract_zip_safely,
    death_probs_filename,
    death_probs_url,
    ensure_ssa_data,
    load_births,
    load_death_probabilities,
)


def _write_mortality_csv(path: Path, years: tuple[int, ...] = (2025,)) -> None:
    header = "Year,0,1,2\n"
    rows = "".join(f"{yr},0.01,0.02,0.03\n" for yr in years)
    path.write_text(f"Title line\n{header}{rows}", encoding="utf-8")


def test_load_births_preserves_categories_and_literal_none(tmp_path: Path) -> None:
    names_dir = tmp_path / "namesbystate"
    names_dir.mkdir()
    (names_dir / "CA.TXT").write_text(
        "CA,F,2025,Ada,10\nCA,F,2025,None,7\n", encoding="utf-8"
    )
    (names_dir / "NY.TXT").write_text("NY,M,2025,John,20\n", encoding="utf-8")

    # Accepts str path as well as Path
    df = load_births(str(tmp_path))
    assert len(df) == 3
    assert isinstance(df["state"].dtype, pd.CategoricalDtype)
    assert isinstance(df["sex"].dtype, pd.CategoricalDtype)
    assert isinstance(df["name"].dtype, pd.CategoricalDtype)
    assert list(df["state"].cat.categories) == ["CA", "NY"]
    # Literal "None" must not be coerced to NA
    assert "None" in set(df["name"])
    assert not df["name"].isna().any()


def test_load_death_probabilities_and_url_routing(tmp_path: Path) -> None:
    csv_path = tmp_path / death_probs_filename("F", 2026)
    _write_mortality_csv(csv_path, years=(2024, 2025))

    table = load_death_probabilities(str(tmp_path), "F", tr_year=2026)
    assert list(table.index) == [2024, 2025]
    assert list(table.columns) == [0, 1, 2]
    assert table.loc[2025, 1] == pytest.approx(0.02)

    assert "/CY/" in death_probs_url("F", 2026)
    assert "/2024/" in death_probs_url("M", 2024)


def test_ensure_ssa_data_skips_network_when_extracted_files_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names_dir = tmp_path / "namesbystate"
    names_dir.mkdir()
    (names_dir / "CA.TXT").write_text("CA,F,2025,Ada,10\n", encoding="utf-8")
    _write_mortality_csv(tmp_path / death_probs_filename("F", 2026))

    def _forbid_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("Should not attempt network download")

    monkeypatch.setattr("nameage.data._download_atomic", _forbid_download)
    # Neither namesbystate.zip nor Male CSV is present, yet sex="F" must succeed offline
    ensure_ssa_data(tmp_path, sex="F", tr_year=2026, refresh=False)


def test_extract_zip_safely_rejects_zip_slip(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "pwned")

    dest = tmp_path / "namesbystate"
    with pytest.raises(ValueError, match="Unsafe zip member path"):
        _extract_zip_safely(bad_zip, dest)
    assert not dest.exists()
    assert not (tmp_path / "escape.txt").exists()
