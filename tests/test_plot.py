"""Tests for Matplotlib chart rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from nameage.model import estimate_age_distribution
from nameage.plot import plot_age_distribution, render_age_distribution


def _sample_result(
    rows: list[tuple[str, str, int, str, int]],
    *,
    max_age: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ages = list(range(max_age + 1))
    death_probs = pd.DataFrame([[0.01] * len(ages)], index=[2025], columns=ages)
    death_probs.index.name = "Year"
    births = pd.DataFrame(rows, columns=["state", "sex", "birth_year", "name", "count"])
    return births, death_probs


def test_plot_writes_pdf_png_svg_in_nested_dir(tmp_path: Path) -> None:
    births, death_probs = _sample_result(
        [
            ("CA", "F", 2025, "Lucy", 200),
            ("CA", "F", 2020, "Lucy", 400),
            ("CA", "F", 2015, "Lucy", 100),
        ]
    )
    result = estimate_age_distribution(
        births, death_probs, name="Lucy", sex="F", region="CA", reference_year=2025
    )

    backend_before = matplotlib.get_backend()
    for ext in ("pdf", "png", "svg"):
        out = plot_age_distribution(result, tmp_path / "nested" / f"lucy.{ext}")
        assert out.is_file()
        assert out.stat().st_size > 0
    assert matplotlib.get_backend() == backend_before


def test_plot_boundary_median_ages(tmp_path: Path) -> None:
    """Single-age births at age 0 and median_age > max_age render cleanly."""
    births_zero, death_probs = _sample_result([("CA", "F", 2025, "Ada", 500)])
    res_zero = estimate_age_distribution(
        births_zero, death_probs, name="Ada", sex="F", reference_year=2025
    )
    assert res_zero.median_age == 0
    out0 = plot_age_distribution(res_zero, tmp_path / "ada_zero.png")
    assert out0.stat().st_size > 0

    births_old, death_probs_old = _sample_result(
        [("CA", "F", 1920, "Ethel", 500)], max_age=110
    )
    res_old = estimate_age_distribution(
        births_old, death_probs_old, name="Ethel", sex="F", reference_year=2025
    )
    assert res_old.median_age == 105
    out_old = plot_age_distribution(res_old, tmp_path / "ethel_old.png", max_age=100)
    assert out_old.stat().st_size > 0


def test_plot_rejects_non_positive_max_age(tmp_path: Path) -> None:
    births, death_probs = _sample_result([("CA", "F", 2025, "Ada", 500)])
    res = estimate_age_distribution(
        births, death_probs, name="Ada", sex="F", reference_year=2025
    )
    with pytest.raises(ValueError, match="max_age must be positive"):
        plot_age_distribution(res, tmp_path / "bad.png", max_age=0)


def test_render_age_distribution_bytes_and_validation() -> None:
    births, death_probs = _sample_result([("CA", "F", 2025, "Ada", 500)])
    res = estimate_age_distribution(
        births, death_probs, name="Ada", sex="F", reference_year=2025
    )
    png = render_age_distribution(res, fmt="png", dpi=100)
    svg = render_age_distribution(res, fmt="svg")
    assert png.startswith(b"\x89PNG")
    assert b"<svg" in svg

    with pytest.raises(ValueError, match="fmt must be"):
        render_age_distribution(res, fmt="bmp")  # type: ignore[arg-type]
