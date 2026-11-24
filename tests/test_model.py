"""Unit tests for nameage actuarial survival and age distribution model."""

from __future__ import annotations

import pandas as pd
import pytest

from nameage.model import compute_survival_curve, estimate_age_distribution


def _make_death_probs(
    year: int = 2025,
    qx_values: list[float] | None = None,
    *,
    qx: float = 0.1,
    max_age: int = 10,
) -> pd.DataFrame:
    """Create a synthetic mortality table for ages 0..max_age."""
    values = qx_values if qx_values is not None else [qx] * (max_age + 1)
    ages = list(range(len(values)))
    table = pd.DataFrame([values], index=[year], columns=ages)
    table.index.name = "Year"
    return table


def _make_births(rows: list[tuple[str, str, int, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["state", "sex", "birth_year", "name", "count"])


def test_compute_survival_curve_non_uniform_unsorted_columns() -> None:
    """Survival curve sorts unsorted age columns and multiplies non-uniform q_x."""
    table = pd.DataFrame([[0.2, 0.5, 0.1]], index=[2025], columns=[1, 2, 0])
    table.index.name = "Year"

    survival = compute_survival_curve(table, 2025)
    assert survival.index.name == "age"
    assert survival.name == "survival_prob"
    assert list(survival.index) == [0, 1, 2]
    assert survival.loc[0] == pytest.approx(0.9)
    assert survival.loc[1] == pytest.approx(0.9 * 0.8)
    assert survival.loc[2] == pytest.approx(0.9 * 0.8 * 0.5)


def test_survival_curve_preserves_zero_birth_ages_and_true_mortality() -> None:
    """Missing birth years stay in distribution with born=0 and keep cumulative q_x."""
    death_probs = _make_death_probs(
        year=2025, qx_values=[0.1, 0.05, 0.02, 0.03, 0.04, 0.2]
    )
    births = _make_births([("CA", "F", 2020, "Ada", 1000)])

    result = estimate_age_distribution(
        births, death_probs, name="ada", sex="F", region="US", reference_year=2025
    )

    assert result.name == "Ada"
    assert list(result.distribution.index) == [0, 1, 2, 3, 4, 5]
    assert list(result.distribution["born"]) == [0, 0, 0, 0, 0, 1000]
    expected_s5 = 0.9 * 0.95 * 0.98 * 0.97 * 0.96 * 0.8
    assert result.distribution.loc[5, "survival_prob"] == pytest.approx(expected_s5)
    assert result.expected_living == pytest.approx(1000 * expected_s5)
    assert result.median_age == 5


def test_mortality_shifts_median_and_aggregates_multiple_states() -> None:
    """Multi-state same-year births sum together; high mortality shifts median."""
    death_probs = _make_death_probs(year=2025, qx_values=[0.0, 0.0, 0.0, 0.9])
    births = _make_births(
        [
            ("CA", "M", 2024, "John", 100),  # age 1
            ("NY", "M", 2024, "John", 150),  # age 1 -> total 250 living
            ("CA", "M", 2022, "John", 500),  # age 3 -> 50 living after 90% q_3
        ]
    )
    us_result = estimate_age_distribution(
        births, death_probs, name="john", sex="M", region="US", reference_year=2025
    )
    # Raw births: 250 at age 1, 500 at age 3 (raw median = 3).
    # After 90% mortality at age 3: 250 living at age 1, 50 at age 3 (median = 1).
    assert us_result.name == "John"
    assert us_result.distribution.loc[1, "born"] == 250
    assert us_result.expected_living == pytest.approx(300.0)
    assert us_result.median_age == 1


def test_default_reference_year_and_future_birth_filtering() -> None:
    """Default reference_year uses max(birth_year) and filters future births."""
    death_probs = _make_death_probs(year=2024, qx_values=[0.0, 0.0, 0.0, 0.0])
    births = _make_births(
        [
            ("CA", "M", 2024, "John", 100),  # age 0 in 2024
            ("NY", "M", 2024, "John", 200),  # age 0 in 2024 -> 300 total
            ("CA", "M", 2022, "John", 200),  # age 2 in 2024
        ]
    )

    result = estimate_age_distribution(births, death_probs, name="John", sex="M")
    assert result.reference_year == 2024
    assert result.distribution.loc[0, "born"] == 300
    assert result.expected_living == pytest.approx(500.0)
    assert result.median_age == 0


def test_region_filtering_and_error_branches() -> None:
    death_probs = _make_death_probs(year=2025, qx_values=[0.0, 0.0, 0.0, 0.0])
    births = _make_births([("CA", "F", 2025, "Ada", 100)])

    with pytest.raises(ValueError, match="No birth records found"):
        estimate_age_distribution(
            births, death_probs, name="Ada", sex="F", region="NY", reference_year=2025
        )

    # All births after reference_year
    future_births = _make_births([("CA", "F", 2026, "Ada", 100)])
    with pytest.raises(ValueError, match="No birth records on or before"):
        estimate_age_distribution(
            future_births, death_probs, name="Ada", sex="F", reference_year=2025
        )

    # All births older than mortality table max age
    old_births = _make_births([("CA", "F", 1900, "Ada", 100)])
    with pytest.raises(ValueError, match="No overlapping ages"):
        estimate_age_distribution(
            old_births, death_probs, name="Ada", sex="F", reference_year=2025
        )

    # Zero expected living when q_0 == 1.0
    lethal_table = _make_death_probs(year=2025, qx_values=[1.0, 1.0])
    with pytest.raises(ValueError, match="Expected living population is zero"):
        estimate_age_distribution(
            births, lethal_table, name="Ada", sex="F", reference_year=2025
        )


def test_compute_survival_curve_validations() -> None:
    empty_table = pd.DataFrame()
    with pytest.raises(ValueError, match="Mortality table is empty"):
        compute_survival_curve(empty_table, 2025)

    dup_table = pd.DataFrame(
        [[0.1, 0.1], [0.2, 0.2]], index=[2025, 2025], columns=[0, 1]
    )
    with pytest.raises(ValueError, match="duplicate calendar years"):
        compute_survival_curve(dup_table, 2025)

    gapped_table = pd.DataFrame([[0.1, 0.1]], index=[2025], columns=[0, 2])
    with pytest.raises(ValueError, match="contiguous"):
        compute_survival_curve(gapped_table, 2025)

    nan_table = pd.DataFrame([[0.1, float("nan")]], index=[2025], columns=[0, 1])
    with pytest.raises(ValueError, match="finite values"):
        compute_survival_curve(nan_table, 2025)

    with pytest.raises(ValueError, match="Reference year 2030 not in mortality table"):
        compute_survival_curve(_make_death_probs(year=2025), 2030)

    with pytest.raises(ValueError, match="Births dataset is empty"):
        estimate_age_distribution(
            _make_births([]), _make_death_probs(year=2025), name="Ada", sex="F"
        )

    with pytest.raises(ValueError, match="sex must be 'F' or 'M'"):
        estimate_age_distribution(
            _make_births([("CA", "F", 2020, "Ada", 100)]),
            _make_death_probs(year=2025),
            name="Ada",
            sex="X",  # type: ignore[arg-type]
            reference_year=2025,
        )
