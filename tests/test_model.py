"""Unit tests for nameage actuarial survival and age distribution model."""

from __future__ import annotations

import pandas as pd
import pytest

from nameage.model import (
    BirthsIndex,
    compute_survival_curve,
    estimate_age_distribution,
)


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
    assert survival.loc[0] == pytest.approx(0.5 * (1.0 + 0.9))
    assert survival.loc[1] == pytest.approx(0.5 * (0.9 + 0.9 * 0.8))
    assert survival.loc[2] == pytest.approx(0.5 * (0.9 * 0.8 + 0.9 * 0.8 * 0.5))


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
    l5 = 0.9 * 0.95 * 0.98 * 0.97 * 0.96
    expected_s5 = 0.5 * (l5 + l5 * 0.8)
    assert result.distribution.loc[5, "survival_prob"] == pytest.approx(expected_s5)
    assert result.expected_living == pytest.approx(1000 * expected_s5)
    assert result.median_age == 5


def test_mortality_shifts_median_and_aggregates_multiple_states() -> None:
    """Multi-state same-year births sum together; high mortality shifts median."""
    death_probs = _make_death_probs(year=2025, qx_values=[0.0, 0.0, 0.9, 0.0])
    births = _make_births(
        [
            ("CA", "M", 2024, "John", 100),  # age 1
            ("NY", "M", 2024, "John", 150),  # age 1 -> total 250 living
            ("CA", "M", 2022, "John", 500),  # age 3 -> 50 living after 90% q_2
        ]
    )
    us_result = estimate_age_distribution(
        births, death_probs, name="john", sex="M", region="US", reference_year=2025
    )
    # Raw births: 250 at age 1, 500 at age 3 (raw median = 3).
    # After 90% mortality at age 2: 250 living at age 1, 50 at age 3 (median = 1).
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

    # Zero expected living when q_0 == 1.0 and all births are at age >= 1
    lethal_table = _make_death_probs(year=2025, qx_values=[1.0, 1.0])
    age_one_births = _make_births([("CA", "F", 2024, "Ada", 100)])
    with pytest.raises(ValueError, match="Expected living population is zero"):
        estimate_age_distribution(
            age_one_births, lethal_table, name="Ada", sex="F", reference_year=2025
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


def test_posterior_quantiles_and_age_conditioning() -> None:
    """Bimodal distribution exposes PMF/CDF/quantiles and min_age/max_age filtering."""
    death_probs = _make_death_probs(year=2025, qx=0.0, max_age=80)
    births = _make_births(
        [
            ("CA", "F", 2020, "Lucy", 600),  # age 5 (60%)
            ("CA", "F", 1965, "Lucy", 400),  # age 60 (40%)
        ]
    )

    full = estimate_age_distribution(
        births, death_probs, name="Lucy", sex="F", reference_year=2025
    )
    assert full.distribution["prob"].sum() == pytest.approx(1.0)
    assert full.distribution["cdf"].iloc[-1] == pytest.approx(1.0)
    assert full.median_age == 5
    assert full.mode_age == 5
    assert full.mean_age == pytest.approx(0.6 * 5 + 0.4 * 60)
    assert full.iqr == (5, 60)
    assert full.ci90 == (5, 60)

    adults = estimate_age_distribution(
        births, death_probs, name="Lucy", sex="F", reference_year=2025, min_age=18
    )
    assert list(adults.distribution.index)[0] == 18
    assert adults.median_age == 60
    assert adults.mean_age == pytest.approx(60.0)
    assert adults.iqr == (60, 60)
    assert adults.expected_living == pytest.approx(400.0)

    with pytest.raises(ValueError, match="min_age must be non-negative"):
        estimate_age_distribution(
            births, death_probs, name="Lucy", sex="F", reference_year=2025, min_age=-1
        )
    with pytest.raises(ValueError, match="less than or equal to max_age"):
        estimate_age_distribution(
            births,
            death_probs,
            name="Lucy",
            sex="F",
            reference_year=2025,
            min_age=30,
            max_age=20,
        )
    with pytest.raises(ValueError, match="does not overlap mortality table"):
        estimate_age_distribution(
            births, death_probs, name="Lucy", sex="F", reference_year=2025, min_age=90
        )


def test_exact_quantile_boundaries_without_float_drift() -> None:
    """Exact binary-fraction quantiles (0.25, 0.50) are not shifted by float drift."""
    death_probs = _make_death_probs(year=2025, qx=0.0, max_age=10)
    # 7 cohorts of 1 birth (ages 0..6) + 7 births at age 10 -> total=14, cum(6)=7/14=0.5
    rows_half = [("CA", "F", 2025 - age, "Ada", 1) for age in range(7)] + [
        ("CA", "F", 2015, "Ada", 7)
    ]
    res_half = estimate_age_distribution(
        _make_births(rows_half), death_probs, name="Ada", sex="F", reference_year=2025
    )
    assert res_half.median_age == 6
    assert res_half.distribution["cdf"].iloc[-1] == 1.0

    # 7 cohorts of 1 birth (ages 0..6) + 21 births at age 10 -> total=28, cum(6)=0.25
    rows_quarter = [("CA", "F", 2025 - age, "Ada", 1) for age in range(7)] + [
        ("CA", "F", 2015, "Ada", 21)
    ]
    res_quarter = estimate_age_distribution(
        _make_births(rows_quarter),
        death_probs,
        name="Ada",
        sex="F",
        reference_year=2025,
    )
    assert res_quarter.iqr[0] == 6


def test_births_index_matches_dataframe_scan() -> None:
    """BirthsIndex produces identical estimates to linear DataFrame scanning."""
    births = _make_births(
        [
            ("CA", "F", 2025, "Lucy", 100),
            ("NY", "F", 2020, "Lucy", 300),
            ("CA", "M", 2020, "Lucy", 50),
        ]
    )
    births["state"] = births["state"].astype("category")
    births["sex"] = births["sex"].astype("category")
    births["name"] = births["name"].astype("category")

    death_probs = _make_death_probs(year=2025, qx_values=[0.01] * 10)
    index = BirthsIndex.from_dataframe(births)

    for region in ("US", "CA"):
        expected = estimate_age_distribution(
            births, death_probs, name="lucy", sex="F", region=region
        )
        actual = estimate_age_distribution(
            index, death_probs, name="lucy", sex="F", region=region
        )
        assert actual.name == expected.name
        assert actual.median_age == expected.median_age
        assert actual.mean_age == pytest.approx(expected.mean_age)
        assert actual.expected_living == pytest.approx(expected.expected_living)
        pd.testing.assert_frame_equal(actual.distribution, expected.distribution)


def test_births_index_non_categorical_mixed_case_and_lowercase_region() -> None:
    """Non-categorical mixed-case names fold cleanly and lookup normalizes region."""
    births = _make_births(
        [
            ("CA", "F", 2025, "Lucy", 100),
            ("CA", "F", 2020, "lucy", 300),
            ("NY", "F", 2018, "LUCY", 50),
        ]
    )
    death_probs = _make_death_probs(year=2025, qx_values=[0.01] * 10)
    index = BirthsIndex.from_dataframe(births)

    assert len(index.lookup(name="lucy", sex="F", region="ca")) == 2
    assert len(index.lookup(name="lucy", sex="F", region="us")) == 3

    expected = estimate_age_distribution(
        births, death_probs, name="LUCY", sex="F", region="ca"
    )
    actual = estimate_age_distribution(
        index, death_probs, name="LUCY", sex="F", region="ca"
    )
    assert actual.name == "Lucy"
    assert actual.median_age == expected.median_age
    pd.testing.assert_frame_equal(actual.distribution, expected.distribution)


def test_births_index_ignores_null_names_and_handles_empty_frames() -> None:
    """Null names never corrupt neighboring categories or match 'nan'/'none'."""
    cat_births = pd.DataFrame(
        {
            "state": pd.Categorical(["CA", "CA", "CA"]),
            "sex": pd.Categorical(["F", "F", "F"]),
            "birth_year": [2025, 2025, 2025],
            "name": pd.Categorical(["Ada", "Zoe", None], categories=["Ada", "Zoe"]),
            "count": [10, 20, 999],
        }
    )
    cat_idx = BirthsIndex.from_dataframe(cat_births)
    assert int(cat_idx.lookup(name="Zoe", sex="F")["count"].sum()) == 20
    assert cat_idx.lookup(name="nan", sex="F").empty

    # All-null categorical column with zero categories
    all_null_cat = pd.DataFrame(
        {
            "state": pd.Categorical(["CA"]),
            "sex": pd.Categorical(["F"]),
            "birth_year": [2025],
            "name": pd.Categorical([None], categories=[]),
            "count": [50],
        }
    )
    assert BirthsIndex.from_dataframe(all_null_cat).lookup(name="Ada", sex="F").empty

    # Non-categorical null names do not become literal "nan" or "none"
    obj_births = pd.DataFrame(
        {
            "state": ["CA", "CA", "CA"],
            "sex": ["F", "F", "F"],
            "birth_year": [2025, 2025, 2025],
            "name": ["Ada", None, float("nan")],
            "count": [10, 50, 75],
        }
    )
    obj_idx = BirthsIndex.from_dataframe(obj_births)
    assert obj_idx.lookup(name="nan", sex="F").empty
    assert obj_idx.lookup(name="none", sex="F").empty
    assert int(obj_idx.lookup(name="Ada", sex="F")["count"].sum()) == 10

    # Columnless empty DataFrame
    empty_idx = BirthsIndex.from_dataframe(pd.DataFrame())
    assert empty_idx.empty
    death_probs = _make_death_probs(year=2025, qx_values=[0.01] * 5)
    with pytest.raises(ValueError, match="Births dataset is empty"):
        estimate_age_distribution(empty_idx, death_probs, name="Ada", sex="F")


def test_estimate_with_survival_validation_and_missing_latest_year() -> None:
    """Validate precomputed survival and handle latest_birth_year=None."""
    births = _make_births([("CA", "F", 2025, "Ada", 100)])
    death_probs = _make_death_probs(year=2025, qx_values=[0.01] * 5)
    built = BirthsIndex.from_dataframe(births)
    no_latest = BirthsIndex(
        built._frame, built._slices, latest_birth_year=None, regions=built.regions
    )
    curve = compute_survival_curve(death_probs, 2025)
    res = estimate_age_distribution(
        no_latest, death_probs, name="Ada", sex="F", survival=curve
    )
    assert res.reference_year == 2025
    assert res.median_age == 0

    with pytest.raises(ValueError, match="empty"):
        estimate_age_distribution(
            births, death_probs, name="Ada", sex="F", survival=pd.Series(dtype=float)
        )
    with pytest.raises(ValueError, match="contiguous"):
        estimate_age_distribution(
            births,
            death_probs,
            name="Ada",
            sex="F",
            survival=pd.Series([1.0, 0.9], index=[1, 0]),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        estimate_age_distribution(
            births,
            death_probs,
            name="Ada",
            sex="F",
            survival=pd.Series([1.0, float("nan")], index=[0, 1]),
        )
