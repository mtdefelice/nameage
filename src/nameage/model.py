"""Actuarial survival and living-age distribution model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nameage.data import Sex


@dataclass(frozen=True)
class NameAgeResult:
    """Estimated living age distribution for a name/sex/region query."""

    name: str
    sex: Sex
    region: str
    reference_year: int
    median_age: int
    expected_living: float
    distribution: pd.DataFrame


def compute_survival_curve(death_probs: pd.DataFrame, year: int) -> pd.Series:
    """Compute cumulative survival probability by age from a period mortality table.

    Uses the single calendar-year row ``year`` (period life-table approximation)
    across the full contiguous integer age range ``0..max_age`` before any join
    with birth counts, ensuring missing birth years are never treated as 0%
    mortality. Value at age ``a`` represents survival through the end of age
    ``a``: ``prod_{x=0..a} (1 - q_x)``.
    """
    if death_probs.empty:
        raise ValueError("Mortality table is empty.")

    if not death_probs.index.is_unique:
        raise ValueError("Mortality table contains duplicate calendar years.")

    if year not in death_probs.index:
        min_year = int(death_probs.index.min())
        max_year = int(death_probs.index.max())
        raise ValueError(
            f"Reference year {year} not in mortality table ({min_year}–{max_year})."
        )

    qx = death_probs.loc[year]
    try:
        qx.index = qx.index.astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Mortality table age columns must be integers.") from exc

    qx = qx.sort_index().astype("float64")
    expected_ages = pd.RangeIndex(start=0, stop=len(qx))
    if not qx.index.equals(expected_ages):
        raise ValueError(f"Mortality table ages must be contiguous 0..{len(qx) - 1}.")

    if qx.isna().any() or ((qx < 0.0) | (qx > 1.0)).any():
        raise ValueError(
            f"Mortality probabilities for {year} must be finite values in [0, 1]."
        )

    survival = (1.0 - qx).cumprod()
    survival.index.name = "age"
    survival.name = "survival_prob"
    return survival


def estimate_age_distribution(
    births: pd.DataFrame,
    death_probs: pd.DataFrame,
    *,
    name: str,
    sex: Sex,
    region: str = "US",
    reference_year: int | None = None,
) -> NameAgeResult:
    """Estimate born counts, expected living counts, and median age for a name.

    Args:
        births: DataFrame with ``state``, ``sex``, ``birth_year``, ``name``, ``count``.
        death_probs: Period death-probability table indexed by year with age columns.
        name: First name (case-insensitive match against SSA title-cased names).
        sex: ``"F"`` or ``"M"``.
        region: ``"US"`` for all states combined, or a 2-letter U.S. state/DC code.
        reference_year: Calendar year for age calculation and mortality lookup.
            Defaults to the latest ``birth_year`` in ``births``.
    """
    if sex not in ("F", "M"):
        raise ValueError(f"sex must be 'F' or 'M', got {sex!r}")

    if births.empty:
        raise ValueError("Births dataset is empty.")

    normalized_region = region.upper()
    ref_year = (
        reference_year
        if reference_year is not None
        else int(births["birth_year"].max())
    )

    survival = compute_survival_curve(death_probs, ref_year)

    mask = (births["sex"] == sex) & (births["name"].str.casefold() == name.casefold())
    if normalized_region != "US":
        mask &= births["state"] == normalized_region

    subset = births.loc[mask, ["name", "birth_year", "count"]]
    if subset.empty:
        raise ValueError(
            f"No birth records found for name={name!r}, sex={sex!r}, "
            f"region={normalized_region!r}."
        )

    canonical_name = str(subset["name"].iloc[0])
    ages = ref_year - subset["birth_year"].astype("int64")
    born_by_age = (
        subset.assign(age=ages)
        .loc[lambda frame: frame["age"] >= 0]
        .groupby("age", sort=True)["count"]
        .sum()
        .rename("born")
    )

    if born_by_age.empty:
        raise ValueError(
            f"No birth records on or before reference_year={ref_year} for "
            f"name={canonical_name!r}, sex={sex!r}, region={normalized_region!r}."
        )

    if born_by_age.index.intersection(survival.index).empty:
        raise ValueError(
            f"No overlapping ages between births and mortality table for "
            f"{canonical_name!r}."
        )

    # Reindex onto the full contiguous survival curve so zero-birth ages remain 0.
    dist = pd.DataFrame(
        {
            "born": born_by_age.reindex(survival.index, fill_value=0),
            "survival_prob": survival,
        },
        index=survival.index,
    )
    dist["expected_living"] = dist["born"] * dist["survival_prob"]
    dist["birth_year"] = ref_year - dist.index
    dist["cum_living"] = dist["expected_living"].cumsum()

    total_living = float(dist["cum_living"].iloc[-1])
    if total_living <= 0.0:
        raise ValueError(f"Expected living population is zero for {canonical_name!r}.")

    median_age = int(dist.loc[dist["cum_living"] >= total_living / 2.0].index[0])

    return NameAgeResult(
        name=canonical_name,
        sex=sex,
        region=normalized_region,
        reference_year=ref_year,
        median_age=median_age,
        expected_living=total_living,
        distribution=dist,
    )
