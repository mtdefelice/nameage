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
    mean_age: float
    mode_age: int
    iqr: tuple[int, int]
    ci90: tuple[int, int]
    expected_living: float
    distribution: pd.DataFrame


def _quantile_age(cdf: pd.Series, q: float) -> int:
    """Return the smallest integer age whose cumulative probability reaches ``q``."""
    matches = cdf.loc[cdf >= q]
    if matches.empty:
        return int(cdf.index[-1])
    return int(matches.index[0])


def compute_survival_curve(death_probs: pd.DataFrame, year: int) -> pd.Series:
    """Compute mid-interval survival probabilities from a period mortality table.

    Uses the single calendar-year row ``year`` (period life-table approximation)
    across the full contiguous integer age range ``0..max_age`` before any join
    with birth counts, ensuring missing birth years are never treated as 0%
    mortality.

    Under uniform distribution of births within each birth year and deaths
    within each year of age, the expected fraction of the cohort born in
    ``year - a`` still alive at the end of ``year`` is the stationary life-table
    ratio ``L_a / l_0 = 0.5 * (l_a + l_{a+1}) / l_0``, where
    ``l_0 = 1`` and ``l_{a+1} = prod_{x=0..a} (1 - q_x)``.
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

    l_next = (1.0 - qx).cumprod()
    l_start = l_next.shift(1, fill_value=1.0)
    survival = 0.5 * (l_start + l_next)
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
    min_age: int | None = None,
    max_age: int | None = None,
) -> NameAgeResult:
    """Estimate born counts, expected living counts, and age quantiles for a name.

    Args:
        births: DataFrame with ``state``, ``sex``, ``birth_year``, ``name``, ``count``.
        death_probs: Period death-probability table indexed by year with age columns.
        name: First name (case-insensitive match against SSA title-cased names).
        sex: ``"F"`` or ``"M"``.
        region: ``"US"`` for all states combined, or a 2-letter U.S. state/DC code.
        reference_year: Calendar year for age calculation and mortality lookup.
            Defaults to the latest ``birth_year`` in ``births``.
        min_age: Optional inclusive lower age bound for conditioning the posterior
            (e.g., ``18`` for adults).
        max_age: Optional inclusive upper age bound for conditioning the posterior.
    """
    if sex not in ("F", "M"):
        raise ValueError(f"sex must be 'F' or 'M', got {sex!r}")

    if min_age is not None and min_age < 0:
        raise ValueError(f"min_age must be non-negative, got {min_age!r}")
    if max_age is not None and max_age < 0:
        raise ValueError(f"max_age must be non-negative, got {max_age!r}")
    if min_age is not None and max_age is not None and min_age > max_age:
        raise ValueError(
            f"min_age ({min_age}) must be less than or equal to max_age ({max_age})."
        )

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
    if min_age is not None or max_age is not None:
        lower = min_age if min_age is not None else int(dist.index.min())
        upper = max_age if max_age is not None else int(dist.index.max())
        dist = dist.loc[(dist.index >= lower) & (dist.index <= upper)]
        if dist.empty:
            raise ValueError(
                f"Age range [{lower}, {upper}] does not overlap mortality table "
                f"({int(survival.index.min())}–{int(survival.index.max())})."
            )

    dist["expected_living"] = dist["born"] * dist["survival_prob"]
    dist["birth_year"] = ref_year - dist.index
    dist["cum_living"] = dist["expected_living"].cumsum()

    total_living = float(dist["cum_living"].iloc[-1])
    if total_living <= 0.0:
        raise ValueError(f"Expected living population is zero for {canonical_name!r}.")

    dist["prob"] = dist["expected_living"] / total_living
    dist["cdf"] = dist["cum_living"] / total_living

    median_age = _quantile_age(dist["cdf"], 0.50)
    mean_age = float((dist.index.to_series() * dist["prob"]).sum())
    mode_age = int(dist["expected_living"].idxmax())
    iqr = (_quantile_age(dist["cdf"], 0.25), _quantile_age(dist["cdf"], 0.75))
    ci90 = (_quantile_age(dist["cdf"], 0.05), _quantile_age(dist["cdf"], 0.95))

    return NameAgeResult(
        name=canonical_name,
        sex=sex,
        region=normalized_region,
        reference_year=ref_year,
        median_age=median_age,
        mean_age=mean_age,
        mode_age=mode_age,
        iqr=iqr,
        ci90=ci90,
        expected_living=total_living,
        distribution=dist,
    )
