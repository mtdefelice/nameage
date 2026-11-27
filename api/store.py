"""In-memory indexed dataset store for the nameage web service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nameage.data import (
    DEFAULT_TR_YEAR,
    Sex,
    has_national_births,
    load_births,
    load_death_probabilities,
    load_national_births,
)
from nameage.model import (
    BirthsIndex,
    NameAgeResult,
    compute_survival_curve,
    estimate_age_distribution,
)


class NameNotFoundError(LookupError):
    """Raised when a requested (name, sex, region) has no SSA birth records."""


class InvalidQueryError(ValueError):
    """Raised when query parameters are outside supported dataset bounds."""


@dataclass(frozen=True)
class NameAgeStore:
    """Read-only, pre-indexed SSA birth and mortality tables."""

    state_births: BirthsIndex
    national_births: BirthsIndex | None
    death_probs: dict[Sex, pd.DataFrame]
    survival_curves: dict[tuple[Sex, int], pd.Series]
    tr_year: int

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path | str,
        *,
        tr_year: int = DEFAULT_TR_YEAR,
    ) -> NameAgeStore:
        """Load and index all SSA datasets from disk (fails fast if missing)."""
        state_births = BirthsIndex.from_dataframe(load_births(data_dir))
        national_births = (
            BirthsIndex.from_dataframe(load_national_births(data_dir))
            if has_national_births(data_dir)
            else None
        )
        death_probs: dict[Sex, pd.DataFrame] = {
            "F": load_death_probabilities(data_dir, "F", tr_year=tr_year),
            "M": load_death_probabilities(data_dir, "M", tr_year=tr_year),
        }
        return cls.from_tables(
            state_births=state_births,
            national_births=national_births,
            death_probs=death_probs,
            tr_year=tr_year,
        )

    @classmethod
    def from_tables(
        cls,
        *,
        state_births: BirthsIndex,
        death_probs: dict[Sex, pd.DataFrame],
        national_births: BirthsIndex | None = None,
        tr_year: int = DEFAULT_TR_YEAR,
    ) -> NameAgeStore:
        """Construct a store from already-loaded tables and precompute survival."""
        survival_curves: dict[tuple[Sex, int], pd.Series] = {}
        for sex, table in death_probs.items():
            for year in table.index:
                yr = int(year)
                survival_curves[(sex, yr)] = compute_survival_curve(table, yr)
        return cls(
            state_births=state_births,
            national_births=national_births,
            death_probs=death_probs,
            survival_curves=survival_curves,
            tr_year=tr_year,
        )

    @property
    def using_national_us(self) -> bool:
        return self.national_births is not None

    @property
    def regions(self) -> list[str]:
        states = sorted(r for r in self.state_births.regions if r != "US")
        return ["US", *states]

    @property
    def latest_birth_year_state(self) -> int:
        assert self.state_births.latest_birth_year is not None
        return self.state_births.latest_birth_year

    @property
    def latest_birth_year_us(self) -> int:
        if (
            self.national_births is not None
            and self.national_births.latest_birth_year is not None
        ):
            return self.national_births.latest_birth_year
        return self.latest_birth_year_state

    @property
    def mortality_year_range(self) -> tuple[int, int]:
        years = [yr for (_, yr) in self.survival_curves]
        return (min(years), max(years))

    def estimate(
        self,
        *,
        name: str,
        sex: Sex,
        region: str = "US",
        year: int | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
    ) -> tuple[NameAgeResult, list[str]]:
        """Run an indexed age-distribution estimate and return (result, warnings)."""
        cleaned_name = name.strip()
        if not cleaned_name:
            raise InvalidQueryError("name must not be blank.")

        normalized_region = region.strip().upper()
        if (
            normalized_region != "US"
            and normalized_region not in self.state_births.regions
        ):
            raise InvalidQueryError(
                f"Unknown region {region!r}; expected 'US' or a 2-letter "
                "U.S. state/DC code."
            )
        births_index = (
            self.national_births
            if normalized_region == "US" and self.national_births is not None
            else self.state_births
        )

        latest_year = births_index.latest_birth_year
        if latest_year is None:
            raise InvalidQueryError("Births dataset is empty.")

        ref_year = year if year is not None else latest_year
        survival = self.survival_curves.get((sex, ref_year))
        if survival is None:
            min_yr, max_yr = self.mortality_year_range
            raise InvalidQueryError(
                f"Reference year {ref_year} not in mortality table ({min_yr}–{max_yr})."
            )

        warnings: list[str] = []
        if year is not None and year > latest_year:
            warnings.append(
                f"Reference year {year} is after latest birth year {latest_year}; "
                f"ages 0–{year - latest_year - 1} have no birth records and are "
                "treated as 0."
            )
        if births_index.lookup(
            name=cleaned_name, sex=sex, region=normalized_region
        ).empty:
            raise NameNotFoundError(
                f"No birth records found for name={cleaned_name!r}, "
                f"sex={sex!r}, region={normalized_region!r}."
            )

        try:
            result = estimate_age_distribution(
                births_index,
                self.death_probs[sex],
                name=cleaned_name,
                sex=sex,
                region=normalized_region,
                reference_year=ref_year,
                min_age=min_age,
                max_age=max_age,
                survival=survival,
            )
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc

        return result, warnings
