"""Pydantic v2 response schemas for the nameage FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from nameage.data import Sex
from nameage.model import NameAgeResult


class AgePoint(BaseModel):
    """Single-year-of-age entry in the estimated living distribution."""

    age: int
    birth_year: int
    born: int
    survival_prob: float
    expected_living: float
    prob: float
    cdf: float


class EstimateResponse(BaseModel):
    """Estimated living age distribution and summary statistics for a name."""

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
    warnings: list[str]
    distribution: list[AgePoint] | None = None

    @classmethod
    def from_result(
        cls,
        result: NameAgeResult,
        *,
        warnings: list[str],
        include_distribution: bool = True,
    ) -> EstimateResponse:
        points: list[AgePoint] | None = None
        if include_distribution:
            points = [
                AgePoint(
                    age=int(row.Index),
                    birth_year=int(row.birth_year),
                    born=int(row.born),
                    survival_prob=float(row.survival_prob),
                    expected_living=float(row.expected_living),
                    prob=float(row.prob),
                    cdf=float(row.cdf),
                )
                for row in result.distribution.itertuples(index=True)
            ]
        return cls(
            name=result.name,
            sex=result.sex,
            region=result.region,
            reference_year=result.reference_year,
            median_age=result.median_age,
            mean_age=result.mean_age,
            mode_age=result.mode_age,
            iqr=result.iqr,
            ci90=result.ci90,
            expected_living=result.expected_living,
            warnings=warnings,
            distribution=points,
        )


class HealthResponse(BaseModel):
    """Service liveness and loaded-dataset summary."""

    status: Literal["ok"]
    tr_year: int
    using_national_us: bool
    regions_count: int
    latest_birth_year: int
    mortality_year_range: tuple[int, int]


class MetadataResponse(BaseModel):
    """Available regions and calendar year coverage for the loaded datasets."""

    tr_year: int
    using_national_us: bool
    regions: list[str]
    latest_birth_year_us: int
    latest_birth_year_state: int
    mortality_year_range: tuple[int, int]
