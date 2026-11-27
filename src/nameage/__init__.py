"""Estimate U.S. living age distributions by first name from SSA datasets."""

from nameage.data import load_births, load_death_probabilities, load_national_births
from nameage.model import (
    BirthsIndex,
    NameAgeResult,
    compute_survival_curve,
    estimate_age_distribution,
)
from nameage.plot import plot_age_distribution, render_age_distribution

__all__ = [
    "BirthsIndex",
    "NameAgeResult",
    "compute_survival_curve",
    "estimate_age_distribution",
    "load_births",
    "load_death_probabilities",
    "load_national_births",
    "plot_age_distribution",
    "render_age_distribution",
]
