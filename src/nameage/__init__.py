"""Estimate U.S. living age distributions by first name from SSA datasets."""

from nameage.data import load_births, load_death_probabilities
from nameage.model import (
    NameAgeResult,
    compute_survival_curve,
    estimate_age_distribution,
)
from nameage.plot import plot_age_distribution

__all__ = [
    "NameAgeResult",
    "compute_survival_curve",
    "estimate_age_distribution",
    "load_births",
    "load_death_probabilities",
    "plot_age_distribution",
]
