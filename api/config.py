"""Environment-driven configuration for the nameage FastAPI service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from nameage.data import DEFAULT_TR_YEAR


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the nameage web service."""

    data_dir: Path = Path("ds")
    tr_year: int = DEFAULT_TR_YEAR
    cache_max_age: int = 86_400
    cors_origins: tuple[str, ...] = ()
    plot_max_concurrency: int = 4
    plot_cache_size: int = 128

    def __post_init__(self) -> None:
        if self.cache_max_age < 0:
            raise ValueError(
                f"cache_max_age must be non-negative, got {self.cache_max_age!r}"
            )
        if self.plot_max_concurrency < 1:
            raise ValueError(
                "plot_max_concurrency must be at least 1, "
                f"got {self.plot_max_concurrency!r}"
            )
        if self.plot_cache_size < 0:
            raise ValueError(
                f"plot_cache_size must be non-negative, got {self.plot_cache_size!r}"
            )

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from ``NAMEAGE_*`` environment variables."""
        data_dir = Path(os.environ.get("NAMEAGE_DATA_DIR", "ds"))
        tr_year = int(os.environ.get("NAMEAGE_TR_YEAR", str(DEFAULT_TR_YEAR)))
        cache_max_age = int(os.environ.get("NAMEAGE_CACHE_MAX_AGE", "86400"))
        plot_max_concurrency = int(os.environ.get("NAMEAGE_PLOT_MAX_CONCURRENCY", "4"))
        plot_cache_size = int(os.environ.get("NAMEAGE_PLOT_CACHE_SIZE", "128"))
        raw_origins = os.environ.get("NAMEAGE_CORS_ORIGINS", "")
        cors_origins = tuple(
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        )
        return cls(
            data_dir=data_dir,
            tr_year=tr_year,
            cache_max_age=cache_max_age,
            cors_origins=cors_origins,
            plot_max_concurrency=plot_max_concurrency,
            plot_cache_size=plot_cache_size,
        )
