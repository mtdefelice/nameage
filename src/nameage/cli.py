"""Command-line interface for nameage."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from nameage.data import (
    DEFAULT_TR_YEAR,
    Sex,
    ensure_ssa_data,
    has_national_births,
    load_births,
    load_death_probabilities,
    load_national_births,
)
from nameage.model import estimate_age_distribution
from nameage.plot import plot_age_distribution

_REGION_PATTERN = re.compile(r"^[A-Z]{2}$")
_SAFE_SLUG_PATTERN = re.compile(r"[^a-z0-9_-]+")


def _parse_region(value: str) -> str:
    region = value.strip().upper()
    if not _REGION_PATTERN.match(region):
        raise argparse.ArgumentTypeError(
            f"region must be 'US' or a 2-letter state/DC code, got {value!r}"
        )
    return region


def _safe_output_path(output_dir: Path, region: str, name: str, fmt: str) -> Path:
    reg_slug = _SAFE_SLUG_PATTERN.sub("_", region.lower())
    name_slug = _SAFE_SLUG_PATTERN.sub("_", name.lower())
    out_path = output_dir / f"nameage-{reg_slug}-{name_slug}.{fmt}"
    if not out_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(f"Unsafe output path derived for {name!r}: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="nameage",
        description=(
            "Estimate U.S. living age distributions by first name using SSA data."
        ),
    )
    parser.add_argument(
        "names",
        nargs="+",
        help="One or more first names to plot (e.g., John Paul).",
    )
    parser.add_argument(
        "--sex",
        required=True,
        type=str.upper,
        choices=("F", "M"),
        help="Sex recorded on SSA birth records ('F' or 'M').",
    )
    parser.add_argument(
        "--region",
        default="US",
        type=_parse_region,
        help="Region: 'US' (default) or a 2-letter U.S. state/DC code (e.g., CA, NY).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Reference calendar year (defaults to latest birth year in dataset).",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=None,
        help="Optional inclusive lower age bound (e.g., 18 to condition on adults).",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="Optional inclusive upper age bound.",
    )
    parser.add_argument(
        "--tr-year",
        type=int,
        default=DEFAULT_TR_YEAR,
        help=(
            f"SSA Trustees Report year for mortality tables "
            f"(default: {DEFAULT_TR_YEAR})."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ds"),
        help="Directory containing SSA source datasets (default: ./ds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated plots (default: ./output).",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        default="pdf",
        choices=("pdf", "png", "svg"),
        help="Output plot file format (default: pdf).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download and re-extract SSA source files before plotting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    sex: Sex = args.sex

    try:
        ensure_ssa_data(
            args.data_dir,
            sex=sex,
            region=args.region,
            tr_year=args.tr_year,
            refresh=args.refresh,
        )
        if args.region == "US" and has_national_births(args.data_dir):
            births = load_national_births(args.data_dir)
        else:
            births = load_births(args.data_dir)
        death_probs = load_death_probabilities(args.data_dir, sex, tr_year=args.tr_year)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"nameage: error: {exc}", file=sys.stderr)
        return 1

    if args.year is not None and not births.empty:
        latest_birth_year = int(births["birth_year"].max())
        if args.year > latest_birth_year:
            print(
                f"nameage: warning: reference year {args.year} is after latest birth "
                f"year {latest_birth_year}; ages 0–{args.year - latest_birth_year - 1} "
                "have no birth records and are treated as 0.",
                file=sys.stderr,
            )

    had_error = False
    for raw_name in args.names:
        try:
            result = estimate_age_distribution(
                births,
                death_probs,
                name=raw_name,
                sex=sex,
                region=args.region,
                reference_year=args.year,
                min_age=args.min_age,
                max_age=args.max_age,
            )
            out_path = _safe_output_path(
                args.output_dir, result.region, result.name, args.image_format
            )
            plot_age_distribution(result, out_path)
        except (ValueError, OSError) as exc:
            print(f"nameage: error: {exc}", file=sys.stderr)
            had_error = True
            continue

        q25, q75 = result.iqr
        print(
            f"{result.name} ({result.region}, {result.sex}, {result.reference_year}): "
            f"median age {result.median_age} (IQR {q25}–{q75}, "
            f"mean {result.mean_age:.1f}), "
            f"expected living {result.expected_living:,.0f} -> {out_path}"
        )

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
