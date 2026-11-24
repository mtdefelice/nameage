"""Matplotlib visualization for nameage distributions."""

from __future__ import annotations

from pathlib import Path

from matplotlib import style as mpl_style
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from nameage.model import NameAgeResult

PLOT_STYLE = "seaborn-v0_8-paper"


def plot_age_distribution(
    result: NameAgeResult,
    output_path: Path | str,
    *,
    max_age: int = 100,
) -> Path:
    """Render and save the born vs. expected-living age distribution chart.

    Uses ``matplotlib.figure.Figure`` directly so importing or calling this
    function never mutates Matplotlib's global backend or ``pyplot`` state.
    """
    if max_age <= 0:
        raise ValueError(f"max_age must be positive, got {max_age!r}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dist = result.distribution
    x_upper = max(max_age, result.median_age)

    with mpl_style.context(PLOT_STYLE):
        fig = Figure(figsize=(8, 5))
        ax = fig.subplots()

        ax.plot(dist.index, dist["born"], label="Born")
        ax.fill_between(
            dist.index,
            dist["expected_living"],
            alpha=0.6,
            label=f"Expected Living ({result.reference_year})",
        )
        ax.axvline(x=result.median_age, color="k", linestyle=":", linewidth=0.8)

        ax.set_title(
            f"{result.name}: {result.region} Age Distribution; "
            f"{result.expected_living:,.0f} Expected"
        )
        ax.set_xlabel("Age")
        ax.set_xlim(0, x_upper)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

        if result.median_age == 0:
            ha, offset = "left", (4, 6)
        elif result.median_age >= x_upper:
            ha, offset = "right", (-4, 6)
        else:
            ha, offset = "center", (0, 6)

        ax.annotate(
            f"{result.median_age}",
            xy=(result.median_age, 0),
            xycoords="data",
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=8,
            fontweight="bold",
            annotation_clip=True,
            bbox={"facecolor": "white", "edgecolor": "white", "pad": 3, "alpha": 1.0},
        )
        ax.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(out)

    return out
