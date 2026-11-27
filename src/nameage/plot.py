"""Matplotlib visualization for nameage distributions."""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Literal

from matplotlib import style as mpl_style
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from nameage.model import NameAgeResult

PLOT_STYLE = "seaborn-v0_8-paper"
DEFAULT_DPI = 300
PlotFormat = Literal["png", "svg", "pdf"]
PlotTheme = Literal["light", "dark"]
_DARK_BG = "#111827"
_DARK_FG = "#e2e8f0"
_DARK_SPINE = "#334155"
_DARK_LINE = "#60a5fa"
_DARK_FILL = "#3b82f6"
_DARK_MEDIAN = "#cbd5e1"

# Matplotlib's style.context mutates global rcParams; serialize styled renders.
_PLOT_LOCK = threading.Lock()


def render_age_distribution(
    result: NameAgeResult,
    *,
    fmt: PlotFormat = "png",
    max_age: int = 100,
    dpi: int = DEFAULT_DPI,
    theme: PlotTheme = "light",
    transparent: bool = False,
) -> bytes:
    """Render the age distribution chart to in-memory bytes (thread-safe).

    Uses ``matplotlib.figure.Figure`` directly so importing or calling this
    function never mutates Matplotlib's global backend or ``pyplot`` state.
    """
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError(f"fmt must be 'png', 'svg', or 'pdf', got {fmt!r}")
    if theme not in ("light", "dark"):
        raise ValueError(f"theme must be 'light' or 'dark', got {theme!r}")
    if max_age <= 0:
        raise ValueError(f"max_age must be positive, got {max_age!r}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi!r}")
    dist = result.distribution
    q25, q75 = result.iqr
    x_lower = int(dist.index.min())
    x_upper = max(max_age, result.median_age, q75)
    if x_lower >= x_upper:
        x_upper = max(int(dist.index.max()), x_lower + 1)

    buf = io.BytesIO()
    is_dark = theme == "dark"
    with _PLOT_LOCK, mpl_style.context(PLOT_STYLE):
        fig = Figure(figsize=(8, 5))
        ax = fig.subplots()

        if is_dark:
            fig.patch.set_facecolor(_DARK_BG)
            ax.set_facecolor(_DARK_BG)
            for spine in ax.spines.values():
                spine.set_color(_DARK_SPINE)
            ax.tick_params(colors=_DARK_FG)

        plot_kwargs = {"color": _DARK_LINE} if is_dark else {}
        fill_kwargs = {"color": _DARK_FILL} if is_dark else {}
        ax.plot(dist.index, dist["born"], label="Born", **plot_kwargs)
        living_fill = ax.fill_between(
            dist.index,
            dist["expected_living"],
            alpha=0.25,
            label=f"Expected Living ({result.reference_year})",
            **fill_kwargs,
        )
        if q75 > q25:
            iqr_mask = (dist.index >= q25) & (dist.index <= q75)
            ax.fill_between(
                dist.index,
                dist["expected_living"],
                where=iqr_mask,
                color=living_fill.get_facecolor()[0][:3],
                alpha=0.55,
                label=f"IQR ({q25}–{q75})",
            )
        median_color = _DARK_MEDIAN if is_dark else "k"
        ax.axvline(
            x=result.median_age, color=median_color, linestyle=":", linewidth=0.8
        )

        text_color = _DARK_FG if is_dark else None
        title_kwargs = {"color": text_color} if text_color else {}
        ax.set_title(
            f"{result.name}: {result.region} Age Distribution; "
            f"{result.expected_living:,.0f} Expected",
            **title_kwargs,
        )
        ax.set_xlabel("Age", **title_kwargs)
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

        if result.median_age <= x_lower:
            ha, offset = "left", (4, 6)
        elif result.median_age >= x_upper:
            ha, offset = "right", (-4, 6)
        else:
            ha, offset = "center", (0, 6)

        bbox_color = _DARK_BG if is_dark else "white"
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
            bbox={
                "facecolor": bbox_color,
                "edgecolor": bbox_color,
                "pad": 3,
                "alpha": 1.0,
            },
            **title_kwargs,
        )
        legend = ax.legend(frameon=False)
        if text_color:
            for text in legend.get_texts():
                text.set_color(text_color)

        fig.tight_layout()
        fig.savefig(buf, format=fmt, dpi=dpi, transparent=transparent)
    return buf.getvalue()


def plot_age_distribution(
    result: NameAgeResult,
    output_path: Path | str,
    *,
    max_age: int = 100,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render and save the born vs. expected-living age distribution chart.

    Uses ``matplotlib.figure.Figure`` directly so importing or calling this
    function never mutates Matplotlib's global backend or ``pyplot`` state.
    """
    out = Path(output_path)
    ext = out.suffix.lstrip(".").lower() or "png"
    if ext not in ("png", "svg", "pdf"):
        raise ValueError(
            f"Unsupported output extension {out.suffix!r}; "
            "expected .png, .svg, or .pdf."
        )
    image_bytes = render_age_distribution(
        result,
        fmt=ext,
        max_age=max_age,
        dpi=dpi,  # type: ignore[arg-type]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image_bytes)
    return out
