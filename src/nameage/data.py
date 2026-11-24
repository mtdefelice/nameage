"""SSA dataset download, extraction, and loading utilities."""

from __future__ import annotations

import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd

Sex = Literal["F", "M"]

DEFAULT_TR_YEAR = 2026
SSA_BABYNAMES_STATE_URL = "https://www.ssa.gov/oact/babynames/state/namesbystate.zip"
SSA_BABYNAMES_NATIONAL_URL = "https://www.ssa.gov/oact/babynames/names.zip"
_YOB_FILE_PATTERN = re.compile(r"^yob(\d{4})\.txt$", re.IGNORECASE)
_SEX_DTYPE = pd.CategoricalDtype(categories=["F", "M"])


def _validate_sex(sex: str) -> Sex:
    if sex not in ("F", "M"):
        raise ValueError(f"sex must be 'F' or 'M', got {sex!r}")
    return sex


def death_probs_filename(sex: Sex, tr_year: int = DEFAULT_TR_YEAR) -> str:
    """Return the SSA death-probability CSV filename for sex and report year."""
    valid_sex = _validate_sex(sex)
    if not isinstance(tr_year, int) or isinstance(tr_year, bool) or tr_year < 1900:
        raise ValueError(f"Invalid Trustees Report year: {tr_year!r}")
    return f"DeathProbsE_{valid_sex}_Alt2_TR{tr_year}.csv"


def death_probs_url(sex: Sex, tr_year: int = DEFAULT_TR_YEAR) -> str:
    """Return the SSA OACT URL for a death-probability CSV."""
    filename = death_probs_filename(sex, tr_year)
    folder = "CY" if tr_year == DEFAULT_TR_YEAR else str(tr_year)
    return f"https://www.ssa.gov/oact/Downloadables/{folder}/{filename}"


def _download_atomic(url: str, dest: Path, timeout: float = 30.0) -> None:
    """Download a URL via a temp file so partial writes never corrupt the cache."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; nameage/0.2)"},
    )
    tmp_path: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            tempfile.NamedTemporaryFile(
                dir=dest.parent, prefix=f".{dest.name}.", delete=False
            ) as tmp,
        ):
            tmp_path = Path(tmp.name)
            while chunk := response.read(65536):
                tmp.write(chunk)
        tmp_path.replace(dest)
    except BaseException as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
            raise RuntimeError(
                f"SSA returned HTTP 403 for {url} (Akamai bot protection). "
                "Download the file manually in a browser and place it in "
                f"'{dest.parent}' (see README.md)."
            ) from exc
        raise


def _extract_zip_safely(zip_path: Path, dest_dir: Path) -> None:
    """Validate zip members against Zip Slip and extract atomically."""
    stage_dir = Path(tempfile.mkdtemp(dir=dest_dir.parent, prefix=f".{dest_dir.name}."))
    try:
        resolved_stage = stage_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                target = (stage_dir / member.filename).resolve()
                if not target.is_relative_to(resolved_stage):
                    raise ValueError(
                        f"Unsafe zip member path in {zip_path}: {member.filename!r}"
                    )
            archive.extractall(stage_dir)

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        stage_dir.replace(dest_dir)
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def has_national_births(data_dir: Path | str) -> bool:
    """Return True if extracted SSA national ``yobYYYY.txt`` files exist locally."""
    names_dir = Path(data_dir) / "names"
    return names_dir.is_dir() and any(
        _YOB_FILE_PATTERN.match(p.name) for p in names_dir.iterdir() if p.is_file()
    )


def ensure_ssa_data(
    data_dir: Path | str,
    *,
    sex: Sex | None = None,
    region: str = "US",
    tr_year: int = DEFAULT_TR_YEAR,
    refresh: bool = False,
) -> None:
    """Ensure required SSA baby-name and death-probability files exist locally.

    When ``region == "US"`` and national files (``names/yob*.txt`` or
    ``names.zip``) exist locally, refreshes/extracts ``names.zip`` instead of
    ``namesbystate.zip``. Otherwise skips downloading ``namesbystate.zip`` when
    extracted ``*.TXT`` files already exist (unless ``refresh=True``), and only
    checks the requested ``sex`` CSV when ``sex`` is specified.
    """
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    names_dir = base_dir / "namesbystate"
    zip_path = base_dir / "namesbystate.zip"
    nat_dir = base_dir / "names"
    nat_zip_path = base_dir / "names.zip"

    has_nat_files = has_national_births(base_dir)
    has_state_files = names_dir.is_dir() and any(names_dir.glob("*.TXT"))
    use_national = region.upper() == "US" and (has_nat_files or nat_zip_path.is_file())
    need_national_files = use_national and (refresh or not has_nat_files)
    need_state_files = not use_national and (refresh or not has_state_files)

    targets: list[tuple[str, Path]] = []
    if need_national_files:
        targets.append((SSA_BABYNAMES_NATIONAL_URL, nat_zip_path))
    if need_state_files:
        targets.append((SSA_BABYNAMES_STATE_URL, zip_path))

    sexes: tuple[Sex, ...] = (_validate_sex(sex),) if sex is not None else ("F", "M")
    for s in sexes:
        targets.append(
            (death_probs_url(s, tr_year), base_dir / death_probs_filename(s, tr_year))
        )

    for url, dest in targets:
        if refresh or not dest.is_file():
            _download_atomic(url, dest)

    if need_national_files:
        _extract_zip_safely(nat_zip_path, nat_dir)
    if need_state_files:
        _extract_zip_safely(zip_path, names_dir)


def load_births(data_dir: Path | str) -> pd.DataFrame:
    """Load and concatenate all state-level SSA baby-name files into one DataFrame.

    Returns columns: ``state``, ``sex``, ``birth_year``, ``name``, ``count``.
    """
    names_dir = Path(data_dir) / "namesbystate"
    state_files = sorted(names_dir.glob("*.TXT"))
    if not state_files:
        raise FileNotFoundError(
            f"No state *.TXT files found in '{names_dir}'. "
            "Download and extract namesbystate.zip (see README.md)."
        )

    frames = [
        pd.read_csv(
            path,
            header=None,
            names=["state", "sex", "birth_year", "name", "count"],
            dtype={
                "state": "category",
                "sex": _SEX_DTYPE,
                "birth_year": "int16",
                "name": "string",
                "count": "int32",
            },
            na_filter=False,
        )
        for path in state_files
    ]

    # Unify per-file state categories so pd.concat preserves CategoricalDtype.
    all_states = sorted(
        {cat for frame in frames for cat in frame["state"].cat.categories}
    )
    state_dtype = pd.CategoricalDtype(categories=all_states)
    for frame in frames:
        frame["state"] = frame["state"].astype(state_dtype)

    df = pd.concat(frames, ignore_index=True)
    df["name"] = df["name"].astype("category")
    return df


def load_national_births(data_dir: Path | str) -> pd.DataFrame:
    """Load SSA national ``names/yobYYYY.txt`` files into the standard births schema.

    Avoids the per-state ``< 5`` privacy suppression bias when querying ``US``.
    Returns columns: ``state`` (``"US"``), ``sex``, ``birth_year``, ``name``, ``count``.
    """
    names_dir = Path(data_dir) / "names"
    year_files_by_year: dict[int, Path] = {}
    if names_dir.is_dir():
        for path in sorted(names_dir.iterdir()):
            if not path.is_file():
                continue
            match = _YOB_FILE_PATTERN.match(path.name)
            if match:
                year = int(match.group(1))
                if year in year_files_by_year:
                    raise ValueError(
                        f"Duplicate national birth files for year {year}: "
                        f"'{year_files_by_year[year].name}' and '{path.name}'."
                    )
                year_files_by_year[year] = path
    year_files = sorted(year_files_by_year.items())
    if not year_files:
        raise FileNotFoundError(
            f"No national yobYYYY.txt files found in '{names_dir}'. "
            "Download and extract names.zip (see README.md)."
        )

    frames = []
    for year, path in year_files:
        frame = pd.read_csv(
            path,
            header=None,
            names=["name", "sex", "count"],
            dtype={"name": "string", "sex": _SEX_DTYPE, "count": "int32"},
            na_filter=False,
        )
        frame["birth_year"] = pd.array([year] * len(frame), dtype="int16")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["state"] = pd.Categorical(["US"] * len(df), categories=["US"])
    df["name"] = df["name"].astype("category")
    return df[["state", "sex", "birth_year", "name", "count"]]


def load_death_probabilities(
    data_dir: Path | str,
    sex: Sex,
    *,
    tr_year: int = DEFAULT_TR_YEAR,
) -> pd.DataFrame:
    """Load an SSA period death-probability CSV indexed by calendar ``Year``.

    Columns are integer ages ``0..119`` containing annual death probabilities ``q_x``.
    """
    valid_sex = _validate_sex(sex)
    csv_path = Path(data_dir) / death_probs_filename(valid_sex, tr_year)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Mortality table not found: '{csv_path}'. "
            "Download it from ssa.gov (see README.md)."
        )

    table = pd.read_csv(csv_path, skiprows=1).set_index("Year")
    table.index = table.index.astype("int64")
    table.columns = table.columns.astype("int64")
    return table
