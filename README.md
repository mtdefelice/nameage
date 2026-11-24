# nameage

Estimate the living age distribution and median age of people in the United States (nationally or by state) with a given first name, using U.S. Social Security Administration (SSA) baby-name birth counts and actuarial death-probability tables.

## Quick Start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Ensure the SSA datasets are present in `ds/` (see [Updating the Source Data](#updating-the-source-data)), then run:

```bash
# Single name (US national aggregate, PDF output in gitignored ./output/)
uv run nameage John --sex M

# Curated sample charts live in ./examples/

# Multiple names in one pass (loads the 6.7M-row dataset once)
uv run nameage John Paul --sex M --format png

# State-specific distribution and explicit reference year
uv run nameage Lucy --sex F --region CA --year 2025
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `names` | *(required)* | One or more first names to plot (case-insensitive) |
| `--sex {F,M}` | *(required)* | Sex recorded on SSA birth records |
| `--region` | `US` | `US` (all states combined) or 2-letter U.S. state/DC code |
| `--year` | latest birth year | Reference calendar year (must exist in mortality table, e.g. `2024–2100` for `TR2026`) |
| `--tr-year` | `2026` | SSA Trustees Report year for mortality tables |
| `--data-dir` | `ds` | Directory containing SSA source files |
| `--output-dir` | `output` | Output directory for generated charts |
| `--format` | `pdf` | Output format: `pdf`, `png`, or `svg` |
| `--refresh` | `false` | Re-download and re-extract SSA source files |

## Methodology & Caveats

- **Period life-table approximation**: For a given reference year $Y$, `nameage` takes that year's single-year-of-age death probabilities $q_0, q_1, \dots, q_{119}$ and computes cumulative survival $S(a) = \prod_{x=0}^{a} (1 - q_x)$ across the full contiguous age range before joining with birth counts. Expected living counts at age $a$ are $\text{born}(Y - a) \cdot S(a)$. This is a standard period life-table approximation (it applies reference-year mortality rather than historical cohort mortality).
- **SSA `< 5` privacy suppression**: SSA's state-level dataset omits any name with fewer than 5 occurrences in a given state and year. Summing all states for `US` slightly undercounts rarer names compared to SSA's national archive (`https://www.ssa.gov/oact/babynames/names.zip`).

## Data Sources

All source datasets are published by the **U.S. Social Security Administration, Office of the Chief Actuary (OACT)**:

| Dataset | Local Path | Official SSA Page | Direct File URL | Coverage |
|---|---|---|---|---|
| Baby Names by State | `ds/namesbystate.zip` (`ds/namesbystate/*.TXT`) | [Popular Baby Names — Beyond the Top 1000](https://www.ssa.gov/oact/babynames/limits.html) | `https://www.ssa.gov/oact/babynames/state/namesbystate.zip` | 1910–2025 (50 states + DC; 6,696,687 rows) |
| Female Death Probabilities (TR2026, Alt2) | `ds/DeathProbsE_F_Alt2_TR2026.csv` | [OASDI Trustees Report Downloadable Files](https://www.ssa.gov/oact/Downloadables/CY/index.html) | `https://www.ssa.gov/oact/Downloadables/CY/DeathProbsE_F_Alt2_TR2026.csv` | Calendar years 2024–2100, ages 0–119 |
| Male Death Probabilities (TR2026, Alt2) | `ds/DeathProbsE_M_Alt2_TR2026.csv` | [OASDI Trustees Report Downloadable Files](https://www.ssa.gov/oact/Downloadables/CY/index.html) | `https://www.ssa.gov/oact/Downloadables/CY/DeathProbsE_M_Alt2_TR2026.csv` | Calendar years 2024–2100, ages 0–119 |

SSA publishes the **current** Trustees Report under `/oact/Downloadables/CY/` and archives prior reports under `/oact/Downloadables/<YYYY>/` (replacing the older `/oact/HistEst/Death/<YYYY>/` path).

## Updating the Source Data

SSA updates `namesbystate.zip` annually (typically around May, adding the prior calendar year) and publishes a new Trustees Report (`TR<YYYY>`) each year. `ds/` is gitignored so large public datasets are not tracked in git.

> **Note on automated downloads (`HTTP 403`)**: `www.ssa.gov` sits behind Akamai bot protection, which currently returns `403 Forbidden` to `curl`, `wget`, and Python's `urllib.request`. Download the files in a standard web browser and place them in `ds/`.

### Steps to Update

1. **Download Baby Names by State**:
   - Open [https://www.ssa.gov/oact/babynames/limits.html](https://www.ssa.gov/oact/babynames/limits.html) in a browser and click **State-specific data** (`namesbystate.zip`).
   - Save it to `ds/namesbystate.zip` and extract into `ds/namesbystate/`:
     ```bash
     rm -rf ds/namesbystate && mkdir -p ds/namesbystate
     unzip -o ds/namesbystate.zip -d ds/namesbystate
     ```

2. **Download Death Probabilities (Intermediate / Alt2)**:
   - Open [https://www.ssa.gov/oact/Downloadables/CY/index.html](https://www.ssa.gov/oact/Downloadables/CY/index.html) in a browser.
   - Under **Death Probabilities** (Intermediate / `Alt2`), download the **CSV** files for both females and males:
     - `DeathProbsE_F_Alt2_TR<YYYY>.csv`
     - `DeathProbsE_M_Alt2_TR<YYYY>.csv`
   - Place them in `ds/` and remove the previous `TR<OLD_YEAR>` CSVs.

3. **Pass `--tr-year <YYYY>`** (or update `DEFAULT_TR_YEAR` in `src/nameage/data.py`) when the Trustees Report year changes.

4. **Verify the updated data**:
   ```bash
   head -n 3 ds/DeathProbsE_F_Alt2_TR*.csv | cut -c1-100
   awk -F, '{if($3>max)max=$3; if(min==""||$3<min)min=$3} END{print "Years:", min, "-", max, "Rows:", NR}' ds/namesbystate/*.TXT
   ```

## Development

```bash
uv run pytest                  # run test suite
uv run ruff check .            # lint
uv run ruff format --check .   # check formatting
```

## License

MIT — see [LICENSE](LICENSE).
