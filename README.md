# nameage

Estimate the living age distribution and median age of people in the United States (nationally or by state) with a given first name, using U.S. Social Security Administration (SSA) baby-name birth counts and actuarial death-probability tables. Inspired by FiveThirtyEight's 2014 article [*How to Tell Someone's Age When All You Know Is Her Name*](https://web.archive.org/web/20240413083600/https://fivethirtyeight.com/features/how-to-tell-someones-age-when-all-you-know-is-her-name/) (archived).

## Quick Start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Ensure the SSA datasets are present in `ds/` (see [Updating the Source Data](#updating-the-source-data)), then run:

```bash
# Single name (US national aggregate, PNG output in gitignored ./output/)
uv run nameage John --sex M

# Curated sample charts live in ./examples/

# Multiple names in one pass (loads the 6.7M-row dataset once)
uv run nameage John Paul --sex M --format pdf

# State-specific distribution and explicit reference year
uv run nameage Lucy --sex F --region CA --year 2025
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `names` | *(required)* | One or more first names to plot (case-insensitive) |
| `--sex {F,M}` | *(required)* | Sex recorded on SSA birth records |
| `--region` | `US` | `US` (all states combined) or 2-letter U.S. state/DC code |
| `--year` | latest birth year | Reference calendar year (must exist in mortality table; years after the latest birth year warn and treat unobserved recent births as 0) |
| `--min-age` | *(none)* | Optional inclusive lower age bound for conditioning the posterior (e.g., `18` for adults) |
| `--max-age` | *(none)* | Optional inclusive upper age bound for conditioning the posterior |
| `--tr-year` | `2026` | SSA Trustees Report year for mortality tables |
| `--data-dir` | `ds` | Directory containing SSA source files |
| `--output-dir` | `output` | Output directory for generated charts |
| `--format` | `png` | Output format: `png`, `pdf`, or `svg` |
| `--refresh` | `false` | Re-download and re-extract SSA source files |

## Web Service (FastAPI)

Install the optional `api` dependencies and run the service locally:

```bash
uv sync --extra api
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open the web UI at `http://localhost:8000/` or interactive OpenAPI docs at `http://localhost:8000/docs`.

### Endpoints

| Method & Path | Description |
|---|---|
| `GET /` | Web UI for interactive name lookups and SVG charts |
| `GET /health` | Liveness/readiness check and loaded dataset summary |
| `GET /v1/metadata` | Available regions, latest birth years, and mortality year range |
| `GET /v1/estimate` | JSON age distribution & summary stats (`?name=John&sex=M&region=US`) |
| `GET /v1/plot` | Rendered chart bytes (`?name=John&sex=M&region=US&format=png\|svg\|pdf`) |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NAMEAGE_DATA_DIR` | `ds` | Directory containing SSA source files |
| `NAMEAGE_TR_YEAR` | `2026` | SSA Trustees Report year for mortality tables |
| `NAMEAGE_CACHE_MAX_AGE` | `86400` | `Cache-Control` max-age (seconds) on read endpoints |
| `NAMEAGE_CORS_ORIGINS` | *(unset)* | Comma-separated allowed CORS origins |
| `NAMEAGE_PLOT_MAX_CONCURRENCY` | `4` | Max concurrent `/v1/plot` renders before returning `503` |
| `NAMEAGE_PLOT_CACHE_SIZE` | `128` | In-memory LRU cache entries for rendered plots (`0` disables) |

### Docker

Ensure the SSA datasets are present in `ds/` (see [Updating the Source Data](#updating-the-source-data)), then build and run the self-contained image (includes the API, web UI, and `ds/` baked into `/data`):

```bash
docker build -f api/Dockerfile -t nameage-api .
docker run --rm -p 8000:8000 nameage-api
```

You can still override the baked-in dataset at runtime by mounting a different directory at `/data` (`-v /path/to/ds:/data:ro`).

### Repository Layout

```text
nameage/
├── src/nameage/      # Core library (data, indexed model, thread-safe plot) + CLI
├── api/              # FastAPI service, static web UI (api/static/), and Dockerfile
├── tests/            # Unit and integration tests (core, CLI, API)
└── ds/               # Gitignored SSA datasets (baked into container at /data)
```

## Methodology & Caveats

- **Mid-interval period life-table survival**: For a given reference year $Y$, `nameage` takes that year's single-year-of-age death probabilities $q_0, q_1, \dots, q_{119}$ and computes exact-age survival $l_0 = 1$, $l_{a+1} = \prod_{x=0}^{a} (1 - q_x)$, then the stationary mid-interval survival ratio $S(a) = L_a / l_0 = \tfrac{1}{2}(l_a + l_{a+1})$ across the full contiguous age range before joining with birth counts. Expected living counts at age $a$ are $\text{born}(Y - a) \cdot S(a)$, normalized into a posterior PMF with median, mean, mode, IQR (25th–75th), and 90% interval (5th–95th).
- **Period vs. cohort mortality**: Applying reference-year mortality $q_x(Y)$ rather than historical cohort mortality $q_{x,\, Y - a + x}$ overestimates survival of older cohorts (who experienced higher historical mortality), introducing an upward bias in predicted age for older and multi-generational names.
- **SSA `< 5` per-state privacy suppression**: SSA's state-level dataset omits any name with fewer than 5 occurrences in a given state and year. Because modern name diversity is much higher than mid-century, summing all states for `US` drops ~23% of 2024–2025 births (vs. ~8% in 1950), clipping recent cohorts more aggressively for uncommon names. To eliminate this for `--region US`, optionally download SSA's national archive (`https://www.ssa.gov/oact/babynames/names.zip`) and extract `yobYYYY.txt` into `ds/names/`; `nameage` uses it automatically for `US` when present.
- **Closed population & migration**: SSA birth records cover U.S.-born SSN holders only (excluding foreign-born residents, ~14% of the U.S. population). For `--region <STATE>`, `namesbystate.zip` reflects **state of birth** (or state of SSN issuance prior to 1987), not state of residence, and uses national rather than state-specific mortality.
- **Pre-1937 / pre-1987 SSN registration bias**: Because SSNs were introduced in 1936 and Enumeration at Birth began in 1987, older cohorts appear in SSA data only if they survived long enough to apply for an SSN; applying full-life survival double-counts their early-life mortality.
- **Name-independent mortality & legal names**: Mortality is conditioned only on sex and age (not socioeconomic/demographic correlates of first names), and counts reflect legal first names at birth rather than nicknames or name changes. Use `--min-age` / `--max-age` when you know the subject is an adult or within a specific age range.

## Data Sources

All source datasets are published by the **U.S. Social Security Administration, Office of the Chief Actuary (OACT)**:

| Dataset | Local Path | Official SSA Page | Direct File URL | Coverage |
|---|---|---|---|---|
| Baby Names by State | `ds/namesbystate.zip` (`ds/namesbystate/*.TXT`) | [Popular Baby Names — Beyond the Top 1000](https://www.ssa.gov/oact/babynames/limits.html) | `https://www.ssa.gov/oact/babynames/state/namesbystate.zip` | 1910–2025 (50 states + DC; 6,696,687 rows) |
| Baby Names National *(optional, preferred for `US`)* | `ds/names.zip` (`ds/names/yob*.txt`) | [Popular Baby Names — Beyond the Top 1000](https://www.ssa.gov/oact/babynames/limits.html) | `https://www.ssa.gov/oact/babynames/names.zip` | 1880–2025 (national `< 5` threshold) |
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
