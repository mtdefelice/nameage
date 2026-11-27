(() => {
  const STATE_NAMES = {
    US: "United States (All)",
    AK: "Alaska", AL: "Alabama", AR: "Arkansas", AZ: "Arizona",
    CA: "California", CO: "Colorado", CT: "Connecticut", DC: "District of Columbia",
    DE: "Delaware", FL: "Florida", GA: "Georgia", HI: "Hawaii",
    IA: "Iowa", ID: "Idaho", IL: "Illinois", IN: "Indiana",
    KS: "Kansas", KY: "Kentucky", LA: "Louisiana", MA: "Massachusetts",
    MD: "Maryland", ME: "Maine", MI: "Michigan", MN: "Minnesota",
    MO: "Missouri", MS: "Mississippi", MT: "Montana", NC: "North Carolina",
    ND: "North Dakota", NE: "Nebraska", NH: "New Hampshire", NJ: "New Jersey",
    NM: "New Mexico", NV: "Nevada", NY: "New York", OH: "Ohio",
    OK: "Oklahoma", OR: "Oregon", PA: "Pennsylvania", RI: "Rhode Island",
    SC: "South Carolina", SD: "South Dakota", TN: "Tennessee", TX: "Texas",
    UT: "Utah", VA: "Virginia", VT: "Vermont", WA: "Washington",
    WI: "Wisconsin", WV: "West Virginia", WY: "Wyoming",
  };

  const form = document.getElementById("query-form");
  const nameInput = document.getElementById("name");
  const sexSelect = document.getElementById("sex");
  const regionSelect = document.getElementById("region");
  const yearInput = document.getElementById("year");
  const minAgeInput = document.getElementById("min_age");
  const maxAgeInput = document.getElementById("max_age");
  const submitBtn = document.getElementById("submit-btn");
  const badge = document.getElementById("dataset-badge");
  const errorBanner = document.getElementById("error-banner");
  const warningBanner = document.getElementById("warning-banner");
  const resultsSection = document.getElementById("results");
  const chartTitle = document.getElementById("chart-title");
  const chartImg = document.getElementById("chart-img");
  const chartLoading = document.getElementById("chart-loading");
  const downloadSvg = document.getElementById("download-svg");
  const downloadPng = document.getElementById("download-png");

  const statMedian = document.getElementById("stat-median");
  const statSubMedian = document.getElementById("stat-sub-median");
  const statIqr = document.getElementById("stat-iqr");
  const statMeanMode = document.getElementById("stat-mean-mode");
  const statCi90 = document.getElementById("stat-ci90");
  const statLiving = document.getElementById("stat-living");
  const statContext = document.getElementById("stat-context");

  let activeController = null;

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function buildParams() {
    const params = new URLSearchParams();
    params.set("name", nameInput.value.trim());
    params.set("sex", sexSelect.value);
    params.set("region", regionSelect.value);
    if (yearInput.value.trim()) params.set("year", yearInput.value.trim());
    if (minAgeInput.value.trim()) params.set("min_age", minAgeInput.value.trim());
    if (maxAgeInput.value.trim()) params.set("max_age", maxAgeInput.value.trim());
    return params;
  }

  async function loadMetadata() {
    try {
      const resp = await fetch("/v1/metadata");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const meta = await resp.json();

      regionSelect.innerHTML = "";
      for (const code of meta.regions) {
        const opt = document.createElement("option");
        opt.value = code;
        const label = STATE_NAMES[code] ? `${code} — ${STATE_NAMES[code]}` : code;
        opt.textContent = label;
        regionSelect.appendChild(opt);
      }

      const [minYr, maxYr] = meta.mortality_year_range;
      yearInput.min = String(minYr);
      yearInput.max = String(maxYr);
      yearInput.placeholder = String(meta.latest_birth_year_us);

      const sourceTag = meta.using_national_us ? "National + State" : "State";
      badge.textContent = `SSA TR${meta.tr_year} · ${sourceTag} · thru ${meta.latest_birth_year_us}`;
    } catch {
      badge.textContent = "SSA dataset ready";
    }
  }

  async function runQuery({ syncUrl = true } = {}) {
    clearError();
    const cleanedName = nameInput.value.trim();
    if (!cleanedName) {
      showError("Enter a first name to estimate.");
      nameInput.focus();
      return;
    }

    if (activeController) activeController.abort();
    activeController = new AbortController();

    const params = buildParams();
    if (syncUrl) {
      const nextUrl = `${window.location.pathname}?${params.toString()}`;
      window.history.replaceState(null, "", nextUrl);
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Estimating…";

    try {
      const estParams = new URLSearchParams(params);
      estParams.set("include_distribution", "false");
      const resp = await fetch(`/v1/estimate?${estParams.toString()}`, {
        signal: activeController.signal,
      });

      if (!resp.ok) {
        let detail = `Request failed (HTTP ${resp.status})`;
        try {
          const errBody = await resp.json();
          if (typeof errBody.detail === "string") {
            detail = errBody.detail;
          } else if (Array.isArray(errBody.detail) && errBody.detail[0]?.msg) {
            detail = errBody.detail[0].msg;
          }
        } catch {
          // ignore non-JSON error body
        }
        resultsSection.hidden = true;
        showError(detail);
        return;
      }

      const data = await resp.json();
      renderResults(data, params);
    } catch (err) {
      if (err && err.name === "AbortError") return;
      resultsSection.hidden = true;
      showError("Could not reach the nameage API.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Estimate";
    }
  }

  function renderResults(data, params) {
    if (data.warnings && data.warnings.length > 0) {
      warningBanner.textContent = data.warnings.join(" ");
      warningBanner.hidden = false;
    } else {
      warningBanner.hidden = true;
      warningBanner.textContent = "";
    }

    const fmtInt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
    statMedian.textContent = `${data.median_age} yrs`;
    statSubMedian.textContent = `50% are age ${data.median_age} or younger`;
    statIqr.textContent = `${data.iqr[0]} – ${data.iqr[1]}`;
    statMeanMode.textContent = `${data.mean_age.toFixed(1)} / ${data.mode_age}`;
    statCi90.textContent = `90% interval: ${data.ci90[0]} – ${data.ci90[1]}`;
    statLiving.textContent = fmtInt.format(Math.round(data.expected_living));
    statContext.textContent = `${data.name} (${data.sex}, ${data.region}, ${data.reference_year})`;

    chartTitle.textContent = `${data.name} — ${data.region} (${data.sex}, ${data.reference_year})`;

    const svgParams = new URLSearchParams(params);
    svgParams.set("format", "svg");
    const svgUrl = `/v1/plot?${svgParams.toString()}`;

    const pngParams = new URLSearchParams(params);
    pngParams.set("format", "png");
    const pngUrl = `/v1/plot?${pngParams.toString()}`;

    const slug = `nameage-${data.region.toLowerCase()}-${data.name.toLowerCase()}`;
    downloadSvg.href = svgUrl;
    downloadSvg.download = `${slug}.svg`;
    downloadPng.href = pngUrl;
    downloadPng.download = `${slug}.png`;

    chartLoading.hidden = false;
    chartImg.hidden = true;
    chartImg.alt = `Estimated living age distribution for ${data.name} (${data.sex}, ${data.region}, ${data.reference_year})`;
    chartImg.onload = () => {
      chartLoading.hidden = true;
      chartImg.hidden = false;
    };
    chartImg.onerror = () => {
      chartLoading.hidden = true;
      showError("Chart could not be rendered right now. Please try again.");
    };
    chartImg.src = svgUrl;

    resultsSection.hidden = false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runQuery();
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      nameInput.value = chip.dataset.name || "";
      sexSelect.value = chip.dataset.sex || "F";
      runQuery();
    });
  });

  async function init() {
    await loadMetadata();
    const initial = new URLSearchParams(window.location.search);
    if (initial.has("name")) {
      nameInput.value = initial.get("name") || "";
      if (initial.has("sex")) sexSelect.value = initial.get("sex").toUpperCase();
      if (initial.has("region")) regionSelect.value = initial.get("region").toUpperCase();
      if (initial.has("year")) yearInput.value = initial.get("year") || "";
      if (initial.has("min_age")) minAgeInput.value = initial.get("min_age") || "";
      if (initial.has("max_age")) maxAgeInput.value = initial.get("max_age") || "";
      runQuery({ syncUrl: false });
    }
  }

  init();
})();
