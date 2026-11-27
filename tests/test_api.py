"""Integration and unit tests for the nameage FastAPI web service."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from api.store import NameAgeStore
from nameage.data import death_probs_filename
from nameage.model import BirthsIndex


def _make_store(*, include_national: bool = False) -> NameAgeStore:
    state_df = pd.DataFrame(
        [
            ("CA", "F", 2025, "Lucy", 100),
            ("CA", "F", 2020, "Lucy", 300),
            ("NY", "F", 2015, "Lucy", 200),
            ("CA", "M", 2024, "John", 400),
        ],
        columns=["state", "sex", "birth_year", "name", "count"],
    )
    state_df["state"] = state_df["state"].astype("category")
    state_df["sex"] = state_df["sex"].astype("category")
    state_df["name"] = state_df["name"].astype("category")

    national_index: BirthsIndex | None = None
    if include_national:
        nat_df = pd.DataFrame(
            [
                ("US", "F", 2025, "Lucy", 1000),
                ("US", "F", 2020, "Lucy", 500),
            ],
            columns=["state", "sex", "birth_year", "name", "count"],
        )
        national_index = BirthsIndex.from_dataframe(nat_df)

    ages = list(range(20))
    dp_f = pd.DataFrame(
        [[0.01] * len(ages), [0.01] * len(ages)],
        index=[2025, 2026],
        columns=ages,
    )
    dp_f.index.name = "Year"
    dp_m = dp_f.copy()

    return NameAgeStore.from_tables(
        state_births=BirthsIndex.from_dataframe(state_df),
        national_births=national_index,
        death_probs={"F": dp_f, "M": dp_m},
        tr_year=2026,
    )


def test_health_and_metadata_endpoints() -> None:
    app = create_app(store=_make_store(include_national=True))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        h_body = health.json()
        assert h_body["status"] == "ok"
        assert h_body["using_national_us"] is True
        assert h_body["regions_count"] == 3
        assert h_body["mortality_year_range"] == [2025, 2026]

        meta = client.get("/v1/metadata")
        assert meta.status_code == 200
        assert "public, max-age=" in meta.headers["cache-control"]
        m_body = meta.json()
        assert m_body["regions"] == ["US", "CA", "NY"]
        assert m_body["latest_birth_year_us"] == 2025


def test_estimate_endpoint_json_and_filtering() -> None:
    app = create_app(store=_make_store())
    with TestClient(app) as client:
        resp = client.get(
            "/v1/estimate",
            params={"name": "lucy", "sex": "f", "region": "ca", "min_age": 1},
        )
        assert resp.status_code == 200
        assert "public, max-age=" in resp.headers["cache-control"]
        body = resp.json()
        assert body["name"] == "Lucy"
        assert body["sex"] == "F"
        assert body["region"] == "CA"
        assert body["median_age"] == 5
        assert body["warnings"] == []
        assert body["distribution"] is not None
        assert all(pt["age"] >= 1 for pt in body["distribution"])

        # Summary-only omits distribution array
        summary = client.get(
            "/v1/estimate",
            params={
                "name": "Lucy",
                "sex": "F",
                "include_distribution": "false",
            },
        )
        assert summary.status_code == 200
        assert summary.json()["distribution"] is None


def test_estimate_prefers_national_dataset_for_us_only() -> None:
    app = create_app(store=_make_store(include_national=True))
    with TestClient(app) as client:
        us = client.get("/v1/estimate", params={"name": "Lucy", "sex": "F"})
        ca = client.get(
            "/v1/estimate", params={"name": "Lucy", "sex": "F", "region": "CA"}
        )
        assert us.status_code == 200
        assert ca.status_code == 200
        # National table has 1000 born in 2025 -> median age 0
        assert us.json()["median_age"] == 0
        # State table for CA has 300 in 2020 vs 100 in 2025 -> median age 5
        assert ca.json()["median_age"] == 5


def test_estimate_future_year_warning_and_error_codes() -> None:
    app = create_app(store=_make_store())
    with TestClient(app) as client:
        future = client.get(
            "/v1/estimate",
            params={"name": "Lucy", "sex": "F", "year": 2026},
        )
        assert future.status_code == 200
        assert len(future.json()["warnings"]) == 1
        assert "after latest birth year 2025" in future.json()["warnings"][0]

        missing_name = client.get("/v1/estimate", params={"name": "Zzzzz", "sex": "F"})
        assert missing_name.status_code == 404

        bad_region = client.get(
            "/v1/estimate", params={"name": "Lucy", "sex": "F", "region": "ZZ"}
        )
        assert bad_region.status_code == 422

        bad_year = client.get(
            "/v1/estimate", params={"name": "Lucy", "sex": "F", "year": 2099}
        )
        assert bad_year.status_code == 422

        bad_ages = client.get(
            "/v1/estimate",
            params={"name": "Lucy", "sex": "F", "min_age": 10, "max_age": 5},
        )
        assert bad_ages.status_code == 422


def test_plot_endpoint_formats_and_concurrency() -> None:
    app = create_app(store=_make_store())
    with TestClient(app) as client:
        for fmt, media_type, magic in (
            ("png", "image/png", b"\x89PNG"),
            ("svg", "image/svg+xml", b"<svg"),
            ("pdf", "application/pdf", b"%PDF"),
        ):
            resp = client.get(
                "/v1/plot",
                params={"name": "Lucy", "sex": "F", "format": fmt, "dpi": 100},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith(media_type)
            assert magic in resp.content[:256]

        def _fetch_png(_: int) -> bytes:
            r = client.get("/v1/plot", params={"name": "Lucy", "sex": "F", "dpi": 72})
            assert r.status_code == 200
            return r.content

        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(_fetch_png, range(4)))
        assert all(out.startswith(b"\x89PNG") for out in outputs)


def test_lifespan_loads_from_data_dir_and_fails_fast_when_missing(
    tmp_path: Path,
) -> None:
    # Missing data dir fails fast on startup
    missing_app = create_app(settings=Settings(data_dir=tmp_path / "missing"))
    with pytest.raises(FileNotFoundError), TestClient(missing_app):
        pass

    # Seeded data dir loads both sexes cleanly via lifespan
    data_dir = tmp_path / "ds"
    (data_dir / "namesbystate").mkdir(parents=True)
    (data_dir / "namesbystate" / "CA.TXT").write_text(
        "CA,F,2025,Ada,250\nCA,M,2025,Alan,300\n", encoding="utf-8"
    )
    for sex in ("F", "M"):
        (data_dir / death_probs_filename(sex, 2026)).write_text(
            "Title\nYear,0,1,2\n2025,0.01,0.01,0.01\n", encoding="utf-8"
        )

    app = create_app(settings=Settings(data_dir=data_dir, tr_year=2026))
    with TestClient(app) as client:
        r = client.get("/v1/estimate", params={"name": "Ada", "sex": "F"})
        assert r.status_code == 200
        assert r.json()["median_age"] == 0


def test_security_headers_and_plot_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    from api import app as api_app

    real_render = api_app.render_age_distribution

    def _counting_render(*args: object, **kwargs: object) -> bytes:
        nonlocal calls
        calls += 1
        return real_render(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api_app, "render_age_distribution", _counting_render)

    app = create_app(
        settings=Settings(plot_cache_size=4),
        store=_make_store(),
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.headers["x-content-type-options"] == "nosniff"

        svg1 = client.get(
            "/v1/plot", params={"name": "Lucy", "sex": "F", "format": "svg"}
        )
        assert svg1.status_code == 200
        assert svg1.headers["x-content-type-options"] == "nosniff"
        assert (
            svg1.headers["content-security-policy"]
            == "default-src 'none'; style-src 'unsafe-inline'"
        )

        # Second identical request hits the LRU cache without re-rendering.
        svg2 = client.get(
            "/v1/plot", params={"name": "lucy", "sex": "f", "format": "svg"}
        )
        assert svg2.status_code == 200
        assert svg2.content == svg1.content
        assert calls == 1

        png = client.get(
            "/v1/plot", params={"name": "Lucy", "sex": "F", "format": "png"}
        )
        assert png.status_code == 200
        assert png.headers["content-security-policy"] == "default-src 'none'"
        assert calls == 2


def test_plot_concurrency_cap_returns_503_without_starving_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import app as api_app

    entered = threading.Event()
    release = threading.Event()

    def _slow_render(*_args: object, **_kwargs: object) -> bytes:
        entered.set()
        release.wait(timeout=5.0)
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(api_app, "render_age_distribution", _slow_render)

    app = create_app(
        settings=Settings(plot_max_concurrency=1, plot_cache_size=0),
        store=_make_store(),
    )
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.get, "/v1/plot", params={"name": "Lucy", "sex": "F"})
        assert entered.wait(timeout=5.0)
        try:
            second = client.get("/v1/plot", params={"name": "John", "sex": "M"})
            assert second.status_code == 503
            assert second.headers["retry-after"] == "1"
            assert second.headers["x-content-type-options"] == "nosniff"

            # Health and estimate remain responsive while plot renderer is saturated.
            assert client.get("/health").status_code == 200
            assert (
                client.get(
                    "/v1/estimate", params={"name": "Lucy", "sex": "F"}
                ).status_code
                == 200
            )
        finally:
            release.set()
        assert first.result(timeout=5.0).status_code == 200
