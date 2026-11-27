"""FastAPI application for the nameage estimation service."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config import Settings
from api.schemas import EstimateResponse, HealthResponse, MetadataResponse
from api.store import InvalidQueryError, NameAgeStore, NameNotFoundError
from nameage.data import Sex
from nameage.plot import PlotFormat, render_age_distribution

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_MEDIA_TYPES: dict[PlotFormat, str] = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}
_SAFE_SLUG = re.compile(r"[^a-z0-9_-]+")
_PLOT_CSP: dict[PlotFormat, str] = {
    "png": "default-src 'none'",
    "pdf": "default-src 'none'",
    "svg": "default-src 'none'; style-src 'unsafe-inline'",
}

type _PlotCacheKey = tuple[
    str, Sex, str, int, int | None, int | None, PlotFormat, int, int
]


class _PlotCache:
    """Thread-safe bounded LRU cache for rendered plot bytes."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._lock = threading.Lock()
        self._entries: OrderedDict[_PlotCacheKey, bytes] = OrderedDict()

    def get(self, key: _PlotCacheKey) -> bytes | None:
        if self._max_size <= 0:
            return None
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: _PlotCacheKey, value: bytes) -> None:
        if self._max_size <= 0:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)


def _get_store(request: Request) -> NameAgeStore:
    return request.app.state.store


def _normalize_sex(raw: str) -> Sex:
    upper = raw.upper()
    if upper == "F":
        return "F"
    return "M"


def create_app(
    *,
    settings: Settings | None = None,
    store: NameAgeStore | None = None,
) -> FastAPI:
    """Construct the FastAPI application with lifespan-managed dataset loading."""
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = cfg
        application.state.store = (
            store
            if store is not None
            else NameAgeStore.from_data_dir(cfg.data_dir, tr_year=cfg.tr_year)
        )
        yield

    application = FastAPI(
        title="nameage API",
        description=(
            "Estimate U.S. living age distributions by first name using SSA "
            "baby-name birth counts and actuarial death-probability tables."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    if cfg.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["GET", "HEAD", "OPTIONS"],
            allow_headers=["*"],
        )

    @application.middleware("http")
    async def _security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    plot_semaphore = threading.BoundedSemaphore(cfg.plot_max_concurrency)
    plot_cache = _PlotCache(cfg.plot_cache_size)

    @application.exception_handler(NameNotFoundError)
    async def _handle_not_found(
        _request: Request, exc: NameNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(InvalidQueryError)
    async def _handle_invalid_query(
        _request: Request, exc: InvalidQueryError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    cache_header = f"public, max-age={cfg.cache_max_age}"

    application.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(
        data_store: Annotated[NameAgeStore, Depends(_get_store)],
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            tr_year=data_store.tr_year,
            using_national_us=data_store.using_national_us,
            regions_count=len(data_store.regions),
            latest_birth_year=data_store.latest_birth_year_us,
            mortality_year_range=data_store.mortality_year_range,
        )

    @application.get("/v1/metadata", response_model=MetadataResponse, tags=["v1"])
    def metadata(
        response: Response,
        data_store: Annotated[NameAgeStore, Depends(_get_store)],
    ) -> MetadataResponse:
        response.headers["Cache-Control"] = cache_header
        return MetadataResponse(
            tr_year=data_store.tr_year,
            using_national_us=data_store.using_national_us,
            regions=data_store.regions,
            latest_birth_year_us=data_store.latest_birth_year_us,
            latest_birth_year_state=data_store.latest_birth_year_state,
            mortality_year_range=data_store.mortality_year_range,
        )

    @application.get("/v1/estimate", response_model=EstimateResponse, tags=["v1"])
    def estimate(
        response: Response,
        data_store: Annotated[NameAgeStore, Depends(_get_store)],
        name: Annotated[str, Query(min_length=1, max_length=64)],
        sex: Annotated[str, Query(pattern="^[FfMm]$")],
        region: Annotated[str, Query(pattern="^[A-Za-z]{2}$")] = "US",
        year: Annotated[int | None, Query(ge=1880, le=2200)] = None,
        min_age: Annotated[int | None, Query(ge=0, le=120)] = None,
        max_age: Annotated[int | None, Query(ge=0, le=120)] = None,
        include_distribution: bool = True,
    ) -> EstimateResponse:
        result, warnings = data_store.estimate(
            name=name,
            sex=_normalize_sex(sex),
            region=region,
            year=year,
            min_age=min_age,
            max_age=max_age,
        )
        response.headers["Cache-Control"] = cache_header
        return EstimateResponse.from_result(
            result,
            warnings=warnings,
            include_distribution=include_distribution,
        )

    @application.get("/v1/plot", tags=["v1"])
    def plot(
        data_store: Annotated[NameAgeStore, Depends(_get_store)],
        name: Annotated[str, Query(min_length=1, max_length=64)],
        sex: Annotated[str, Query(pattern="^[FfMm]$")],
        region: Annotated[str, Query(pattern="^[A-Za-z]{2}$")] = "US",
        year: Annotated[int | None, Query(ge=1880, le=2200)] = None,
        min_age: Annotated[int | None, Query(ge=0, le=120)] = None,
        max_age: Annotated[int | None, Query(ge=0, le=120)] = None,
        fmt: Annotated[Literal["png", "svg", "pdf"], Query(alias="format")] = "png",
        plot_max_age: Annotated[int, Query(ge=1, le=120)] = 100,
        dpi: Annotated[int, Query(ge=72, le=300)] = 150,
    ) -> Response:
        result, _ = data_store.estimate(
            name=name,
            sex=_normalize_sex(sex),
            region=region,
            year=year,
            min_age=min_age,
            max_age=max_age,
        )
        cache_key: _PlotCacheKey = (
            result.name,
            result.sex,
            result.region,
            result.reference_year,
            min_age,
            max_age,
            fmt,
            plot_max_age,
            dpi,
        )
        payload = plot_cache.get(cache_key)
        if payload is None:
            if not plot_semaphore.acquire(blocking=False):
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Plot renderer is busy; please retry shortly."},
                    headers={"Retry-After": "1"},
                )
            try:
                payload = plot_cache.get(cache_key)
                if payload is None:
                    payload = render_age_distribution(
                        result,
                        fmt=fmt,
                        max_age=plot_max_age,
                        dpi=dpi,
                    )
                    plot_cache.put(cache_key, payload)
            finally:
                plot_semaphore.release()
        reg_slug = _SAFE_SLUG.sub("_", result.region.lower())
        name_slug = _SAFE_SLUG.sub("_", result.name.lower())
        filename = f"nameage-{reg_slug}-{name_slug}.{fmt}"
        return Response(
            content=payload,
            media_type=_MEDIA_TYPES[fmt],
            headers={
                "Cache-Control": cache_header,
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Security-Policy": _PLOT_CSP[fmt],
            },
        )

    return application


app = create_app()
