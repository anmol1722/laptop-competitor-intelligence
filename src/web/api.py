"""FastAPI backend for the laptop competitor-intelligence system.

The backend modules (``config``, ``pricing``, ``matching``, ``sentiment``, ``rag``)
already do the analysis; this layer does three things and nothing else:

1.  Loads the parquet artifacts **once** at startup (FastAPI lifespan) into
    module-level frames, plus a small in-memory search index, so no request
    touches the disk.
2.  Sanitises every payload before it reaches the socket.  pandas/numpy scalars,
    ``NaN``/``NaT``/``Inf`` and Timestamps are converted to plain JSON types;
    ``NaN`` becomes ``null`` because ``json.dumps`` would otherwise emit the
    literal ``NaN``, which is not valid JSON and breaks ``JSON.parse``.
3.  Keeps the 7B LLM **out** of startup.  ``rag`` is imported and the weights are
    materialised on the first ``POST /api/chat``, behind a GPU-availability check,
    so the server boots in a couple of seconds and an unavailable GPU surfaces as
    a structured 503 rather than a stack trace.

Honesty rules baked into the responses (these are data realities, not bugs):
  * ~69% of products have no price.  Those come back as ``price: null`` with
    ``price_available: false`` and the display string "price not listed" - never 0.
  * Only ~26.5% of products have mined review sentiment.  Those come back with
    ``no_sentiment_data: true`` and a reason, never an empty/zero chart.
  * Every market statistic carries ``n`` and ``coverage``; brand/segment rows keep
    the ``reliable`` flag (n < 5 priced products) from ``pricing.py``.

Run::

    .venv/bin/python -m uvicorn src.web.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# Environment guard - MUST run before pandas (and therefore pyarrow) is imported.
# pyarrow's default mimalloc pool keeps a per-thread heap that is torn down when the
# thread that first touched it exits.  FastAPI runs sync endpoints in an anyio worker
# thread pool whose threads come and go, which is exactly the pattern that segfaults the
# interpreter inside pyarrow (the same reason this project is not on Streamlit).  Arrow's
# plain system allocator has no thread-local heap; the cost is negligible at this size.
# --------------------------------------------------------------------------------------
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

_SRC = Path(__file__).resolve().parent.parent          # .../project/src
_ROOT = _SRC.parent                                     # .../project
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import asdict, is_dataclass  # noqa: E402
from datetime import date, datetime, time as _time, timedelta  # noqa: E402
from decimal import Decimal  # noqa: E402
from functools import lru_cache  # noqa: E402
from typing import Any, Literal  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import Body, FastAPI, HTTPException, Path as PathParam, Query, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
import matching  # noqa: E402
import pricing  # noqa: E402
import sentiment  # noqa: E402
from config import ASPECTS, PRODUCT_KEY, SEGMENTS  # noqa: E402

LOG = logging.getLogger("web.api")

API_VERSION = "1.0.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: How much VRAM (MiB) another process may hold before we refuse to load the 7B LLM.
GPU_BUSY_MIB = 2000
#: Waiting policy when the GPU is busy at chat time.
GPU_WAIT_ATTEMPTS = 3
GPU_WAIT_SECONDS = 4.0

PRICE_NOT_LISTED = "price not listed"

ASPECT_NAMES: list[str] = list(ASPECTS.keys())

SortKey = Literal[
    "relevance", "price_asc", "price_desc", "rating_desc",
    "reviews_desc", "ram_desc", "title_asc",
]


# ======================================================================================
# 1. Strict JSON sanitizer
# ======================================================================================

_MAX_DEPTH = 24


def _clean_float(value: float) -> float | None:
    """NaN / +-Inf -> None, everything else -> plain float."""
    f = float(value)
    return f if math.isfinite(f) else None


def sanitize(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert *obj* into something ``json.dumps(allow_nan=False)`` accepts.

    Handles the whole zoo that pandas 3 / numpy / pyarrow hand back:

    * ``np.int64`` / ``np.float32`` / ``np.bool_``           -> ``int`` / ``float`` / ``bool``
    * ``NaN`` / ``Inf`` / ``-Inf`` / ``NaT`` / ``pd.NA``     -> ``None``
    * ``pd.Timestamp`` / ``datetime`` / ``date`` / ``time``  -> ISO-8601 ``str``
    * ``timedelta`` / ``np.timedelta64``                     -> seconds (``float``)
    * ``np.ndarray`` / ``pd.Series`` / ``pd.DataFrame``      -> ``list`` / ``dict`` / records
    * dataclasses and objects exposing ``to_dict()``         -> ``dict``
    * dict keys                                              -> ``str``

    Anything unrecognised degrades to ``str(obj)`` rather than exploding at render
    time, so a stray object can never take an endpoint down.
    """
    if _depth > _MAX_DEPTH:
        return str(obj)

    # -- fast path: exact JSON-native scalars -----------------------------------------
    if obj is None:
        return None
    if obj is True or obj is False:
        return bool(obj)
    t = type(obj)
    if t is str:
        return obj
    if t is int:
        return obj
    if t is float:
        return _clean_float(obj)

    # -- pandas / numpy missing sentinels ---------------------------------------------
    if obj is pd.NaT or obj is pd.NA:
        return None

    # -- numpy scalars ------------------------------------------------------------------
    if isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _clean_float(float(obj))
        if isinstance(obj, np.datetime64):
            ts = pd.Timestamp(obj)
            return None if pd.isna(ts) else ts.isoformat()
        if isinstance(obj, np.timedelta64):
            td = pd.Timedelta(obj)
            return None if pd.isna(td) else td.total_seconds()
        if isinstance(obj, np.bytes_):
            return bytes(obj).decode("utf-8", "replace")
        if isinstance(obj, np.str_):
            return str(obj)
        return sanitize(obj.item(), _depth + 1)

    # -- python numeric / temporal ------------------------------------------------------
    if isinstance(obj, bool):          # numpy-free subclasses of bool
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, Decimal):
        return _clean_float(float(obj))
    if isinstance(obj, (pd.Timestamp, datetime)):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, (date, _time)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, timedelta)):
        return None if pd.isna(obj) else obj.total_seconds()

    # -- pandas containers --------------------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        return [sanitize(rec, _depth + 1) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): sanitize(v, _depth + 1) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.Index):
        return [sanitize(v, _depth + 1) for v in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [sanitize(v, _depth + 1) for v in obj.tolist()]

    # -- generic containers -------------------------------------------------------------
    if isinstance(obj, dict):
        return {str(k): sanitize(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize(v, _depth + 1) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", "replace")

    # -- structured objects -------------------------------------------------------------
    if is_dataclass(obj) and not isinstance(obj, type):
        return sanitize(asdict(obj), _depth + 1)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return sanitize(to_dict(), _depth + 1)
        except Exception:  # pragma: no cover - defensive
            pass

    try:
        if pd.isna(obj):        # scalar-only; arrays already handled above
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


class SafeJSONResponse(JSONResponse):
    """JSONResponse that sanitises first and then refuses to emit ``NaN``.

    ``allow_nan=False`` is the belt-and-braces half: if the sanitizer ever missed a
    non-finite float the render raises here (a loud 500 in the log) instead of
    shipping invalid JSON that dies silently in ``JSON.parse``.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            sanitize(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


# ======================================================================================
# 2. Module-level state, loaded once in the lifespan
# ======================================================================================


class AppState:
    """Everything loaded once at startup and shared by every request."""

    def __init__(self) -> None:
        self.started_at: float = time.time()
        self.load_seconds: float | None = None
        self.products: pd.DataFrame | None = None
        self.errors: dict[str, str] = {}
        self.module_status: dict[str, dict[str, Any]] = {}

        # search index (aligned, positional with self.products)
        self.n: int = 0
        self.asins: list[str] = []
        self.titles: list[str] = []
        self.blob: list[str] = []               # lowercased title+brand+store+gpu+cpu+asin
        self.title_lower: list[str] = []
        self.brand_lower: np.ndarray | None = None
        self.segment_arr: np.ndarray | None = None
        self.price: np.ndarray | None = None
        self.ram: np.ndarray | None = None
        self.discrete: np.ndarray | None = None
        self.renewed: np.ndarray | None = None
        self.rating: np.ndarray | None = None
        self.rating_number: np.ndarray | None = None
        self.n_reviews: np.ndarray | None = None
        self.pos_of: dict[str, int] = {}

        # sentiment coverage
        self.sentiment_asins: set[str] = set()
        self.n_product_sentiment: int = 0
        self.n_review_sentiment: int = 0

        # artifact bookkeeping
        self.artifacts: list[dict[str, Any]] = []
        self.n_reviews_total: int | None = None

        # background warm-up of the expensive caches
        self.warm_state: str = "cold"
        self.warm_seconds: float | None = None
        self.warm_error: str | None = None

    # -- helpers ------------------------------------------------------------------------
    def require_products(self) -> pd.DataFrame:
        if self.products is None:
            raise HTTPException(status_code=503, detail="products.parquet is not loaded")
        return self.products

    def row_for(self, asin: str) -> pd.Series:
        """Positional lookup of one product, or 404."""
        df = self.require_products()
        pos = self.pos_of.get(asin)
        if pos is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown parent_asin {asin!r}; not present in products.parquet",
            )
        return df.iloc[pos]


STATE = AppState()

_TOKEN_RE = re.compile(r"[a-z0-9\.\-\+]+")


def _artifact_record(name: str, path: Path) -> dict[str, Any]:
    """Existence / size / row-count for one on-disk artifact (rows read from metadata)."""
    rec: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "size_mb": None,
        "rows": None,
        "modified": None,
    }
    if not rec["exists"]:
        return rec
    st = path.stat()
    rec["size_mb"] = round(st.st_size / 1e6, 2)
    rec["modified"] = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    if path.suffix == ".parquet":
        try:  # parquet footer only - never reads the 150 MB review body
            import pyarrow.parquet as pq

            rec["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception as exc:  # pragma: no cover
            rec["error"] = str(exc)
    return rec


def _build_state() -> None:
    """Read every artifact once and build the in-memory search index."""
    t0 = time.time()
    st = STATE

    st.module_status = {}
    for name, mod in (("config", config), ("pricing", pricing),
                      ("matching", matching), ("sentiment", sentiment)):
        st.module_status[name] = {
            "imported": True,
            "file": getattr(mod, "__file__", None),
            "error": None,
        }
    # rag is deliberately NOT imported at startup (it pulls torch/transformers).
    st.module_status["rag"] = {
        "imported": "rag" in sys.modules,
        "file": str(_SRC / "rag.py"),
        "error": None,
        "note": "imported lazily on the first /api/chat request",
    }

    st.artifacts = [
        _artifact_record("products", config.PRODUCTS_PARQUET),
        _artifact_record("reviews", config.REVIEWS_PARQUET),
        _artifact_record("product_sentiment", config.PRODUCT_SENTIMENT_PARQUET),
        _artifact_record("review_sentiment", config.REVIEW_SENTIMENT_PARQUET),
        _artifact_record("product_embeddings", config.EMBEDDINGS_NPY),
        _artifact_record("product_embedding_ids", config.EMBED_IDS_JSON),
    ]
    st.n_reviews_total = next(
        (a["rows"] for a in st.artifacts if a["name"] == "reviews"), None)

    # -- products (shared with pricing.py's memoised cache) -----------------------------
    df = pricing.load_products()
    st.products = df
    st.n = int(len(df))
    st.asins = [str(x) for x in df[PRODUCT_KEY].tolist()]
    st.pos_of = {a: i for i, a in enumerate(st.asins)}
    st.titles = [str(x) for x in df["title"].tolist()]

    brands = [str(x) for x in df["brand"].tolist()]
    stores = [str(x) for x in df["store"].tolist()]
    gpus = [str(x) for x in df["gpu_model"].tolist()]
    cpus = [str(x) for x in df["cpu_family"].tolist()]
    st.title_lower = [t.lower() for t in st.titles]
    st.blob = [
        f"{tl} | {b} | {s} | {g} | {c} | {a}".lower()
        for tl, b, s, g, c, a in zip(st.title_lower, brands, stores, gpus, cpus, st.asins)
    ]

    st.brand_lower = np.array([b.lower() for b in brands], dtype=object)
    st.segment_arr = np.array([str(x) for x in df["segment"].tolist()], dtype=object)
    st.price = df["price"].to_numpy(dtype="float64", na_value=np.nan)
    st.ram = df["ram_gb"].to_numpy(dtype="float64", na_value=np.nan)
    st.discrete = df["is_discrete_gpu"].to_numpy(dtype=bool)
    st.renewed = df["is_renewed"].to_numpy(dtype=bool)
    st.rating = df["average_rating"].to_numpy(dtype="float64", na_value=np.nan)
    st.rating_number = df["rating_number"].to_numpy(dtype="int64")
    st.n_reviews = df["n_reviews"].to_numpy(dtype="int64")

    # -- sentiment coverage --------------------------------------------------------------
    try:
        ps = sentiment.load_product_sentiment()
        st.sentiment_asins = set(str(a) for a in ps[PRODUCT_KEY].tolist())
        st.n_product_sentiment = int(len(ps))
    except Exception as exc:
        st.errors["product_sentiment"] = f"{type(exc).__name__}: {exc}"
        st.module_status["sentiment"]["error"] = st.errors["product_sentiment"]
    try:
        st.n_review_sentiment = int(len(sentiment.load_review_sentiment()))
    except Exception as exc:
        st.errors["review_sentiment"] = f"{type(exc).__name__}: {exc}"

    st.load_seconds = time.time() - t0
    LOG.info("state loaded in %.2fs (%d products, %d with sentiment)",
             st.load_seconds, st.n, len(st.sentiment_asins))


def _warm_caches() -> None:
    """Warm the expensive lru_caches off the request path (background thread)."""
    st = STATE
    st.warm_state = "warming"
    t0 = time.time()
    try:
        matching.get_matcher()          # ~0.9 s: parquet + 40 MB embedding matrix
        _segment_table()                # ~0.1 s
        _brand_table()                  # ~2.7 s
        _market_overview()              # ~1.5 s
        st.warm_state = "warm"
    except Exception as exc:            # pragma: no cover - warming must never kill the app
        st.warm_state = "error"
        st.warm_error = f"{type(exc).__name__}: {exc}"
        LOG.exception("cache warm-up failed")
    st.warm_seconds = round(time.time() - t0, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load parquet once on startup; release the frames on shutdown."""
    try:
        _build_state()
    except Exception as exc:  # pragma: no cover - a broken artifact must still boot /health
        STATE.errors["startup"] = f"{type(exc).__name__}: {exc}"
        LOG.exception("startup load failed")
    threading.Thread(target=_warm_caches, name="warm-caches", daemon=True).start()
    yield
    STATE.products = None
    LOG.info("shutdown")


# ======================================================================================
# 3. Cached aggregate tables (pure pricing.py output, kept intact)
# ======================================================================================


@lru_cache(maxsize=1)
def _segment_table() -> list[dict[str, Any]]:
    tbl = pricing.segment_price_table()
    rows = tbl.to_dict(orient="records")
    for r in rows:
        seg = str(r["segment"])
        n_products = int(r["n_products"])
        r["n_priced"] = int(r["n"])                 # explicit alias next to pricing's `n`
        r["n_unpriced"] = n_products - int(r["n"])
        r["unreliable"] = not bool(r["reliable"])
        r["share_of_catalogue"] = (n_products / STATE.n) if STATE.n else None
        seg_asins = [a for a, s in zip(STATE.asins, STATE.segment_arr) if s == seg]
        with_sent = sum(1 for a in seg_asins if a in STATE.sentiment_asins)
        r["n_with_sentiment"] = int(with_sent)
        r["sentiment_coverage"] = (with_sent / n_products) if n_products else 0.0
    return rows


@lru_cache(maxsize=1)
def _brand_table() -> list[dict[str, Any]]:
    tbl = pricing.brand_price_table(min_products=1)
    rows = tbl.to_dict(orient="records")
    for r in rows:
        n_products = int(r["n_products"])
        r["n_priced"] = int(r["n"])
        r["n_unpriced"] = n_products - int(r["n"])
        # pricing.py flags a group as unreliable when it has < MIN_RELIABLE_N priced rows
        r["unreliable"] = not bool(r["reliable"])
        r["share_of_catalogue"] = (n_products / STATE.n) if STATE.n else None
    return rows


@lru_cache(maxsize=1)
def _market_overview() -> dict[str, Any]:
    df = STATE.require_products()
    n = int(len(df))
    price_all = pricing.price_summary(df["price"], n_total=n, label="catalogue")

    segments = _segment_table()
    brands = _brand_table()
    top_brands = sorted(brands, key=lambda r: -int(r["n_products"]))[:12]

    overview: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "catalogue": {
            "n_products": n,
            "n_brands": int(df["brand"].nunique()),
            "n_segments": int(df["segment"].nunique()),
            "n_reviews_retained": int(df["n_reviews"].sum()),
            "n_reviews_in_parquet": STATE.n_reviews_total,
            "n_renewed": int(df["is_renewed"].sum()),
            "renewed_share": float(df["is_renewed"].mean()),
            "n_discrete_gpu": int(df["is_discrete_gpu"].sum()),
            "discrete_gpu_share": float(df["is_discrete_gpu"].mean()),
            "mean_rating": (float(df["average_rating"].mean())
                            if df["average_rating"].notna().any() else None),
            "total_ratings": int(df["rating_number"].sum()),
        },
        "price": {
            **price_all,
            "n_priced": int(price_all["n"]),
            "n_unpriced": n - int(price_all["n"]),
            "min_reliable_n": int(pricing.MIN_RELIABLE_N),
        },
        "sentiment": {
            "n_products_with_sentiment": len(STATE.sentiment_asins),
            "n_products": n,
            "coverage": (len(STATE.sentiment_asins) / n) if n else 0.0,
            "n_reviews_scored": STATE.n_review_sentiment,
            "n_reviews_available": STATE.n_reviews_total,
            "note": (
                "Review sentiment has been mined for a sample of the catalogue; a full "
                "pass is queued. Products outside the sample report no_sentiment_data "
                "rather than a zeroed chart."
            ),
        },
        "segments": [
            {
                "segment": r["segment"], "n_products": r["n_products"],
                "share_of_catalogue": r["share_of_catalogue"],
                "n_priced": r["n_priced"], "coverage": r["coverage"],
                "reliable": r["reliable"], "median": r["median"],
                "p25": r["p25"], "p75": r["p75"], "mean_rating": r["mean_rating"],
                "sentiment_coverage": r["sentiment_coverage"],
            }
            for r in segments
        ],
        "top_brands": [
            {
                "brand": r["brand"], "n_products": r["n_products"],
                "n_priced": r["n_priced"], "coverage": r["coverage"],
                "reliable": r["reliable"], "unreliable": r["unreliable"],
                "median": r["median"], "mean_rating": r["mean_rating"],
            }
            for r in top_brands
        ],
        "caveats": [
            f"{100.0 * (1 - price_all['coverage']):.0f}% of the {n:,} listings have no "
            f"price in the source data; every price statistic here describes the "
            f"{price_all['n']:,} priced listings only.",
            f"Review sentiment exists for {len(STATE.sentiment_asins):,} of {n:,} "
            f"products ({100.0 * len(STATE.sentiment_asins) / n if n else 0:.1f}%).",
            f"Groups with fewer than {pricing.MIN_RELIABLE_N} priced products are "
            f"flagged unreliable and their medians should not be quoted.",
        ],
    }

    # scipy-backed extras: never let a missing optional dep take the dashboard down
    try:
        gpu = pricing.discrete_gpu_premium()
        rows = gpu.to_dict(orient="records")
        overview["discrete_gpu_premium"] = {
            "overall": next((r for r in rows if r["segment"] == "ALL"), None),
            "by_segment": [r for r in rows if r["segment"] != "ALL"],
        }
    except Exception as exc:
        overview["discrete_gpu_premium"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    try:
        pr = pricing.price_rating_relationship()
        rows = pr.to_dict(orient="records")
        overview["price_vs_rating"] = {
            "overall": next((r for r in rows if r["segment"] == "ALL"), None),
            "by_segment": [r for r in rows if r["segment"] != "ALL"],
        }
    except Exception as exc:
        overview["price_vs_rating"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    try:
        overview["coverage_bias"] = pricing.coverage_bias()
    except Exception as exc:
        overview["coverage_bias"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    try:
        vo = pricing.value_outliers(top=5)
        best, worst = vo["best_value"], vo["overpriced"]
        overview["value"] = {
            "price_model": best.attrs.get("price_model"),
            "n_pool": best.attrs.get("n_pool"),
            "best_value": best.to_dict(orient="records"),
            "overpriced": worst.to_dict(orient="records"),
            "note": ("Value scores compare the listed price with a spec-based price model "
                     "fitted on priced listings only."),
        }
    except Exception as exc:
        overview["value"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    return overview


# ======================================================================================
# 4. Product shaping helpers
# ======================================================================================

_SPEC_FIELDS = [
    "cpu_brand", "cpu_family", "cpu_tier", "cpu_ghz", "ram_gb", "ram_type",
    "storage_gb", "storage_type", "screen_in", "screen_w", "screen_h",
    "gpu_brand", "gpu_model", "is_discrete_gpu", "os_family", "weight_lb",
]


def _price_display(price: Any) -> str:
    """Human string for a price cell; never '0' and never blank for a missing price."""
    if price is None or (isinstance(price, float) and not math.isfinite(price)):
        return PRICE_NOT_LISTED
    try:
        return f"${float(price):,.2f}"
    except (TypeError, ValueError):
        return PRICE_NOT_LISTED


def _scalar(value: Any) -> Any:
    """One cell -> JSON scalar (NaN/NaT -> None)."""
    return sanitize(value)


def _specs_of(row: pd.Series) -> dict[str, Any]:
    specs = {k: _scalar(row.get(k)) for k in _SPEC_FIELDS}
    w, h = specs.get("screen_w"), specs.get("screen_h")
    specs["screen_res"] = f"{int(w)}x{int(h)}" if w and h else None
    return specs


def _card(pos: int) -> dict[str, Any]:
    """Compact product card used by search results (built from the numpy index)."""
    st = STATE
    df = st.require_products()
    row = df.iloc[pos]
    price = _clean_float(st.price[pos])
    asin = st.asins[pos]
    return {
        PRODUCT_KEY: asin,
        "title": st.titles[pos],
        "brand": str(row["brand"]),
        "store": str(row["store"]),
        "segment": str(row["segment"]),
        "price": price,
        "price_available": price is not None,
        "price_display": _price_display(price),
        "is_renewed": bool(st.renewed[pos]),
        "specs": _specs_of(row),
        "average_rating": _clean_float(st.rating[pos]),
        "rating_number": int(st.rating_number[pos]),
        "n_reviews": int(st.n_reviews[pos]),
        "has_sentiment": asin in st.sentiment_asins,
    }


def _competitor_record(row: pd.Series) -> dict[str, Any]:
    """One matching.find_competitors row -> UI shape (price, specs, rating, score)."""
    price = _scalar(row.get("price"))
    eff = _scalar(row.get("price_effective"))
    asin = str(row[PRODUCT_KEY])
    return {
        PRODUCT_KEY: asin,
        "title": str(row["title"]),
        "brand": str(row["brand"]),
        "segment": str(row["segment"]),
        "price": price,
        "price_available": price is not None,
        "price_display": _price_display(price),
        "price_effective": eff,
        "price_is_estimated": bool(row.get("price_is_estimated", False)),
        "price_effective_note": (
            None if price is not None else
            "no listed price; the matcher used a hierarchical estimate for the price guard"
        ),
        "specs": {
            "cpu_brand": _scalar(row.get("cpu_brand")),
            "cpu_family": _scalar(row.get("cpu_family")),
            "cpu_tier": _scalar(row.get("cpu_tier")),
            "ram_gb": _scalar(row.get("ram_gb")),
            "storage_gb": _scalar(row.get("storage_gb")),
            "storage_type": _scalar(row.get("storage_type")),
            "screen_in": _scalar(row.get("screen_in")),
            "gpu_model": _scalar(row.get("gpu_model")),
            "is_discrete_gpu": bool(row.get("is_discrete_gpu", False)),
            "os_family": _scalar(row.get("os_family")),
        },
        "is_renewed": bool(row.get("is_renewed", False)),
        "average_rating": _scalar(row.get("average_rating")),
        "rating_number": int(row.get("rating_number", 0) or 0),
        "n_reviews": int(row.get("n_reviews", 0) or 0),
        "has_sentiment": asin in STATE.sentiment_asins,
        "score": _scalar(row.get("score")),
        "similarity": {
            "score": _scalar(row.get("score")),
            "text_sim": _scalar(row.get("text_sim")),
            "spec_sim": _scalar(row.get("spec_sim")),
            "segment_affinity": _scalar(row.get("segment_affinity")),
            "price_ratio": _scalar(row.get("price_ratio")),
        },
    }


# ======================================================================================
# 5. The app
# ======================================================================================

app = FastAPI(
    title="Laptop Competitor Intelligence API",
    version=API_VERSION,
    description=(
        "JSON backend over the laptop catalogue: search, competitor matching, price "
        "positioning, review-sentiment evidence and a grounded RAG chat endpoint. "
        "Missing prices and missing sentiment are reported explicitly, never zeroed."
    ),
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local dev: the static UI may be served from any port
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException) -> SafeJSONResponse:
    """Structured error body (never a stack trace)."""
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status": exc.status_code,
                "type": "http_error",
                "message": exc.detail,
                "path": str(request.url.path),
            },
            "detail": exc.detail,
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> SafeJSONResponse:
    """422 for bad query/body params, with the offending fields listed."""
    return SafeJSONResponse(
        status_code=422,
        content={
            "error": {
                "status": 422,
                "type": "validation_error",
                "message": "invalid request parameters",
                "path": str(request.url.path),
            },
            "detail": sanitize(exc.errors()),
        },
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception) -> SafeJSONResponse:
    """Anything unexpected becomes a structured 500; the trace stays in the log."""
    LOG.error("unhandled error on %s\n%s", request.url.path, traceback.format_exc())
    return SafeJSONResponse(
        status_code=500,
        content={
            "error": {
                "status": 500,
                "type": type(exc).__name__,
                "message": str(exc),
                "path": str(request.url.path),
            },
            "detail": str(exc),
        },
    )


# --------------------------------------------------------------------------------------
# /api/health
# --------------------------------------------------------------------------------------


@app.get("/api/health", summary="Module load status, row counts and artifact inventory")
def health() -> dict[str, Any]:
    st = STATE
    n = st.n
    priced = int(np.isfinite(st.price).sum()) if st.price is not None else 0
    ok = st.products is not None and not st.errors
    return {
        "status": "ok" if ok else "degraded",
        "api_version": API_VERSION,
        "uptime_s": round(time.time() - st.started_at, 1),
        "startup_load_s": st.load_seconds,
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "modules": st.module_status,
        "artifacts": st.artifacts,
        "rows": {
            "products": n,
            "reviews": st.n_reviews_total,
            "product_sentiment": st.n_product_sentiment,
            "review_sentiment": st.n_review_sentiment,
        },
        "coverage": {
            "price": {
                "n_priced": priced,
                "n_total": n,
                "coverage": (priced / n) if n else 0.0,
                "note": "products without a price render as 'price not listed', never 0",
            },
            "sentiment": {
                "n_with_sentiment": len(st.sentiment_asins),
                "n_total": n,
                "coverage": (len(st.sentiment_asins) / n) if n else 0.0,
                "note": "a full sentiment pass is queued; uncovered products say so",
            },
        },
        "caches": {
            "state": st.warm_state,
            "seconds": st.warm_seconds,
            "error": st.warm_error,
            "matcher_loaded": matching.get_matcher.cache_info().currsize > 0,
        },
        "llm": LLM.status(),
        "static_dir": {"path": str(STATIC_DIR), "exists": STATIC_DIR.is_dir(),
                       "files": len(list(STATIC_DIR.glob("*"))) if STATIC_DIR.is_dir() else 0},
        "errors": st.errors or None,
    }


# --------------------------------------------------------------------------------------
# /api/products/search
# --------------------------------------------------------------------------------------


def _match_indices(q: str | None, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forgiving text match over title+brand+store+gpu+cpu+asin.

    Every token of *q* must appear as a substring somewhere in the product blob (AND
    semantics, substring-tolerant so "thinkpad t14" and "16gb" both work).  The score
    prefers exact phrase hits in the title, then title-token hits, then blob hits, with
    a small popularity tiebreak so identical matches come back in a stable order.
    """
    if not q or not q.strip():
        return candidates, np.zeros(candidates.size, dtype="float64")

    ql = q.strip().lower()
    tokens = _TOKEN_RE.findall(ql)
    if not tokens:
        return candidates, np.zeros(candidates.size, dtype="float64")

    blob, title = STATE.blob, STATE.title_lower
    keep: list[int] = []
    scores: list[float] = []
    phrase = ql if len(tokens) > 1 else None
    for i in candidates:
        b = blob[i]
        s = 0.0
        hit = True
        for tok in tokens:
            if tok in b:
                s += 2.0 if tok in title[i] else 1.0
            else:
                hit = False
                break
        if not hit:
            continue
        if phrase is not None and phrase in title[i]:
            s += 6.0
        elif tokens[0] in title[i][:60]:
            s += 1.0
        keep.append(int(i))
        scores.append(s)

    if not keep:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="float64")
    idx = np.array(keep, dtype="int64")
    sc = np.array(scores, dtype="float64")
    sc += np.log1p(STATE.rating_number[idx].astype("float64")) / 25.0
    return idx, sc


@app.get("/api/products/search", summary="Paginated product search with spec filters")
def search_products(
    q: str | None = Query(None, max_length=200, description="free text over title/brand/model"),
    brand: str | None = Query(None, max_length=200, description="comma-separated brand names"),
    segment: str | None = Query(None, max_length=200,
                                description=f"comma-separated; one of {SEGMENTS}"),
    min_price: float | None = Query(None, ge=0, le=100_000),
    max_price: float | None = Query(None, ge=0, le=100_000),
    min_ram: float | None = Query(None, ge=0, le=1024, description="minimum RAM in GB"),
    has_discrete_gpu: bool | None = Query(None),
    is_renewed: bool | None = Query(None),
    min_rating: float | None = Query(None, ge=0, le=5),
    has_sentiment: bool | None = Query(None, description="only products with mined reviews"),
    sort: SortKey = Query("relevance"),
    limit: int = Query(24, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    t0 = time.time()
    st = STATE
    st.require_products()
    notes: list[str] = []

    if min_price is not None and max_price is not None and min_price > max_price:
        raise HTTPException(422, "min_price must be <= max_price")

    mask = np.ones(st.n, dtype=bool)

    if brand:
        wanted = {b.strip().lower() for b in brand.split(",") if b.strip()}
        if wanted:
            mask &= np.isin(st.brand_lower, list(wanted))
    if segment:
        wanted_seg = [s.strip().lower() for s in segment.split(",") if s.strip()]
        unknown = [s for s in wanted_seg if s not in SEGMENTS]
        if unknown:
            raise HTTPException(422, f"unknown segment(s) {unknown}; expected one of {SEGMENTS}")
        mask &= np.isin(st.segment_arr, wanted_seg)

    priced = np.isfinite(st.price)
    if min_price is not None or max_price is not None:
        mask &= priced
        if min_price is not None:
            mask &= np.nan_to_num(st.price, nan=-1.0) >= min_price
        if max_price is not None:
            mask &= np.nan_to_num(st.price, nan=1e18) <= max_price
        notes.append(
            f"A price filter excludes the {int((~priced).sum()):,} listings with no price "
            f"({100.0 * (~priced).mean():.0f}% of the catalogue) - they are unknown, not cheap."
        )
    if min_ram is not None:
        has_ram = np.isfinite(st.ram)
        mask &= has_ram & (np.nan_to_num(st.ram, nan=-1.0) >= min_ram)
        notes.append("Listings with no parsed RAM value are excluded by the min_ram filter.")
    if has_discrete_gpu is not None:
        mask &= (st.discrete == bool(has_discrete_gpu))
    if is_renewed is not None:
        mask &= (st.renewed == bool(is_renewed))
    if min_rating is not None:
        mask &= np.isfinite(st.rating) & (np.nan_to_num(st.rating, nan=-1.0) >= min_rating)
    if has_sentiment is not None:
        flags = np.array([a in st.sentiment_asins for a in st.asins], dtype=bool)
        mask &= (flags == bool(has_sentiment))

    candidates = np.flatnonzero(mask)
    idx, scores = _match_indices(q, candidates)
    total = int(idx.size)

    # -- ordering ------------------------------------------------------------------------
    effective_sort = sort
    if sort == "relevance" and not (q and q.strip()):
        effective_sort = "reviews_desc"
        notes.append("No query text, so 'relevance' falls back to most-reviewed first.")

    if total:
        if effective_sort == "relevance":
            order = np.lexsort((idx, -scores))          # score desc, stable by position
        elif effective_sort == "price_asc":
            key = np.where(np.isfinite(st.price[idx]), st.price[idx], np.inf)
            order = np.lexsort((idx, key))
        elif effective_sort == "price_desc":
            key = np.where(np.isfinite(st.price[idx]), st.price[idx], -np.inf)
            order = np.lexsort((idx, -key))
        elif effective_sort == "rating_desc":
            key = np.where(np.isfinite(st.rating[idx]), st.rating[idx], -1.0)
            order = np.lexsort((-st.rating_number[idx].astype("float64"), -key))
        elif effective_sort == "reviews_desc":
            order = np.lexsort((-st.rating_number[idx].astype("float64"),
                                -st.n_reviews[idx].astype("float64")))
        elif effective_sort == "ram_desc":
            key = np.where(np.isfinite(st.ram[idx]), st.ram[idx], -1.0)
            order = np.lexsort((idx, -key))
        else:  # title_asc
            order = np.argsort(np.array([st.title_lower[i] for i in idx], dtype=object),
                               kind="stable")
        idx = idx[order]
        scores = scores[order]

    page = idx[offset: offset + limit]
    items = []
    for rank, pos in enumerate(page.tolist()):
        card = _card(pos)
        card["rank"] = offset + rank + 1
        if effective_sort == "relevance":
            card["match_score"] = float(scores[offset + rank])
        items.append(card)

    n_priced_hits = int(np.isfinite(st.price[idx]).sum()) if total else 0
    return {
        "query": {
            "q": q, "brand": brand, "segment": segment,
            "min_price": min_price, "max_price": max_price, "min_ram": min_ram,
            "has_discrete_gpu": has_discrete_gpu, "is_renewed": is_renewed,
            "min_rating": min_rating, "has_sentiment": has_sentiment,
            "sort": sort, "effective_sort": effective_sort,
            "limit": limit, "offset": offset,
        },
        "total": total,
        "returned": len(items),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(items) < total,
        "coverage": {
            "n_priced_in_result": n_priced_hits,
            "price_coverage_in_result": (n_priced_hits / total) if total else 0.0,
            "catalogue_size": st.n,
        },
        "notes": notes,
        "took_ms": round(1000.0 * (time.time() - t0), 2),
        "items": items,
    }


# --------------------------------------------------------------------------------------
# /api/products/{asin}
# --------------------------------------------------------------------------------------


def _sentiment_block(asin: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    """(sentiment dict | None, no_sentiment_data, reason)."""
    try:
        sent = sentiment.get_product_sentiment(asin)
    except Exception as exc:  # artifact missing / unreadable
        return None, True, f"sentiment artifact unavailable: {type(exc).__name__}: {exc}"
    if not sent:
        return None, True, (
            "No mined review sentiment for this product yet. Only "
            f"{len(STATE.sentiment_asins):,} of {STATE.n:,} products "
            f"({100.0 * len(STATE.sentiment_asins) / STATE.n if STATE.n else 0:.1f}%) "
            "have been scored; a full pass is queued."
        )
    return sent, False, None


@app.get("/api/products/{asin}", summary="Full product detail: specs, price position, sentiment")
def product_detail(
    asin: str = PathParam(..., min_length=3, max_length=32),
) -> dict[str, Any]:
    row = STATE.row_for(asin)
    price = _scalar(row["price"])

    try:
        position = pricing.price_position(asin)
    except KeyError:
        raise HTTPException(404, f"unknown parent_asin {asin!r}")
    except Exception as exc:
        position = {"unavailable": f"{type(exc).__name__}: {exc}"}

    sent, no_sent, reason = _sentiment_block(asin)

    return {
        PRODUCT_KEY: asin,
        "title": str(row["title"]),
        "brand": str(row["brand"]),
        "store": str(row["store"]),
        "segment": str(row["segment"]),
        "is_renewed": bool(row["is_renewed"]),
        "price": price,
        "price_available": price is not None,
        "price_display": _price_display(price),
        "price_is_missing": bool(row["price_is_missing"]),
        "specs": _specs_of(row),
        "market": {
            "average_rating": _scalar(row["average_rating"]),
            "rating_number": int(row["rating_number"]),
            "n_variants": int(row["n_variants"]),
        },
        "reviews": {
            "n_reviews_retained": int(row["n_reviews"]),
            "rating_number": int(row["rating_number"]),
            "n_reviews_scored": int(sent["n_reviews_scored"]) if sent else 0,
            "has_snippets": bool(sent),
        },
        "price_position": position,
        "sentiment": sent,
        "no_sentiment_data": no_sent,
        "sentiment_note": reason,
        "links": {
            "competitors": f"/api/products/{asin}/competitors",
            "reviews": f"/api/products/{asin}/reviews",
        },
    }


# --------------------------------------------------------------------------------------
# /api/products/{asin}/competitors
# --------------------------------------------------------------------------------------


@app.get("/api/products/{asin}/competitors",
         summary="Ranked competitors, with the segment+price guard as a toggle")
def product_competitors(
    asin: str = PathParam(..., min_length=3, max_length=32),
    k: int = Query(10, ge=1, le=50),
    guard: bool = Query(True, description="segment+price guard; set false for the ablation"),
    text_weight: float = Query(matching.DEFAULT_TEXT_WEIGHT, ge=0.0, le=1.0),
    max_per_brand: int | None = Query(3, ge=1, le=50,
                                      description="diversity cap; omit/null to disable"),
    exclude_same_brand: bool = Query(False),
    include_renewed: bool = Query(True),
    min_rating_number: int = Query(0, ge=0),
) -> dict[str, Any]:
    STATE.row_for(asin)  # 404 before touching the matcher
    t0 = time.time()
    try:
        res = matching.find_competitors(
            asin, k=k, apply_guard=guard, text_weight=text_weight,
            max_per_brand=max_per_brand, exclude_same_brand=exclude_same_brand,
            include_renewed=include_renewed, min_rating_number=min_rating_number,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    took = time.time() - t0

    items = [
        {**_competitor_record(r), "rank": i + 1}
        for i, (_, r) in enumerate(res.iterrows())
    ]
    query_row = STATE.row_for(asin)
    q_price = _scalar(query_row["price"])
    n_priced = sum(1 for it in items if it["price_available"])
    off_segment = [it["segment"] for it in items if it["segment"] != str(query_row["segment"])]

    return {
        "query_product": {
            PRODUCT_KEY: asin,
            "title": str(query_row["title"]),
            "brand": str(query_row["brand"]),
            "segment": str(query_row["segment"]),
            "price": q_price,
            "price_available": q_price is not None,
            "price_display": _price_display(q_price),
            "specs": _specs_of(query_row),
            "average_rating": _scalar(query_row["average_rating"]),
            "rating_number": int(query_row["rating_number"]),
        },
        "params": {
            "k": k, "guard": guard, "text_weight": text_weight,
            "max_per_brand": max_per_brand, "exclude_same_brand": exclude_same_brand,
            "include_renewed": include_renewed, "min_rating_number": min_rating_number,
        },
        "guard": {
            "applied": guard,
            "min_segment_affinity": matching.MIN_SEGMENT_AFFINITY,
            "price_band": matching.PRICE_BAND,
            "estimated_band_multiplier": matching.ESTIMATED_BAND_MULTIPLIER,
            "segment_penalty_weight": matching.SEGMENT_PENALTY_WEIGHT,
            "price_penalty_weight": matching.PRICE_PENALTY_WEIGHT,
            "description": (
                "Guarded ranking drops incompatible segments and out-of-band prices and "
                "penalises adjacent segments. Call the same URL with guard=false for the "
                "raw hybrid ranking (the ablation that shows why the guard exists)."
            ),
            "ablation_url": f"/api/products/{asin}/competitors?k={k}&guard={str(not guard).lower()}",
        },
        "count": len(items),
        "coverage": {
            "n_priced": n_priced,
            "price_coverage": (n_priced / len(items)) if items else 0.0,
            "n_off_segment": len(off_segment),
            "off_segments": sorted(set(off_segment)),
        },
        "took_ms": round(1000.0 * took, 2),
        "competitors": items,
    }


# --------------------------------------------------------------------------------------
# /api/products/{asin}/reviews
# --------------------------------------------------------------------------------------


@app.get("/api/products/{asin}/reviews",
         summary="Representative positive/negative verbatim snippets per aspect")
def product_reviews(
    asin: str = PathParam(..., min_length=3, max_length=32),
    k: int = Query(3, ge=1, le=10, description="snippets per aspect, per polarity"),
    aspect: str | None = Query(None, description=f"restrict to one of {ASPECT_NAMES}"),
) -> dict[str, Any]:
    row = STATE.row_for(asin)
    if aspect is not None and aspect not in ASPECT_NAMES:
        raise HTTPException(422, f"unknown aspect {aspect!r}; expected one of {ASPECT_NAMES}")

    sent, no_sent, reason = _sentiment_block(asin)
    base: dict[str, Any] = {
        PRODUCT_KEY: asin,
        "title": str(row["title"]),
        "brand": str(row["brand"]),
        "n_reviews_retained": int(row["n_reviews"]),
        "rating_number": int(row["rating_number"]),
        "no_sentiment_data": no_sent,
        "sentiment_note": reason,
        "aspect_filter": aspect,
        "k_per_aspect": k,
    }
    if no_sent:
        return {**base, "n_reviews_scored": 0, "overall": None, "aspects": [],
                "praises": [], "complaints": []}

    wanted = [aspect] if aspect else ASPECT_NAMES
    aspects_out: list[dict[str, Any]] = []
    for asp in wanted:
        try:
            praises = sentiment.top_praises(asin, k=k, aspect=asp)
            complaints = sentiment.top_complaints(asin, k=k, aspect=asp)
        except Exception as exc:
            aspects_out.append({"aspect": asp, "error": f"{type(exc).__name__}: {exc}"})
            continue
        stats = (sent or {}).get("aspects", {}).get(asp)
        if not praises and not complaints and not stats:
            continue
        aspects_out.append({
            "aspect": asp,
            "mentions": (stats or {}).get("mentions", 0),
            "polarity": (stats or {}).get("polarity"),
            "pos_share": (stats or {}).get("pos_share"),
            "label": (stats or {}).get("label"),
            "n_praises": len(praises),
            "n_complaints": len(complaints),
            "praises": praises,
            "complaints": complaints,
        })

    aspects_out.sort(key=lambda a: -(a.get("mentions") or 0))
    return {
        **base,
        "n_reviews_scored": int(sent["n_reviews_scored"]),
        "overall": {
            "polarity": sent["overall_polarity"],
            "pos_share": sent["overall_pos_share"],
            "mean_rating": sent["mean_rating"],
            "label": sentiment.polarity_label(
                sent["overall_polarity"] if sent["overall_polarity"] is not None else float("nan")),
        },
        "aspects": aspects_out,
        "praises": sentiment.top_praises(asin, k=k, aspect=aspect),
        "complaints": sentiment.top_complaints(asin, k=k, aspect=aspect),
        "note": (
            "Snippets are verbatim clauses from customer reviews, ranked by opinion "
            "strength x helpful votes and de-duplicated."
        ),
    }


# --------------------------------------------------------------------------------------
# /api/segments and /api/brands
# --------------------------------------------------------------------------------------


@app.get("/api/segments", summary="Per-segment price table with n and coverage on every stat")
def segments() -> dict[str, Any]:
    rows = _segment_table()
    n = STATE.n
    total_priced = sum(int(r["n_priced"]) for r in rows)
    return {
        "n_products": n,
        "n_segments": len(rows),
        "min_reliable_n": int(pricing.MIN_RELIABLE_N),
        "overall_price_coverage": (total_priced / n) if n else 0.0,
        "expected_order": list(pricing.EXPECTED_SEGMENT_ORDER),
        "columns": {
            "n_products": "catalogue rows in the segment (the denominator)",
            "n_priced": "rows with a real price (same as pricing.py's `n`)",
            "coverage": "n_priced / n_products",
            "reliable": f"n_priced >= {pricing.MIN_RELIABLE_N}",
            "median": "median price of the priced rows only",
        },
        "notes": [
            "Every statistic describes the priced subset; unpriced rows are unknown, "
            "not free.",
            "sentiment_coverage is the share of the segment with mined review sentiment.",
        ],
        "rows": rows,
    }


@app.get("/api/brands", summary="Per-brand price table, unreliable (<5 priced) flag preserved")
def brands(
    min_products: int = Query(1, ge=1, le=10_000),
    q: str | None = Query(None, max_length=100, description="substring filter on brand name"),
    sort: Literal["n_products", "median", "brand", "coverage", "mean_rating"] =
        Query("n_products"),
    desc: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    rows = [r for r in _brand_table() if int(r["n_products"]) >= min_products]
    if q:
        ql = q.strip().lower()
        rows = [r for r in rows if ql in str(r["brand"]).lower()]

    def _key(r: dict[str, Any]) -> tuple:
        """Sort key that keeps missing values last in both directions."""
        v = r.get(sort)
        missing = v is None or (isinstance(v, float) and not math.isfinite(v))
        # sorted(reverse=True) puts the largest key first, so the missing bucket needs
        # the smallest rank when descending and the largest when ascending
        rank = (0 if desc else 1) if missing else (1 if desc else 0)
        if missing:
            return (rank, 0.0, str(r["brand"]).lower())
        if isinstance(v, str):
            return (rank, 0.0, v.lower())
        return (rank, float(v), str(r["brand"]).lower())

    rows = sorted(rows, key=_key, reverse=desc)
    total = len(rows)
    page = rows[offset: offset + limit]
    n_unreliable = sum(1 for r in rows if r["unreliable"])
    return {
        "n_brands": total,
        "returned": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total,
        "min_reliable_n": int(pricing.MIN_RELIABLE_N),
        "n_unreliable": n_unreliable,
        "params": {"min_products": min_products, "q": q, "sort": sort, "desc": desc},
        "notes": [
            f"`unreliable` = fewer than {pricing.MIN_RELIABLE_N} priced products; that "
            f"brand's median price must not be quoted as a benchmark "
            f"({n_unreliable} of {total} brands shown are in that state).",
            "Brands are kept in the table even at zero price coverage so low-coverage "
            "brands stay visible instead of being silently deleted.",
        ],
        "rows": page,
    }


# --------------------------------------------------------------------------------------
# /api/market/overview
# --------------------------------------------------------------------------------------


@app.get("/api/market/overview", summary="Headline market stats for the dashboard landing view")
def market_overview() -> dict[str, Any]:
    return _market_overview()


# ======================================================================================
# 6. Chat (lazy LLM)
# ======================================================================================


def _gpu_memory_used_mib() -> tuple[int | None, int | None, str | None]:
    """(used, total, error) from nvidia-smi; (None, None, reason) if it cannot be read."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, "nvidia-smi not found on PATH"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        used, total = (int(x.strip()) for x in out[0].split(",")[:2])
        return used, total, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


class LLMManager:
    """Lazy, thread-safe owner of the rag agent + 7B LLM.

    States: ``unloaded`` -> ``loading`` -> ``ready``; ``unavailable`` on a failed load.
    Only one generation runs at a time (one GPU, one resident model), so a second
    concurrent chat gets an honest 429 instead of an OOM.
    """

    def __init__(self) -> None:
        self.state: str = "unloaded"
        self.error: str | None = None
        self.error_type: str | None = None
        self.load_seconds: float | None = None
        self.loaded_at: float | None = None
        self.n_answers: int = 0
        self.model_name: str = config.LLM_MODEL
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._agent: Any = None

    # -- introspection -------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        used, total, gpu_err = _gpu_memory_used_mib()
        return {
            "state": self.state,                       # unloaded | loading | ready | unavailable
            "ready": self.state == "ready",
            "loading": self.state == "loading",
            "available": self.state != "unavailable",
            "accepting_input": self.state in ("unloaded", "ready"),
            "model": self.model_name,
            "load_seconds": self.load_seconds,
            "answers_served": self.n_answers,
            "busy": self._gen_lock.locked(),
            "error": self.error,
            "error_type": self.error_type,
            "gpu": {
                "memory_used_mib": used,
                "memory_total_mib": total,
                "busy_threshold_mib": GPU_BUSY_MIB,
                # `free` is the raw reading; once our own model is resident the used
                # figure is mostly *us*, which is why `can_load` is the flag to act on
                "free": used is not None and used <= GPU_BUSY_MIB,
                "held_by_this_process": self.state == "ready",
                "can_load": (used is not None and used <= GPU_BUSY_MIB) or self.state == "ready",
                "error": gpu_err,
            },
            "hint": {
                "unloaded": "The 7B model loads on the first chat request (~30-60 s).",
                "loading": "Model weights are loading; the input should stay disabled.",
                "ready": "Model resident on the GPU; answers take ~2-17 s.",
                "unavailable": "The model could not be loaded; chat is disabled.",
            }[self.state],
        }

    # -- loading -------------------------------------------------------------------------
    def _wait_for_gpu(self) -> tuple[bool, dict[str, Any]]:
        """Refuse to load while another process holds the GPU (an adversarial verifier may)."""
        used = total = None
        for attempt in range(GPU_WAIT_ATTEMPTS):
            used, total, err = _gpu_memory_used_mib()
            if used is None:
                return False, {"reason": "gpu_unreadable", "detail": err}
            if used <= GPU_BUSY_MIB:
                return True, {"memory_used_mib": used, "memory_total_mib": total,
                              "attempts": attempt + 1}
            LOG.warning("GPU busy (%s MiB used); waiting %.0fs", used, GPU_WAIT_SECONDS)
            time.sleep(GPU_WAIT_SECONDS)
        return False, {"reason": "gpu_busy", "memory_used_mib": used,
                       "memory_total_mib": total, "threshold_mib": GPU_BUSY_MIB}

    def ensure_ready(self) -> tuple[bool, dict[str, Any] | None]:
        """Load the agent + weights if needed. Returns ``(ok, structured_error)``."""
        if self.state == "ready":
            return True, None
        with self._load_lock:
            if self.state == "ready":
                return True, None
            ok, gpu_info = self._wait_for_gpu()
            if not ok:
                return False, {
                    "code": gpu_info.get("reason", "gpu_busy"),
                    "message": (
                        f"GPU is not free ({gpu_info.get('memory_used_mib')} MiB in use, "
                        f"threshold {GPU_BUSY_MIB} MiB); refusing to load the 7B model "
                        f"rather than risk an out-of-memory crash."
                        if gpu_info.get("reason") != "gpu_unreadable"
                        else f"Cannot read GPU state: {gpu_info.get('detail')}"
                    ),
                    "gpu": gpu_info,
                    "retryable": True,
                }
            self.state = "loading"
            self.error = self.error_type = None
            t0 = time.time()
            try:
                import rag  # noqa: PLC0415 - deliberately lazy (pulls torch/transformers)

                STATE.module_status["rag"]["imported"] = True
                agent = rag.get_agent()
                agent.llm.load()
                self._agent = agent
                self.model_name = agent.llm.model_name
                self.load_seconds = round(time.time() - t0, 2)
                self.loaded_at = time.time()
                self.state = "ready"
                LOG.info("LLM ready in %.1fs", self.load_seconds)
                return True, None
            except Exception as exc:
                self.state = "unavailable"
                self.error = str(exc)
                self.error_type = type(exc).__name__
                STATE.module_status["rag"]["error"] = f"{self.error_type}: {self.error}"
                LOG.error("LLM load failed\n%s", traceback.format_exc())
                return False, {
                    "code": "model_load_failed",
                    "message": f"{self.error_type}: {self.error}",
                    "retryable": False,
                }

    # -- generation ----------------------------------------------------------------------
    def answer(self, question: str, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        import rag  # already imported by ensure_ready

        res = self._agent.answer(question, max_new_tokens=max_new_tokens,
                                 temperature=temperature)
        self.n_answers += 1
        assert isinstance(res, rag.RagAnswer)
        return res.to_dict()

    @property
    def gen_lock(self) -> threading.Lock:
        return self._gen_lock


LLM = LLMManager()


class ChatRequest(BaseModel):
    """POST /api/chat body."""

    question: str = Field(..., min_length=3, max_length=1000)
    max_new_tokens: int = Field(420, ge=32, le=1200)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


@app.get("/api/chat/status", summary="Is the LLM loaded / loading / unavailable?")
def chat_status() -> dict[str, Any]:
    return LLM.status()


@app.post("/api/chat", summary="Grounded RAG answer with evidence and a hallucination audit")
def chat(body: ChatRequest = Body(...)) -> Any:
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "question must not be blank")

    if not LLM.gen_lock.acquire(timeout=2.0):
        return SafeJSONResponse(
            status_code=429,
            content={"error": {"status": 429, "type": "busy", "code": "generation_in_progress",
                               "message": "Another answer is being generated on the single "
                                          "GPU; retry in a few seconds."},
                     "llm": LLM.status()},
            headers={"Retry-After": "5"},
        )
    try:
        t0 = time.time()
        ok, err = LLM.ensure_ready()
        if not ok:
            status = 503
            return SafeJSONResponse(
                status_code=status,
                content={
                    "error": {"status": status, "type": "llm_unavailable",
                              "code": err.get("code"), "message": err.get("message"),
                              "retryable": err.get("retryable", False)},
                    "detail": err,
                    "llm": LLM.status(),
                    "question": question,
                },
                headers={"Retry-After": "15"} if err.get("retryable") else None,
            )
        try:
            payload = LLM.answer(question, body.max_new_tokens, body.temperature)
        except Exception as exc:
            LOG.error("generation failed\n%s", traceback.format_exc())
            return SafeJSONResponse(
                status_code=500,
                content={
                    "error": {"status": 500, "type": "generation_failed",
                              "code": "generation_failed",
                              "message": f"{type(exc).__name__}: {exc}"},
                    "llm": LLM.status(),
                    "question": question,
                },
            )
        latency = time.time() - t0
    finally:
        LLM.gen_lock.release()

    evidence = payload.get("evidence", [])
    return {
        "question": question,
        "question_type": payload.get("question_type"),
        "answer": payload.get("answer"),
        "evidence": evidence,
        "n_evidence": len(evidence),
        "audit": {
            "grounded": payload.get("grounded"),
            "citation_rate": payload.get("citation_rate"),
            "citations": payload.get("citations", []),
            "unsupported_markers": payload.get("unsupported_markers", []),
            "unverified_numbers": payload.get("unverified_numbers", []),
            "misattributed_reviews": payload.get("misattributed_reviews", []),
            "uncited_sentences": payload.get("uncited_sentences", []),
            "truncated": payload.get("truncated", False),
            "explanation": (
                "unsupported_markers = citation tokens with no matching evidence block; "
                "unverified_numbers = figures in the answer that appear in no evidence "
                "block; misattributed_reviews = a review quoted against the wrong product."
            ),
        },
        "latency_s": round(latency, 3),
        "retrieval": payload.get("retrieval", {}),
        "timings": payload.get("timings", {}),
        "query_spec": payload.get("query_spec", {}),
        "prompt_chars": payload.get("prompt_chars"),
        "llm": {"model": LLM.model_name, "load_seconds": LLM.load_seconds,
                "answers_served": LLM.n_answers},
    }


# ======================================================================================
# 7. Static UI (mounted last so it can never shadow /api/*)
# ======================================================================================


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
               include_in_schema=False)
def api_not_found(rest: str) -> Any:
    """Structured 404 for a mistyped API path.

    Registered after every real route, so it only catches misses. Without it an
    unknown /api/... path falls through to the StaticFiles mount and comes back as
    the generic ``{"detail": "Not Found"}``, which tells the UI nothing.
    """
    raise HTTPException(
        404,
        f"no such API route: /api/{rest}. See /docs for the available endpoints.",
    )


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ======================================================================================
# 8. Dev entry point
# ======================================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the competitor-intelligence API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "src.web.api:app" if args.reload else app,
        host=args.host, port=args.port, reload=args.reload, log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
