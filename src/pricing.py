"""Price positioning and benchmarking for the laptop competitor-intelligence system.

The central constraint of this module is *coverage*: only ~30% of the products in
``products.parquet`` carry a price (the Amazon 2023 dump leaves ``price`` null for
the rest).  Every aggregate produced here therefore travels with

    n          - number of PRICED products the statistic was computed from,
    n_total    - number of products in the group (priced or not),
    coverage   - n / n_total,
    reliable   - False when n < MIN_RELIABLE_N (default 5),

so no caller can accidentally present a 30%-sample mean as a market-wide fact.
Nothing in this module ever imputes a price: unpriced products stay unpriced.

Public API
----------
load_products()                     cached read of products.parquet
price_summary(prices, n_total)      the universal {n, coverage, median, ...} block
segment_price_table()               per-segment aggregates
brand_price_table()                 per-brand aggregates
price_position(parent_asin)         percentile / delta / cheaper-comparable-premium
price_vs_value()                    per-product value metrics (price per RAM GB, ...)
value_outliers()                    best/worst value products vs a spec price model
discrete_gpu_premium()              does a dGPU command a premium, per segment
price_rating_relationship()         price vs average_rating correlations
compare_products([asins])           tidy side-by-side comparison frame
validate()                          writes eval/pricing_eval.json

Run ``python src/pricing.py`` to print the segment and brand tables plus a couple
of worked price positions.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import EVAL, PRODUCTS_PARQUET, SEGMENTS

# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #

#: Below this many priced products an aggregate is reported but flagged unreliable.
MIN_RELIABLE_N = 5

#: A laptop cheaper than this is almost certainly a listing error, an accessory
#: that survived filtering, or a per-unit price on a bulk/parts listing.
IMPLAUSIBLE_LOW_USD = 50.0

#: Above this we assume a data error (workstation halo listings top out ~$8k).
IMPLAUSIBLE_HIGH_USD = 10_000.0

#: |price / segment median - 1| below this band counts as "comparable".
COMPARABLE_BAND = 0.15

#: Ordering assertions the segment medians should satisfy (validation only).
EXPECTED_SEGMENT_ORDER = [
    ("chromebook", "budget"),
    ("budget", "mainstream"),
    ("mainstream", "gaming"),
    ("budget", "gaming"),
    ("budget", "ultrabook"),
    ("mainstream", "ultrabook"),
]

#: Numeric spec columns used by the value model / spec ratios.
_SPEC_NUM = ["ram_gb", "storage_gb", "screen_in", "cpu_ghz", "cpu_tier", "weight_lb"]
_SPEC_BOOL = ["is_discrete_gpu", "is_renewed"]
_SPEC_CAT = ["cpu_brand", "gpu_brand", "os_family", "storage_type", "segment"]

_PRODUCTS_CACHE: pd.DataFrame | None = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_products(path: Path | str = PRODUCTS_PARQUET, refresh: bool = False) -> pd.DataFrame:
    """Load ``products.parquet`` once and memoise it.

    Parameters
    ----------
    path : path to the parquet file (defaults to the contract path in config).
    refresh : ignore the in-process cache and re-read from disk.

    Returns
    -------
    DataFrame indexed positionally, one row per ``parent_asin``.
    """
    global _PRODUCTS_CACHE
    if _PRODUCTS_CACHE is None or refresh:
        df = pd.read_parquet(path)
        if df["parent_asin"].duplicated().any():
            raise ValueError("products.parquet contains duplicate parent_asin values")
        _PRODUCTS_CACHE = df
    return _PRODUCTS_CACHE


def _df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or the cached catalogue when None."""
    return load_products() if df is None else df


# --------------------------------------------------------------------------- #
# The universal aggregate block
# --------------------------------------------------------------------------- #


def price_summary(prices: Iterable[float], n_total: int | None = None,
                  label: str | None = None) -> dict:
    """Summarise a price vector while keeping the denominator explicit.

    Parameters
    ----------
    prices : iterable of prices; NaNs are dropped and counted as "unpriced".
    n_total : size of the population the prices were drawn from.  Defaults to the
        length of ``prices`` (i.e. coverage 1.0), which is only correct if the
        caller already knows every member of the group.
    label : optional name carried through into the result (segment/brand name).

    Returns
    -------
    dict with n, n_total, coverage, reliable, median, mean, std, p25, p75, iqr,
    p10, p90, min, max.  Dispersion stats are None when n < 2; central stats are
    None when n == 0.  ``reliable`` is False when n < MIN_RELIABLE_N.
    """
    s = pd.Series(list(prices), dtype="float64")
    if n_total is None:
        n_total = int(len(s))
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(s.size)

    out: dict = {
        "n": n,
        "n_total": int(n_total),
        "coverage": (n / n_total) if n_total else 0.0,
        "reliable": n >= MIN_RELIABLE_N,
        "median": None, "mean": None, "std": None,
        "p25": None, "p75": None, "iqr": None,
        "p10": None, "p90": None, "min": None, "max": None,
    }
    if label is not None:
        out["label"] = label
    if n == 0:
        return out

    q10, q25, q75, q90 = (float(s.quantile(q)) for q in (0.10, 0.25, 0.75, 0.90))
    out.update(
        median=float(s.median()),
        mean=float(s.mean()),
        std=float(s.std(ddof=1)) if n > 1 else None,
        p25=q25, p75=q75, iqr=q75 - q25,
        p10=q10, p90=q90,
        min=float(s.min()), max=float(s.max()),
    )
    return out


def _group_price_table(df: pd.DataFrame, by: str, min_products: int = 1) -> pd.DataFrame:
    """Build a per-group price table (one row per value of ``by``).

    The row count denominator is the *whole* group, not just its priced members,
    so ``coverage`` is meaningful.  Groups are returned sorted by median price
    with unreliable groups (n < MIN_RELIABLE_N) still present but flagged.
    """
    rows = []
    for key, g in df.groupby(by, observed=True):
        if len(g) < min_products:
            continue
        rec = price_summary(g["price"], n_total=len(g), label=str(key))
        rec[by] = str(key)
        rec["n_products"] = int(len(g))
        rec["n_renewed"] = int(g["is_renewed"].sum())
        rec["mean_rating"] = float(g["average_rating"].mean()) if g["average_rating"].notna().any() else None
        rows.append(rec)

    cols = [by, "n_products", "n", "coverage", "reliable", "median", "mean", "std",
            "p25", "p75", "iqr", "p10", "p90", "min", "max", "n_renewed", "mean_rating"]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out[cols].sort_values("median", ascending=True, na_position="last")
    return out.reset_index(drop=True)


def segment_price_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-segment price aggregates with coverage and reliability flags."""
    return _group_price_table(_df(df), "segment")


def brand_price_table(df: pd.DataFrame | None = None, min_products: int = 1,
                      top: int | None = None) -> pd.DataFrame:
    """Per-brand price aggregates.

    Parameters
    ----------
    min_products : only include brands with at least this many catalogue rows
        (priced or not).  Filtering on catalogue size rather than priced size
        keeps low-coverage brands visible instead of silently deleting them.
    top : if given, return only the ``top`` brands by catalogue size (still
        sorted by median price in the returned frame).
    """
    tbl = _group_price_table(_df(df), "brand", min_products=min_products)
    if top is not None and not tbl.empty:
        keep = tbl.nlargest(top, "n_products")["brand"]
        tbl = tbl[tbl["brand"].isin(set(keep))].reset_index(drop=True)
    return tbl


# --------------------------------------------------------------------------- #
# Positioning
# --------------------------------------------------------------------------- #


def _percentile_of(value: float, population: pd.Series) -> float | None:
    """Mid-rank percentile (0-100) of ``value`` within ``population`` (NaNs dropped)."""
    pop = pd.Series(population, dtype="float64").dropna()
    if pop.empty or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    below = float((pop < value).sum())
    equal = float((pop == value).sum())
    return 100.0 * (below + 0.5 * equal) / float(pop.size)


def _label_for(ratio: float | None) -> str:
    """Map price/median ratio to cheaper / comparable / premium.

    A missing ratio (no price, or no usable peer median) yields ``"unknown"`` -
    never a default of "comparable".
    """
    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)) or pd.isna(ratio):
        return "unknown"
    if ratio < 1.0 - COMPARABLE_BAND:
        return "cheaper"
    if ratio > 1.0 + COMPARABLE_BAND:
        return "premium"
    return "comparable"


def price_position(parent_asin: str, df: pd.DataFrame | None = None) -> dict:
    """Locate one product's price within its segment and within its brand.

    Returns a nested dict::

        {
          "parent_asin", "title", "brand", "segment", "price", "price_available",
          "segment": {percentile, median, delta_usd, ratio, label, n, coverage,
                      reliable, ...},
          "brand":   {... same shape ...},
          "notes":   [human-readable caveats]
        }

    When the product itself has no price the peer-group blocks are still filled
    in (so the app can say "we don't know this one's price, but its segment
    median is $X from n=Y priced peers") and ``label`` is ``"unknown"``.
    """
    d = _df(df)
    hit = d[d["parent_asin"] == parent_asin]
    if hit.empty:
        raise KeyError(f"parent_asin {parent_asin!r} not found in products.parquet")
    row = hit.iloc[0]

    price = float(row["price"]) if pd.notna(row["price"]) else None
    notes: list[str] = []

    def _block(peers: pd.DataFrame, kind: str, key: str) -> dict:
        # exclude the product itself from its own peer group
        peers = peers[peers["parent_asin"] != parent_asin]
        blk = price_summary(peers["price"], n_total=len(peers), label=key)
        blk["group_type"] = kind
        blk["percentile"] = _percentile_of(price, peers["price"]) if price is not None else None
        med = blk["median"]
        if price is not None and med:
            blk["delta_usd"] = price - med
            blk["ratio"] = price / med
            blk["delta_pct"] = 100.0 * (price / med - 1.0)
        else:
            blk["delta_usd"] = blk["ratio"] = blk["delta_pct"] = None
        blk["label"] = _label_for(blk["ratio"])
        if not blk["reliable"]:
            notes.append(
                f"{kind} '{key}' has only n={blk['n']} priced peers "
                f"(<{MIN_RELIABLE_N}); its median is not a reliable benchmark."
            )
        return blk

    seg_block = _block(d[d["segment"] == row["segment"]], "segment", str(row["segment"]))
    brand_block = _block(d[d["brand"] == row["brand"]], "brand", str(row["brand"]))

    if price is None:
        notes.append("This product has no price in the source data; no position can be computed.")
    else:
        if price < IMPLAUSIBLE_LOW_USD:
            notes.append(f"Price ${price:,.2f} is below ${IMPLAUSIBLE_LOW_USD:,.0f} - suspected data error.")
        if price > IMPLAUSIBLE_HIGH_USD:
            notes.append(f"Price ${price:,.2f} is above ${IMPLAUSIBLE_HIGH_USD:,.0f} - suspected data error.")
    if bool(row["is_renewed"]):
        notes.append("Product is renewed/refurbished; it is benchmarked against a mixed new+renewed peer group.")
    if seg_block["coverage"] < 0.5:
        notes.append(
            f"Only {seg_block['coverage']:.0%} of the '{row['segment']}' segment is priced "
            f"({seg_block['n']} of {seg_block['n_total']}); the percentile describes priced peers only."
        )

    return {
        "parent_asin": str(row["parent_asin"]),
        "title": str(row["title"]),
        "brand": str(row["brand"]),
        "segment": str(row["segment"]),
        "price": price,
        "price_available": price is not None,
        "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
        "rating_number": int(row["rating_number"]),
        "vs_segment": seg_block,
        "vs_brand": brand_block,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Price vs value
# --------------------------------------------------------------------------- #


def price_vs_value(df: pd.DataFrame | None = None, min_ratings: int = 0) -> pd.DataFrame:
    """Per-product value metrics for the priced subset of the catalogue.

    Only rows with a real price are returned - the frame's length *is* the
    denominator, and the caller is expected to show it.

    Columns added on top of the identity/spec columns:
      price_per_ram_gb, price_per_storage_100gb, price_per_inch,
      segment_price_percentile, price_vs_segment_median (ratio),
      rating_per_100usd (average_rating / price * 100),
      expected_price / price_residual / price_residual_pct  (spec model, see
      :func:`fit_price_model`), value_score (negative residual pct = good value).
    """
    d = _df(df)
    p = d[d["price"].notna()].copy()
    if min_ratings:
        p = p[p["rating_number"] >= min_ratings]
    if p.empty:
        return p

    p["price_per_ram_gb"] = np.where(p["ram_gb"] > 0, p["price"] / p["ram_gb"], np.nan)
    p["price_per_storage_100gb"] = np.where(
        p["storage_gb"] > 0, p["price"] / (p["storage_gb"] / 100.0), np.nan)
    p["price_per_inch"] = np.where(p["screen_in"] > 0, p["price"] / p["screen_in"], np.nan)
    p["rating_per_100usd"] = np.where(
        p["price"] > 0, p["average_rating"] / p["price"] * 100.0, np.nan)

    seg_med = p.groupby("segment", observed=True)["price"].transform("median")
    p["segment_median_price"] = seg_med
    p["price_vs_segment_median"] = p["price"] / seg_med
    p["segment_price_percentile"] = (
        p.groupby("segment", observed=True)["price"].rank(pct=True, method="average") * 100.0
    )
    p["price_label"] = [_label_for(r) for r in p["price_vs_segment_median"]]

    expected, meta = fit_price_model(p)
    p["expected_price"] = expected
    p["price_residual"] = p["price"] - p["expected_price"]
    p["price_residual_pct"] = 100.0 * (p["price"] / p["expected_price"] - 1.0)
    p["value_score"] = -p["price_residual_pct"]
    # start from a clean attrs dict: parquet-inherited metadata must not leak
    # into the frames the app consumes.
    p.attrs.clear()
    p.attrs["price_model"] = meta
    p.attrs["n_priced"] = int(len(p))
    p.attrs["n_catalogue"] = int(len(d))
    p.attrs["coverage"] = float(len(p) / len(d))
    return p.reset_index(drop=True)


def fit_price_model(priced: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Fit a ridge regression of log(price) on specs; return fitted prices + metadata.

    The model is only a *reference surface* for spotting value outliers - a
    product priced far below what its specs usually cost is a value candidate.
    It is never used to impute a missing price.

    Missing numeric specs are median-imputed with an explicit ``*_missing``
    indicator column so "unknown RAM" cannot masquerade as "average RAM".

    Returns
    -------
    (fitted_prices, metadata) where metadata holds n, feature count, in-sample R^2
    and 5-fold cross-validated R^2 on log price.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score

    X_parts: list[pd.DataFrame] = []
    num = priced[_SPEC_NUM].astype("float64")
    for c in _SPEC_NUM:
        col = num[c]
        X_parts.append(pd.DataFrame({
            c: col.fillna(col.median()),
            f"{c}_missing": col.isna().astype("float64"),
        }, index=priced.index))
    X_parts.append(priced[_SPEC_BOOL].astype("float64"))
    for c in _SPEC_CAT:
        X_parts.append(pd.get_dummies(priced[c].astype("str"), prefix=c, dtype="float64"))
    X = pd.concat(X_parts, axis=1)
    # log-scale the two heavy-tailed capacity specs
    for c in ("ram_gb", "storage_gb"):
        X[f"log_{c}"] = np.log1p(X[c])
    Xv = X.to_numpy(dtype="float64")
    y = np.log(priced["price"].to_numpy(dtype="float64"))

    model = Ridge(alpha=1.0)
    model.fit(Xv, y)
    fitted = np.exp(model.predict(Xv))

    meta = {
        "n": int(len(priced)),
        "n_features": int(Xv.shape[1]),
        "target": "log(price)",
        "r2_in_sample": float(model.score(Xv, y)),
    }
    if len(priced) >= 50:
        cv = cross_val_score(Ridge(alpha=1.0), Xv, y,
                             cv=KFold(5, shuffle=True, random_state=seed), scoring="r2")
        meta["r2_cv_mean"] = float(cv.mean())
        meta["r2_cv_std"] = float(cv.std())
    return fitted, meta


def value_outliers(df: pd.DataFrame | None = None, top: int = 10,
                   min_ratings: int = 20) -> dict[str, pd.DataFrame]:
    """Best- and worst-value priced products vs the spec price model.

    ``min_ratings`` guards against one-review no-name listings dominating the
    list.  Returns ``{"best_value": frame, "overpriced": frame}`` sorted by
    ``value_score``.
    """
    p = price_vs_value(df)
    if p.empty:
        empty = p
        return {"best_value": empty, "overpriced": empty}
    q = p[p["rating_number"] >= min_ratings]
    if q.empty:
        q = p
    cols = ["parent_asin", "brand", "segment", "title", "price", "expected_price",
            "price_residual_pct", "value_score", "average_rating", "rating_number",
            "ram_gb", "storage_gb", "is_discrete_gpu"]
    best = q.nlargest(top, "value_score")[cols].reset_index(drop=True)
    worst = q.nsmallest(top, "value_score")[cols].reset_index(drop=True)
    for frame in (best, worst):
        frame.attrs.clear()
        frame.attrs["n_pool"] = int(len(q))
        frame.attrs["price_model"] = p.attrs.get("price_model")
    return {"best_value": best, "overpriced": worst}


def discrete_gpu_premium(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Median price of discrete-GPU vs integrated-GPU laptops, overall and per segment.

    Each row carries both group sizes, the segment coverage and a Mann-Whitney U
    p-value; rows where either group has fewer than MIN_RELIABLE_N priced
    products are flagged ``reliable=False``.
    """
    from scipy.stats import mannwhitneyu

    d = _df(df)
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", d)]
    groups += [(str(k), g) for k, g in d.groupby("segment", observed=True)]

    for name, g in groups:
        dis = g.loc[g["is_discrete_gpu"] & g["price"].notna(), "price"]
        integ = g.loc[~g["is_discrete_gpu"] & g["price"].notna(), "price"]
        rec = {
            "segment": name,
            "n_products": int(len(g)),
            "n_priced": int(g["price"].notna().sum()),
            "coverage": float(g["price"].notna().mean()) if len(g) else 0.0,
            "n_discrete": int(dis.size),
            "n_integrated": int(integ.size),
            "median_discrete": float(dis.median()) if dis.size else None,
            "median_integrated": float(integ.median()) if integ.size else None,
        }
        if dis.size and integ.size:
            rec["premium_usd"] = rec["median_discrete"] - rec["median_integrated"]
            rec["premium_pct"] = 100.0 * (rec["median_discrete"] / rec["median_integrated"] - 1.0)
        else:
            rec["premium_usd"] = rec["premium_pct"] = None
        rec["reliable"] = dis.size >= MIN_RELIABLE_N and integ.size >= MIN_RELIABLE_N
        if rec["reliable"]:
            rec["mannwhitney_p"] = float(
                mannwhitneyu(dis, integ, alternative="two-sided").pvalue)
        else:
            rec["mannwhitney_p"] = None
        rows.append(rec)
    return pd.DataFrame(rows)


def price_rating_relationship(df: pd.DataFrame | None = None,
                              min_ratings: int = 5) -> pd.DataFrame:
    """Correlation between price and average_rating, overall and per segment.

    Only products with a price AND at least ``min_ratings`` ratings enter the
    correlation (a 5.0 average from 1 rating is noise).  Every row reports the
    pair count actually used and the coverage of that pair count against the
    full group size.
    """
    from scipy.stats import pearsonr, spearmanr

    d = _df(df)
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", d)]
    groups += [(str(k), g) for k, g in d.groupby("segment", observed=True)]

    for name, g in groups:
        ok = g[g["price"].notna() & g["average_rating"].notna()
               & (g["rating_number"] >= min_ratings)]
        rec = {
            "segment": name,
            "n_products": int(len(g)),
            "n_pairs": int(len(ok)),
            "coverage": float(len(ok) / len(g)) if len(g) else 0.0,
            "reliable": len(ok) >= max(MIN_RELIABLE_N, 30),
            "mean_rating_priced": float(ok["average_rating"].mean()) if len(ok) else None,
        }
        if len(ok) >= 3:
            pr = pearsonr(ok["price"], ok["average_rating"])
            sp = spearmanr(ok["price"], ok["average_rating"])
            rec.update(pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
                       spearman_r=float(sp.statistic), spearman_p=float(sp.pvalue))
            # rating by price quartile, an easier read for the app
            qs = pd.qcut(ok["price"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            for lab, sub in ok.groupby(qs, observed=True):
                rec[f"rating_{lab}"] = float(sub["average_rating"].mean())
        else:
            rec.update(pearson_r=None, pearson_p=None, spearman_r=None, spearman_p=None)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

_COMPARE_COLS = [
    "parent_asin", "brand", "segment", "title", "price", "price_available",
    "segment_price_percentile", "price_vs_segment_median_pct", "price_label",
    "cpu_brand", "cpu_family", "cpu_ghz", "ram_gb", "ram_type",
    "storage_gb", "storage_type", "screen_in", "screen_res",
    "gpu_brand", "gpu_model", "is_discrete_gpu", "os_family", "weight_lb",
    "is_renewed", "average_rating", "rating_number", "n_reviews",
    "price_per_ram_gb", "price_per_storage_100gb",
]


def compare_products(asins: Sequence[str], df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tidy side-by-side frame of price, key specs and rating for the given ASINs.

    Rows come back in the order requested.  Unknown ASINs raise ``KeyError``.
    Products without a price get ``price_available=False`` and NaN price columns
    rather than a fabricated number; the price-position columns are computed
    against each product's own segment.
    """
    d = _df(df)
    asins = list(asins)
    missing = [a for a in asins if a not in set(d["parent_asin"])]
    if missing:
        raise KeyError(f"parent_asin(s) not found: {missing}")

    sub = d.set_index("parent_asin").loc[asins].reset_index()
    seg_med = d.groupby("segment", observed=True)["price"].median()

    sub["price_available"] = sub["price"].notna()
    sub["screen_res"] = [
        f"{int(w)}x{int(h)}" if pd.notna(w) and pd.notna(h) else None
        for w, h in zip(sub["screen_w"], sub["screen_h"])
    ]
    med = sub["segment"].map(seg_med)
    sub["price_vs_segment_median_pct"] = 100.0 * (sub["price"] / med - 1.0)
    sub["price_label"] = [_label_for(r) for r in (sub["price"] / med)]
    sub["segment_price_percentile"] = [
        _percentile_of(pr, d.loc[d["segment"] == sg, "price"]) if pd.notna(pr) else None
        for pr, sg in zip(sub["price"], sub["segment"])
    ]
    sub["price_per_ram_gb"] = np.where(sub["ram_gb"] > 0, sub["price"] / sub["ram_gb"], np.nan)
    sub["price_per_storage_100gb"] = np.where(
        sub["storage_gb"] > 0, sub["price"] / (sub["storage_gb"] / 100.0), np.nan)

    out = sub[_COMPARE_COLS].copy()
    n_priced = int(out["price_available"].sum())
    out.attrs.clear()
    out.attrs["n_requested"] = len(asins)
    out.attrs["n_priced"] = n_priced
    out.attrs["price_coverage"] = n_priced / len(asins) if asins else 0.0
    if n_priced < len(asins):
        out.attrs["warning"] = (
            f"{len(asins) - n_priced} of {len(asins)} compared products have no price; "
            "price columns are blank for those rows and must not be treated as $0."
        )
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _jsonable(obj):
    """Recursively convert numpy/pandas scalars and NaNs into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.str_,)):
        return str(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if obj is pd.NA or (hasattr(pd, "isna") and np.ndim(obj) == 0 and pd.isna(obj)):
        return None
    return str(obj)


def implausible_prices(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows whose price is outside [IMPLAUSIBLE_LOW_USD, IMPLAUSIBLE_HIGH_USD].

    These are reported as *suspected data errors*, never silently dropped, so a
    human can decide.  A ``reason`` column says which bound was violated.
    """
    d = _df(df)
    p = d[d["price"].notna()]
    low = p[p["price"] < IMPLAUSIBLE_LOW_USD].assign(reason="below_floor")
    high = p[p["price"] > IMPLAUSIBLE_HIGH_USD].assign(reason="above_ceiling")
    cols = ["parent_asin", "brand", "segment", "title", "price", "is_renewed",
            "average_rating", "rating_number", "reason"]
    out = pd.concat([low, high])[cols] if len(low) or len(high) else pd.DataFrame(columns=cols)
    return out.sort_values("price").reset_index(drop=True)


def coverage_bias(df: pd.DataFrame | None = None) -> dict:
    """Compare priced vs unpriced products to expose selection bias in coverage.

    If priced products are systematically newer / better rated / from different
    segments, then even a correctly-labelled 30% sample is not representative,
    and the app should say so.  Returns per-attribute comparisons.
    """
    d = _df(df)
    priced = d["price"].notna()
    out = {
        "n_priced": int(priced.sum()),
        "n_unpriced": int((~priced).sum()),
        "coverage": float(priced.mean()),
        "mean_rating_priced": float(d.loc[priced, "average_rating"].mean()),
        "mean_rating_unpriced": float(d.loc[~priced, "average_rating"].mean()),
        "median_rating_number_priced": float(d.loc[priced, "rating_number"].median()),
        "median_rating_number_unpriced": float(d.loc[~priced, "rating_number"].median()),
        "renewed_share_priced": float(d.loc[priced, "is_renewed"].mean()),
        "renewed_share_unpriced": float(d.loc[~priced, "is_renewed"].mean()),
        "segment_share_priced": {
            k: float(v) for k, v in d.loc[priced, "segment"].value_counts(normalize=True).items()},
        "segment_share_unpriced": {
            k: float(v) for k, v in d.loc[~priced, "segment"].value_counts(normalize=True).items()},
    }
    out["segment_coverage_spread"] = float(
        d.groupby("segment", observed=True)["price"].apply(lambda s: s.notna().mean()).max()
        - d.groupby("segment", observed=True)["price"].apply(lambda s: s.notna().mean()).min()
    )
    return out


def validate(df: pd.DataFrame | None = None, out_path: Path | str | None = None) -> dict:
    """Run every sanity check and write ``eval/pricing_eval.json``.

    Checks
    ------
    1. Overall and per-segment price coverage (the module's headline caveat).
    2. Segment median ordering against EXPECTED_SEGMENT_ORDER
       (chromebook < budget < mainstream < gaming, budget/mainstream < ultrabook).
    3. Implausible prices (< $50 / > $10,000) listed as suspected data errors.
    4. Tiny-sample guard: how many segments/brands fall under MIN_RELIABLE_N.
    5. Coverage selection bias between priced and unpriced products.
    6. Internal consistency: p25 <= median <= p75, no negative prices, no priced
       row flagged ``price_is_missing`` and vice versa.

    Returns the report dict (also written to disk).
    """
    d = _df(df)
    out_path = Path(out_path) if out_path is not None else EVAL / "pricing_eval.json"

    seg_tbl = segment_price_table(d)
    brand_tbl = brand_price_table(d)
    warnings: list[str] = []
    checks: list[dict] = []

    # -- 1. coverage ------------------------------------------------------- #
    overall = price_summary(d["price"], n_total=len(d), label="ALL")
    if overall["coverage"] < 0.5:
        warnings.append(
            f"Only {overall['coverage']:.1%} of products carry a price "
            f"({overall['n']} of {overall['n_total']}). Every price aggregate in this "
            "system describes that subset, not the full catalogue.")

    # -- 2. segment ordering ------------------------------------------------ #
    seen_segments = set(seg_tbl["segment"])
    unexpected = sorted(seen_segments - set(SEGMENTS))
    absent = sorted(set(SEGMENTS) - seen_segments)
    checks.append({
        "name": "segment_vocabulary",
        "passed": not unexpected and not absent,
        "detail": (f"segments present: {len(seen_segments)}/{len(SEGMENTS)} from the config "
                   f"contract; missing={absent or 'none'}, unexpected={unexpected or 'none'}"),
    })
    if unexpected:
        warnings.append(f"Segments not in config.SEGMENTS present in data: {unexpected}")
    if absent:
        warnings.append(f"Segments declared in config.SEGMENTS but absent from data: {absent}")

    med = {r["segment"]: r["median"] for _, r in seg_tbl.iterrows()}
    nrel = {r["segment"]: (int(r["n"]), bool(r["reliable"])) for _, r in seg_tbl.iterrows()}
    order_results = []
    for lo, hi in EXPECTED_SEGMENT_ORDER:
        m_lo, m_hi = med.get(lo), med.get(hi)
        ok = (m_lo is not None and m_hi is not None and m_lo < m_hi)
        order_results.append({
            "expected": f"median({lo}) < median({hi})",
            "median_lo": m_lo, "median_hi": m_hi,
            "n_lo": nrel.get(lo, (0, False))[0], "n_hi": nrel.get(hi, (0, False))[0],
            "passed": bool(ok),
        })
        if not ok:
            warnings.append(f"Segment ordering violated: median({lo})={m_lo} !< median({hi})={m_hi}")
    n_pass = sum(r["passed"] for r in order_results)
    checks.append({
        "name": "segment_price_ordering",
        "passed": n_pass == len(order_results),
        "detail": f"{n_pass}/{len(order_results)} expected orderings hold",
    })

    # -- 3. implausible prices ---------------------------------------------- #
    bad = implausible_prices(d)
    n_priced = int(d["price"].notna().sum())
    checks.append({
        "name": "implausible_prices",
        "passed": bool(len(bad) / max(n_priced, 1) < 0.01),
        "detail": f"{len(bad)} of {n_priced} priced products outside "
                  f"[${IMPLAUSIBLE_LOW_USD:,.0f}, ${IMPLAUSIBLE_HIGH_USD:,.0f}]",
    })
    if len(bad):
        warnings.append(f"{len(bad)} suspected price data errors flagged (see implausible_prices).")

    # -- 4. tiny-sample guard ----------------------------------------------- #
    unreliable_segments = seg_tbl.loc[~seg_tbl["reliable"], "segment"].tolist()
    brands_priced = brand_tbl[brand_tbl["n"] > 0]
    unreliable_brands = int((~brand_tbl["reliable"]).sum())
    checks.append({
        "name": "tiny_sample_guard",
        "passed": len(unreliable_segments) == 0,
        "detail": f"{len(unreliable_segments)} segments and {unreliable_brands} of "
                  f"{len(brand_tbl)} brands have < {MIN_RELIABLE_N} priced products "
                  "and are flagged unreliable",
    })

    # -- 5. internal consistency -------------------------------------------- #
    consistency_problems = []
    if (d["price"].dropna() <= 0).any():
        consistency_problems.append("non-positive prices present")
    mism = int((d["price"].notna() & d["price_is_missing"]).sum()
               + (d["price"].isna() & ~d["price_is_missing"]).sum())
    if mism:
        consistency_problems.append(f"{mism} rows where price_is_missing disagrees with price")
    for _, r in seg_tbl.iterrows():
        if r["n"] and not (r["p25"] <= r["median"] <= r["p75"]):
            consistency_problems.append(f"quantile order broken for segment {r['segment']}")
    checks.append({
        "name": "internal_consistency",
        "passed": not consistency_problems,
        "detail": "; ".join(consistency_problems) or "price/flag/quantile invariants hold",
    })
    warnings.extend(consistency_problems)

    # -- 6. derived analyses ------------------------------------------------- #
    gpu_tbl = discrete_gpu_premium(d)
    rating_tbl = price_rating_relationship(d)
    pv = price_vs_value(d)
    model_meta = pv.attrs.get("price_model", {})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(PRODUCTS_PARQUET),
        "n_products": int(len(d)),
        "overall_price": overall,
        "coverage_warning": (
            "All price statistics are computed on the priced subset only "
            f"(n={overall['n']}, {overall['coverage']:.1%} of {overall['n_total']} products). "
            "Coverage differs sharply by segment, so cross-segment comparisons are "
            "comparisons of differently-sampled subsets."),
        "min_reliable_n": MIN_RELIABLE_N,
        "implausible_bounds_usd": [IMPLAUSIBLE_LOW_USD, IMPLAUSIBLE_HIGH_USD],
        "checks": checks,
        "segment_table": seg_tbl.to_dict(orient="records"),
        "segment_ordering": order_results,
        "unreliable_segments": unreliable_segments,
        "brand_table_top25_by_catalogue": brand_tbl.nlargest(25, "n_products").to_dict(orient="records"),
        "brand_summary": {
            "n_brands": int(len(brand_tbl)),
            "n_brands_with_any_price": int(len(brands_priced)),
            "n_brands_unreliable": unreliable_brands,
            "n_brands_reliable": int(len(brand_tbl) - unreliable_brands),
        },
        "implausible_prices": {
            "n": int(len(bad)),
            "share_of_priced": float(len(bad) / max(n_priced, 1)),
            "examples": bad.head(25).to_dict(orient="records"),
        },
        "coverage_bias": coverage_bias(d),
        "discrete_gpu_premium": gpu_tbl.to_dict(orient="records"),
        "price_vs_rating": rating_tbl.to_dict(orient="records"),
        "value_model": model_meta,
        "spec_price_ratios": {
            "median_price_per_ram_gb": float(pv["price_per_ram_gb"].median()),
            "n_price_per_ram_gb": int(pv["price_per_ram_gb"].notna().sum()),
            "median_price_per_storage_100gb": float(pv["price_per_storage_100gb"].median()),
            "n_price_per_storage_100gb": int(pv["price_per_storage_100gb"].notna().sum()),
        },
        "warnings": warnings,
        "passed": all(c["passed"] for c in checks),
    }

    report = _jsonable(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _fmt_table(tbl: pd.DataFrame, cols: Sequence[str]) -> str:
    """Render a price table with $ formatting and an explicit coverage column."""
    show = tbl[list(cols)].copy()
    for c in ("median", "mean", "std", "iqr", "p25", "p75", "min", "max"):
        if c in show:
            show[c] = show[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
    if "coverage" in show:
        show["coverage"] = show["coverage"].map(lambda v: f"{v:.1%}")
    if "mean_rating" in show:
        show["mean_rating"] = show["mean_rating"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    if "reliable" in show:
        show["reliable"] = show["reliable"].map(lambda v: "ok" if v else "UNRELIABLE")
    return show.to_string(index=False)


def main() -> None:
    """Print segment/brand price tables, example price positions and validation."""
    pd.set_option("display.width", 200)
    d = load_products()
    overall = price_summary(d["price"], n_total=len(d))
    print("=" * 100)
    print("PRICE COVERAGE")
    print("=" * 100)
    print(f"products                 : {overall['n_total']:,}")
    print(f"with a price             : {overall['n']:,}  ({overall['coverage']:.1%})")
    print(f"median / mean (priced)   : ${overall['median']:,.2f} / ${overall['mean']:,.2f}")
    print(f"IQR (priced)             : ${overall['p25']:,.0f} - ${overall['p75']:,.0f}"
          f"   range ${overall['min']:,.0f} - ${overall['max']:,.0f}")
    print("NOTE: every number below is computed on the priced subset only; the 'n' and")
    print("      'coverage' columns are the denominator and must be shown with the value.")

    cols = ["n_products", "n", "coverage", "reliable", "median", "mean", "std",
            "p25", "p75", "iqr", "min", "max", "mean_rating"]

    print("\n" + "=" * 100)
    print("SEGMENT PRICE TABLE (sorted by median)")
    print("=" * 100)
    print(_fmt_table(segment_price_table(d), ["segment"] + cols))

    print("\n" + "=" * 100)
    print("BRAND PRICE TABLE - top 20 brands by catalogue size")
    print("=" * 100)
    print(_fmt_table(brand_price_table(d, top=20), ["brand"] + cols))

    print("\n" + "=" * 100)
    print("BRAND PRICE TABLE - 10 brands flagged UNRELIABLE (n < %d priced)" % MIN_RELIABLE_N)
    print("=" * 100)
    bt = brand_price_table(d)
    print(_fmt_table(bt[~bt["reliable"]].nlargest(10, "n_products"), ["brand"] + cols))

    print("\n" + "=" * 100)
    print("DISCRETE GPU PREMIUM")
    print("=" * 100)
    g = discrete_gpu_premium(d)
    print(g[["segment", "n_discrete", "n_integrated", "median_discrete", "median_integrated",
             "premium_usd", "premium_pct", "mannwhitney_p", "reliable"]].round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("PRICE vs AVERAGE RATING (products with >=5 ratings and a price)")
    print("=" * 100)
    r = price_rating_relationship(d)
    keep = [c for c in ["segment", "n_pairs", "coverage", "pearson_r", "spearman_r", "spearman_p",
                        "rating_Q1", "rating_Q4", "reliable"] if c in r]
    print(r[keep].round(4).to_string(index=False))

    # --- example price positions ---------------------------------------- #
    priced = d[d["price"].notna() & (d["rating_number"] > 200)]
    examples = [
        priced[priced["segment"] == "gaming"].nlargest(1, "rating_number")["parent_asin"].iloc[0],
        priced[priced["segment"] == "budget"].nlargest(1, "rating_number")["parent_asin"].iloc[0],
    ]
    unpriced = d[d["price"].isna()].nlargest(1, "rating_number")["parent_asin"].iloc[0]
    for asin in examples + [unpriced]:
        pos = price_position(asin, d)
        print("\n" + "=" * 100)
        print(f"PRICE POSITION  {pos['parent_asin']}  [{pos['segment']} / {pos['brand']}]")
        print("=" * 100)
        print(f"  {pos['title'][:96]}")
        pstr = f"${pos['price']:,.2f}" if pos["price_available"] else "NO PRICE IN SOURCE DATA"
        print(f"  price {pstr}   rating {pos['average_rating']} from {pos['rating_number']:,} ratings")
        for key in ("vs_segment", "vs_brand"):
            b = pos[key]
            pct = f"{b['percentile']:.1f}th pct" if b["percentile"] is not None else "n/a"
            dlt = f"{b['delta_usd']:+,.0f} ({b['delta_pct']:+.1f}%)" if b["delta_usd"] is not None else "n/a"
            med = f"${b['median']:,.0f}" if b["median"] is not None else "n/a"
            print(f"  {key:<10} {b['label']:<11} {pct:<14} median {med:<9} "
                  f"delta {dlt:<20} n={b['n']:<5} coverage {b['coverage']:.1%}"
                  f"{'' if b['reliable'] else '  <-- UNRELIABLE'}")
        for note in pos["notes"]:
            print(f"  ! {note}")

    # --- comparison frame ------------------------------------------------ #
    print("\n" + "=" * 100)
    print("COMPARE_PRODUCTS example")
    print("=" * 100)
    cmp = compare_products(list(examples) + [unpriced], d)
    print(cmp[["parent_asin", "brand", "segment", "price", "price_label",
               "segment_price_percentile", "cpu_family", "ram_gb", "storage_gb",
               "gpu_model", "average_rating"]].to_string(index=False))
    print(f"  priced {cmp.attrs['n_priced']}/{cmp.attrs['n_requested']} "
          f"({cmp.attrs['price_coverage']:.0%})")
    if "warning" in cmp.attrs:
        print(f"  ! {cmp.attrs['warning']}")

    # --- value outliers -------------------------------------------------- #
    vo = value_outliers(d, top=5)
    print("\n" + "=" * 100)
    print("VALUE OUTLIERS vs spec price model")
    print("=" * 100)
    for name, frame in vo.items():
        print(f"\n-- {name} (pool n={frame.attrs['n_pool']:,} priced products with >=20 ratings)")
        print(frame[["parent_asin", "brand", "segment", "price", "expected_price",
                     "price_residual_pct", "average_rating"]].round(1).to_string(index=False))

    # --- validation ------------------------------------------------------ #
    rep = validate(d)
    print("\n" + "=" * 100)
    print("VALIDATION -> eval/pricing_eval.json")
    print("=" * 100)
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<26} {c['detail']}")
    print(f"  warnings: {len(rep['warnings'])}")
    for w in rep["warnings"][:6]:
        print(f"    - {w}")
    print(f"  overall passed: {rep['passed']}")


if __name__ == "__main__":
    main()
