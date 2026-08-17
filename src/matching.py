"""Embedding-based competitor matching for the laptop competitor-intelligence system.

The matcher answers one question: *given a laptop, which other laptops on the market
actually compete with it?*  It combines two complementary views of a product:

1. **Text view** - a compact natural-language document (title + brand + parsed specs +
   segment) embedded with :data:`config.EMBED_MODEL` via sentence-transformers.  Cosine
   similarity on those vectors captures the fuzzy, marketing-level notion of "same kind
   of machine" (e.g. "thin and light business ultrabook with a fingerprint reader").
2. **Structured view** - a masked, normalised distance over the parsed numeric specs
   (``cpu_tier``, ``ram_gb``, ``storage_gb``, ``screen_in``, ``is_discrete_gpu``,
   ``price``).  This is what keeps an RTX 4090 desktop-replacement away from a 4 GB
   Celeron even when both titles say "15.6 inch gaming laptop".

The two are blended with a tunable ``text_weight`` (see
:meth:`CompetitorMatcher.find_competitors`).

On top of the raw similarity sits the **guard**, which is the project's headline
analytical problem: a $199 Chromebook is textually and price-wise very close to a
*discounted* $249 business ThinkPad, yet the two do not compete.  The guard is:

* a **segment affinity matrix** (:data:`SEGMENT_AFFINITY`) - incompatible pairs such as
  ``business x chromebook`` or ``gaming x chromebook`` are dropped outright, adjacent
  pairs such as ``business x ultrabook`` survive with a small penalty;
* a **price band** - a candidate must sit inside ``[1/band, band] x query price``.  Price
  is missing for ~69% of the corpus, so a *hierarchical median estimate* is used as a
  stand-in with a deliberately widened band, and the estimated value never enters the
  structured distance (only genuinely observed prices do);
* **self / variant exclusion** - the query itself and any other listing that shares its
  model signature (the same signature the pipeline uses to collapse variants) are removed.

The guard is on by default and every part of it is configurable; passing
``apply_guard=False`` gives the pure hybrid similarity as an ablation.

Artifacts
---------
``config.EMBEDDINGS_NPY``  float32 ``(n_products, dim)`` L2-normalised matrix.
``config.EMBED_IDS_JSON``  ``{"model", "dim", "fingerprint", "ids": [...]}`` - the id list
is aligned row-for-row with the matrix, and the fingerprint lets the app detect a stale
cache instead of silently mis-aligning rows.

CLI
---
``python src/matching.py``            build (or reuse) the index and run the self-test
``python src/matching.py --rebuild``  force re-embedding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    EMBED_IDS_JSON,
    EMBED_MODEL,
    EMBEDDINGS_NPY,
    PRODUCT_KEY,
    PRODUCTS_PARQUET,
    SEGMENTS,
)
from pipeline import _model_signature  # noqa: E402  (reuse, do not duplicate)

# --------------------------------------------------------------------------------------
# 1. Tunables
# --------------------------------------------------------------------------------------

#: Default blend between the text-embedding cosine and the structured-spec similarity.
DEFAULT_TEXT_WEIGHT = 0.60

#: Structured features: (column, weight, normaliser kind, clip lo, clip hi).
#: ``kind`` is "lin" (linear in the raw value) or "log2"/"log10" (linear in the log).
SPEC_FEATURES: list[tuple[str, float, str, float, float]] = [
    ("cpu_tier", 1.00, "lin", 3.0, 9.0),
    ("ram_gb", 1.00, "log2", 2.0, 64.0),
    ("storage_gb", 0.80, "log2", 16.0, 4096.0),
    ("screen_in", 0.70, "lin", 10.0, 18.0),
    ("is_discrete_gpu", 1.00, "lin", 0.0, 1.0),
    ("price", 1.20, "log10", 80.0, 5000.0),
]

#: Distance assumed for a feature that is missing on either side (0 = identical,
#: 1 = opposite ends of the scale).  Slightly below 0.5 so a spec-poor listing is
#: penalised but not exiled.
NEUTRAL_DISTANCE = 0.45

#: Weight multiplier applied to a feature that had to fall back to NEUTRAL_DISTANCE.
#: Missing specs therefore dilute the distance instead of dominating it.
MISSING_SHRINK = 0.50

#: Symmetric segment affinity.  1.0 = same class, 0.0 = never competitors.
#: Only the unordered pairs are listed; the matrix is expanded symmetrically below.
_SEGMENT_PAIR_AFFINITY: dict[frozenset[str], float] = {
    frozenset({"gaming", "mainstream"}): 0.45,
    frozenset({"gaming", "ultrabook"}): 0.30,
    frozenset({"gaming", "business"}): 0.25,
    frozenset({"gaming", "budget"}): 0.10,
    frozenset({"gaming", "chromebook"}): 0.00,
    frozenset({"ultrabook", "business"}): 0.70,
    frozenset({"ultrabook", "mainstream"}): 0.50,
    frozenset({"ultrabook", "budget"}): 0.15,
    frozenset({"ultrabook", "chromebook"}): 0.05,
    frozenset({"business", "mainstream"}): 0.60,
    frozenset({"business", "budget"}): 0.35,
    frozenset({"business", "chromebook"}): 0.10,
    frozenset({"mainstream", "budget"}): 0.55,
    frozenset({"mainstream", "chromebook"}): 0.15,
    frozenset({"budget", "chromebook"}): 0.30,
}


def _build_affinity_matrix() -> dict[str, dict[str, float]]:
    """Expand :data:`_SEGMENT_PAIR_AFFINITY` into a full symmetric lookup."""
    matrix: dict[str, dict[str, float]] = {a: {b: 0.0 for b in SEGMENTS} for a in SEGMENTS}
    for a in SEGMENTS:
        matrix[a][a] = 1.0
    for pair, value in _SEGMENT_PAIR_AFFINITY.items():
        a, b = sorted(pair)
        matrix[a][b] = value
        matrix[b][a] = value
    return matrix


SEGMENT_AFFINITY: dict[str, dict[str, float]] = _build_affinity_matrix()

#: Guard defaults.
MIN_SEGMENT_AFFINITY = 0.25   # below this the candidate is dropped in guarded mode
SEGMENT_PENALTY_WEIGHT = 0.35  # score -= w * (1 - affinity)
PRICE_BAND = 2.25              # candidate price must be within [1/2.25, 2.25] x query
ESTIMATED_BAND_MULTIPLIER = 2.0  # widen the band when either price is an estimate
PRICE_PENALTY_WEIGHT = 0.15    # score -= w * (relative distance inside the band)


# --------------------------------------------------------------------------------------
# 2. Product documents
# --------------------------------------------------------------------------------------

_TITLE_MAX_WORDS = 44


def _fmt_num(value: Any, unit: str = "", decimals: int = 0) -> str:
    """Format a possibly-NaN number, returning ``''`` when it is unusable."""
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(f) or math.isinf(f):
        return ""
    return f"{f:.{decimals}f}{unit}"


def _price_bucket_label(price: float) -> str:
    """Coarse, human-readable price bucket used as a soft signal inside the document."""
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return ""
    for hi, label in ((300, "under $300 entry price"), (600, "$300-600 price"),
                      (1000, "$600-1000 price"), (1500, "$1000-1500 price"),
                      (2500, "$1500-2500 price")):
        if price < hi:
            return label
    return "over $2500 premium price"


def build_document(row: pd.Series | dict[str, Any]) -> str:
    """Build the natural-language document that represents one product for embedding.

    The document deliberately mixes the raw marketing title (truncated, because Amazon
    titles run to 200+ words of keyword stuffing and MiniLM only sees 256 tokens) with a
    canonical rendering of the parsed specs.  Writing the specs out in words lets the
    text encoder align "16 GB RAM" listings with each other even when one title spells it
    "16GB DDR5 Memory" and the other omits it entirely but has it in ``details``.

    Parameters
    ----------
    row:
        A row of ``products.parquet`` (Series or dict) following the ``config`` schema.

    Returns
    -------
    str
        A single-line document, e.g.
        ``"Lenovo ThinkPad T480 business laptop | Lenovo | Intel Core i5 ... "``.
    """
    get = row.get if isinstance(row, dict) else (lambda k, d=None: row[k] if k in row else d)

    title = str(get("title", "") or "")
    words = title.split()
    if len(words) > _TITLE_MAX_WORDS:
        title = " ".join(words[:_TITLE_MAX_WORDS])

    parts: list[str] = [title, f"brand {get('brand', '') or 'Unknown'}"]

    seg = str(get("segment", "") or "")
    if seg:
        parts.append(f"{seg} laptop")

    cpu_bits = [str(get("cpu_brand", "") or ""), str(get("cpu_family", "") or "")]
    cpu = " ".join(b for b in cpu_bits if b and b != "Unknown")
    ghz = _fmt_num(get("cpu_ghz"), " GHz", 1)
    if cpu or ghz:
        parts.append(f"processor {cpu} {ghz}".strip())

    ram = _fmt_num(get("ram_gb"), " GB", 0)
    ram_type = str(get("ram_type", "") or "")
    if ram:
        parts.append(f"{ram} {'' if ram_type in ('', 'Unknown') else ram_type} RAM".replace("  ", " "))

    storage = get("storage_gb")
    storage_type = str(get("storage_type", "") or "")
    if storage is not None and not (isinstance(storage, float) and math.isnan(storage)):
        s = float(storage)
        size = f"{s / 1024:.0f} TB" if s >= 1024 else f"{s:.0f} GB"
        parts.append(f"{size} {'' if storage_type in ('', 'Unknown') else storage_type} storage".replace("  ", " "))

    screen = _fmt_num(get("screen_in"), " inch", 1)
    res_w, res_h = get("screen_w"), get("screen_h")
    res = ""
    if res_w is not None and res_h is not None:
        try:
            if not (math.isnan(float(res_w)) or math.isnan(float(res_h))):
                res = f"{float(res_w):.0f}x{float(res_h):.0f}"
        except (TypeError, ValueError):
            res = ""
    if screen or res:
        parts.append(f"{screen} display {res}".strip())

    gpu_model = str(get("gpu_model", "") or "")
    gpu_brand = str(get("gpu_brand", "") or "")
    gpu_kind = "dedicated" if bool(get("is_discrete_gpu", False)) else "integrated"
    gpu_txt = " ".join(b for b in (gpu_brand, gpu_model) if b and b != "Unknown")
    parts.append(f"{gpu_kind} graphics {gpu_txt}".strip())

    os_family = str(get("os_family", "") or "")
    if os_family and os_family != "Unknown":
        parts.append(f"{os_family} operating system")

    weight = _fmt_num(get("weight_lb"), " lb", 1)
    if weight:
        parts.append(f"{weight} weight")

    price = get("price")
    bucket = _price_bucket_label(price if price is not None else float("nan"))
    if bucket:
        parts.append(bucket)

    if bool(get("is_renewed", False)):
        parts.append("renewed refurbished condition")

    return " | ".join(p for p in parts if p)


def build_documents(products: pd.DataFrame) -> list[str]:
    """Vectorised wrapper around :func:`build_document` for a whole product frame."""
    records = products.to_dict("records")
    return [build_document(r) for r in records]


# --------------------------------------------------------------------------------------
# 3. Embedding + cache
# --------------------------------------------------------------------------------------


def _fingerprint(model_name: str, ids: Sequence[str], docs: Sequence[str]) -> str:
    """SHA1 over the model name, the id order and the document text.

    Used to detect a stale embedding cache (pipeline re-run, doc-builder change, model
    swap) instead of silently serving vectors that no longer line up with the frame.
    """
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(ids)).encode("utf-8"))
    for i, d in zip(ids, docs):
        h.update(i.encode("utf-8"))
        h.update(b"\x00")
        h.update(d.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def embed_documents(
    docs: Sequence[str],
    model_name: str = EMBED_MODEL,
    device: str = "cpu",
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed documents with sentence-transformers and return L2-normalised float32 rows.

    ``device`` defaults to ``'cpu'`` on purpose: several agents share one 8 GB laptop GPU
    and MiniLM over ~26k short documents takes only a couple of minutes on CPU.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        list(docs),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(emb.astype(np.float32))


def load_products() -> pd.DataFrame:
    """Load ``products.parquet`` exactly as written by the pipeline."""
    return pd.read_parquet(PRODUCTS_PARQUET)


def build_index(
    products: pd.DataFrame | None = None,
    force: bool = False,
    device: str = "cpu",
    batch_size: int = 128,
    show_progress: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Return ``(ids, embeddings)``, reusing the on-disk cache when it is still valid.

    The cache is considered valid when the id list, the document fingerprint and the
    model name all match the current products frame, so the Streamlit app can call this
    on every launch and pay only the parquet read.

    Parameters
    ----------
    products:
        Optional pre-loaded product frame (loaded from parquet when omitted).
    force:
        Re-embed even when a valid cache exists.
    device:
        ``'cpu'`` (default) or ``'cuda'``.
    """
    if products is None:
        products = load_products()

    ids = products[PRODUCT_KEY].astype(str).tolist()
    docs = build_documents(products)
    fp = _fingerprint(EMBED_MODEL, ids, docs)

    if not force and EMBEDDINGS_NPY.exists() and EMBED_IDS_JSON.exists():
        try:
            meta = json.loads(EMBED_IDS_JSON.read_text())
            cached_ids = meta["ids"] if isinstance(meta, dict) else list(meta)
            cached_fp = meta.get("fingerprint") if isinstance(meta, dict) else None
            emb = np.load(EMBEDDINGS_NPY)
            if (
                len(cached_ids) == len(ids)
                and emb.shape[0] == len(ids)
                and cached_ids == ids
                and (cached_fp is None or cached_fp == fp)
            ):
                return ids, np.ascontiguousarray(emb.astype(np.float32))
            print("[matching] embedding cache is stale -> rebuilding")
        except Exception as exc:  # pragma: no cover - corrupt cache is rare
            print(f"[matching] could not read embedding cache ({exc}) -> rebuilding")

    t0 = time.time()
    print(f"[matching] embedding {len(docs):,} product documents on {device} ...")
    emb = embed_documents(docs, device=device, batch_size=batch_size, show_progress=show_progress)
    print(f"[matching] embedded in {time.time() - t0:.1f}s -> {emb.shape}")

    np.save(EMBEDDINGS_NPY, emb)
    EMBED_IDS_JSON.write_text(
        json.dumps({"model": EMBED_MODEL, "dim": int(emb.shape[1]), "fingerprint": fp, "ids": ids})
    )
    print(f"[matching] cached -> {EMBEDDINGS_NPY.name}, {EMBED_IDS_JSON.name}")
    return ids, emb


# --------------------------------------------------------------------------------------
# 4. Structured spec space
# --------------------------------------------------------------------------------------


def _normalise_feature(values: np.ndarray, kind: str, lo: float, hi: float) -> np.ndarray:
    """Map a raw feature column onto [0, 1]; NaN stays NaN (handled by the mask)."""
    v = values.astype(np.float64, copy=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "log2":
            v = np.log2(np.clip(v, lo, hi))
            lo_n, hi_n = math.log2(lo), math.log2(hi)
        elif kind == "log10":
            v = np.log10(np.clip(v, lo, hi))
            lo_n, hi_n = math.log10(lo), math.log10(hi)
        else:
            v = np.clip(v, lo, hi)
            lo_n, hi_n = lo, hi
        span = hi_n - lo_n if hi_n > lo_n else 1.0
        return (v - lo_n) / span


def build_spec_matrix(products: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the normalised structured-spec matrix, its presence mask and the weights.

    Returns
    -------
    feats : (n, d) float32
        Every column scaled to [0, 1]; NaN entries are replaced by 0 (never read, the
        mask blanks them out).
    mask : (n, d) float32
        1.0 where the spec was genuinely observed, 0.0 where it was missing.  Price is
        only ever "observed" when ``price_is_missing`` is False - the hierarchical
        estimate used by the price guard never enters the distance.
    weights : (d,) float32
        Per-feature importance from :data:`SPEC_FEATURES`.
    """
    n = len(products)
    d = len(SPEC_FEATURES)
    feats = np.zeros((n, d), dtype=np.float32)
    mask = np.zeros((n, d), dtype=np.float32)
    weights = np.zeros(d, dtype=np.float32)

    for j, (col, weight, kind, lo, hi) in enumerate(SPEC_FEATURES):
        weights[j] = weight
        raw = products[col]
        if raw.dtype == bool:
            values = raw.to_numpy(dtype=np.float64)
            present = np.ones(n, dtype=bool)
        else:
            values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
            present = np.isfinite(values)
        if col == "price":
            present &= ~products["price_is_missing"].to_numpy(dtype=bool)
        norm = _normalise_feature(values, kind, lo, hi)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        feats[:, j] = norm.astype(np.float32)
        mask[:, j] = present.astype(np.float32)

    return feats, mask, weights


def spec_similarity(
    feats: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    qi: int,
) -> np.ndarray:
    """Masked structured similarity of every row against row ``qi``.

    A feature contributes its true absolute distance only when it is present on *both*
    sides.  Otherwise it contributes :data:`NEUTRAL_DISTANCE` at a reduced weight
    (:data:`MISSING_SHRINK`), so a listing with no parsed price is neither dropped nor
    rewarded with a free perfect match - exactly the behaviour needed for a corpus where
    69% of prices are unknown.

    Returns
    -------
    (n,) float32 similarity in [0, 1].
    """
    q_feat = feats[qi]
    q_mask = mask[qi]

    both = mask * q_mask                      # (n, d) 1 where comparable
    diff = np.abs(feats - q_feat)             # (n, d) already in [0, 1]

    eff_w = weights * (both + (1.0 - both) * MISSING_SHRINK)
    num = (eff_w * (both * diff + (1.0 - both) * NEUTRAL_DISTANCE)).sum(axis=1)
    den = eff_w.sum(axis=1)
    den = np.where(den <= 0, 1.0, den)
    dist = num / den
    return (1.0 - dist).astype(np.float32)


def _estimate_prices(products: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing prices with a hierarchical median for *guard purposes only*.

    Fallback ladder, most specific first:
    ``(segment, renewed, cpu_family, ram bucket)`` -> ``(segment, renewed, cpu_tier)``
    -> ``(segment, renewed)`` -> ``segment`` -> global median.

    Returns
    -------
    price_eff : (n,) float64
        Observed price where known, else the estimate.
    is_estimated : (n,) bool
    """
    df = products
    price = pd.to_numeric(df["price"], errors="coerce")
    observed = price.notna() & ~df["price_is_missing"].to_numpy(dtype=bool)
    known = price.where(observed)

    ram_bucket = pd.cut(
        pd.to_numeric(df["ram_gb"], errors="coerce"),
        bins=[-1, 4, 8, 16, 32, 1e9],
        labels=["r4", "r8", "r16", "r32", "r64"],
    ).astype("object").fillna("rna")
    tier = pd.to_numeric(df["cpu_tier"], errors="coerce").fillna(-1).astype(int).astype(str)
    renewed = df["is_renewed"].astype(str)
    seg = df["segment"].astype(str)
    fam = df["cpu_family"].astype(str)

    keys = [
        seg + "|" + renewed + "|" + fam + "|" + ram_bucket.astype(str),
        seg + "|" + renewed + "|" + tier,
        seg + "|" + renewed,
        seg,
    ]

    est = known.copy()
    for key in keys:
        if est.notna().all():
            break
        med = known.groupby(key.to_numpy()).median()
        counts = known.groupby(key.to_numpy()).count()
        med = med.where(counts >= 3)  # ignore medians built on <3 observations
        filler = pd.Series(key.map(med).to_numpy(), index=est.index)
        est = est.fillna(filler)
    est = est.fillna(float(known.median()) if known.notna().any() else 500.0)

    return est.to_numpy(dtype=np.float64), (~observed.to_numpy(dtype=bool))


# --------------------------------------------------------------------------------------
# 5. The matcher
# --------------------------------------------------------------------------------------

_RESULT_COLUMNS = [
    PRODUCT_KEY, "title", "brand", "segment", "price", "price_effective", "price_is_estimated",
    "cpu_brand", "cpu_family", "cpu_tier", "ram_gb", "storage_gb", "storage_type", "screen_in",
    "gpu_model", "is_discrete_gpu", "os_family", "is_renewed", "average_rating", "rating_number",
    "n_reviews", "text_sim", "spec_sim", "segment_affinity", "price_ratio", "score",
]


class CompetitorMatcher:
    """Hybrid text + spec competitor index over ``products.parquet``.

    Examples
    --------
    >>> m = CompetitorMatcher.load()                       # doctest: +SKIP
    >>> m.find_competitors("B07KML89F3", k=10)             # doctest: +SKIP
    """

    def __init__(self, products: pd.DataFrame, ids: Sequence[str], embeddings: np.ndarray) -> None:
        if len(ids) != len(products) or embeddings.shape[0] != len(products):
            raise ValueError(
                f"embedding/product misalignment: {len(products)} products, "
                f"{len(ids)} ids, {embeddings.shape[0]} vectors"
            )
        if list(products[PRODUCT_KEY].astype(str)) != list(ids):
            raise ValueError("embedding id order does not match the products frame")

        self.products = products.reset_index(drop=True)
        self.ids = list(ids)
        self.index = {pid: i for i, pid in enumerate(self.ids)}

        # L2-normalise defensively so cosine == dot product.
        emb = np.ascontiguousarray(embeddings.astype(np.float32))
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms

        self.feats, self.mask, self.weights = build_spec_matrix(self.products)
        self.price_eff, self.price_estimated = _estimate_prices(self.products)
        self.log_price = np.log(np.clip(self.price_eff, 1.0, None))

        self.segment = self.products["segment"].astype(str).to_numpy()
        self.model_sig = np.array(
            [
                _model_signature(t, b)
                for t, b in zip(self.products["title"].astype(str), self.products["brand"].astype(str))
            ],
            dtype=object,
        )
        self.rating_number = pd.to_numeric(
            self.products["rating_number"], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.float64)

        # affinity lookup as a dense (n_segments, n_segments) array for vectorised use
        self._seg_codes = np.array(
            [SEGMENTS.index(s) if s in SEGMENTS else -1 for s in self.segment], dtype=np.int32
        )
        self._affinity = np.zeros((len(SEGMENTS), len(SEGMENTS)), dtype=np.float32)
        for i, a in enumerate(SEGMENTS):
            for j, b in enumerate(SEGMENTS):
                self._affinity[i, j] = SEGMENT_AFFINITY[a][b]

    # -- construction ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        products: pd.DataFrame | None = None,
        force_rebuild: bool = False,
        device: str = "cpu",
        show_progress: bool = True,
    ) -> "CompetitorMatcher":
        """Build the matcher, embedding the corpus only if the cache is missing/stale."""
        if products is None:
            products = load_products()
        ids, emb = build_index(
            products, force=force_rebuild, device=device, show_progress=show_progress
        )
        return cls(products, ids, emb)

    # -- helpers -----------------------------------------------------------------------

    def row(self, parent_asin: str) -> pd.Series:
        """Return the products row for ``parent_asin`` (raises ``KeyError`` if absent)."""
        if parent_asin not in self.index:
            raise KeyError(f"{parent_asin!r} is not in products.parquet")
        return self.products.iloc[self.index[parent_asin]]

    def segment_affinity_vector(self, segment: str) -> np.ndarray:
        """Affinity of every product in the corpus to ``segment``."""
        if segment not in SEGMENTS:
            return np.full(len(self.products), 0.5, dtype=np.float32)
        row = self._affinity[SEGMENTS.index(segment)]
        out = np.full(len(self.products), 0.5, dtype=np.float32)
        valid = self._seg_codes >= 0
        out[valid] = row[self._seg_codes[valid]]
        return out

    # -- main API ----------------------------------------------------------------------

    def find_competitors(
        self,
        parent_asin: str,
        k: int = 10,
        text_weight: float = DEFAULT_TEXT_WEIGHT,
        apply_guard: bool = True,
        min_segment_affinity: float = MIN_SEGMENT_AFFINITY,
        price_band: float = PRICE_BAND,
        estimated_band_multiplier: float = ESTIMATED_BAND_MULTIPLIER,
        segment_penalty_weight: float = SEGMENT_PENALTY_WEIGHT,
        price_penalty_weight: float = PRICE_PENALTY_WEIGHT,
        exclude_same_model: bool = True,
        exclude_same_brand: bool = False,
        max_per_brand: int | None = 3,
        min_rating_number: int = 0,
        include_renewed: bool = True,
    ) -> pd.DataFrame:
        """Rank the ``k`` most credible competitors of ``parent_asin``.

        Parameters
        ----------
        parent_asin:
            Query product id (must exist in ``products.parquet``).
        k:
            Number of competitors to return.
        text_weight:
            Blend between the embedding cosine (``text_weight``) and the structured spec
            similarity (``1 - text_weight``).  0 = specs only, 1 = text only.
        apply_guard:
            Master switch for the segment + price-band guard.  ``True`` (default) drops
            incompatible-segment and out-of-band candidates and applies the soft
            penalties; ``False`` returns the raw hybrid ranking (useful as an ablation to
            show *why* the guard exists).
        min_segment_affinity:
            Candidates whose segment affinity to the query is below this are removed.
            The default (0.25) blocks e.g. business x chromebook and gaming x chromebook
            while keeping business x ultrabook and budget x mainstream.
        price_band:
            Multiplicative price window: a candidate must satisfy
            ``1/band <= price_c / price_q <= band``.
        estimated_band_multiplier:
            The band is widened by this factor when either side's price is a hierarchical
            estimate rather than an observed value, so the 69% of listings with no price
            are not thrown away on the strength of a guess.
        segment_penalty_weight, price_penalty_weight:
            Soft penalties subtracted from the score for adjacent-but-not-identical
            segments and for price gaps inside the band.
        exclude_same_model:
            Drop other listings that share the query's model signature (remaining
            configuration variants / re-listings of the same machine).
        exclude_same_brand:
            Drop the query's own brand entirely (handy for "who else sells against us?").
        max_per_brand:
            Diversity cap - at most this many results per brand (``None`` disables it).
            Without it a query for an Acer Predator returns ten other Acer Predators,
            because the title text dominates the embedding; the cap surfaces the
            cross-brand rivals that competitor intelligence actually needs.  The cap is
            relaxed automatically if it would leave fewer than ``k`` results.
        min_rating_number:
            Ignore listings with fewer than this many Amazon ratings (0 = keep all).
        include_renewed:
            When False, renewed/refurbished listings are excluded.

        Returns
        -------
        pandas.DataFrame
            ``k`` rows sorted by ``score`` descending with the key specs plus the score
            components (``text_sim``, ``spec_sim``, ``segment_affinity``, ``price_ratio``).
        """
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError("text_weight must be in [0, 1]")
        qi = self.index.get(parent_asin)
        if qi is None:
            raise KeyError(f"{parent_asin!r} is not in products.parquet")

        n = len(self.products)
        text_sim = self.embeddings @ self.embeddings[qi]          # cosine, rows are unit norm
        spec_sim = spec_similarity(self.feats, self.mask, self.weights, qi)
        score = text_weight * text_sim + (1.0 - text_weight) * spec_sim

        affinity = self.segment_affinity_vector(self.segment[qi])

        # price ratio against the (possibly estimated) query price
        log_ratio = self.log_price - self.log_price[qi]
        price_ratio = np.exp(log_ratio)
        est_either = self.price_estimated | bool(self.price_estimated[qi])
        band = np.where(est_either, price_band * estimated_band_multiplier, price_band)
        log_band = np.log(np.maximum(band, 1.0001))

        keep = np.ones(n, dtype=bool)
        keep[qi] = False                                          # never return the query

        if exclude_same_model:
            sig = self.model_sig[qi]
            if sig:
                keep &= self.model_sig != sig
        if exclude_same_brand:
            keep &= self.products["brand"].astype(str).to_numpy() != str(self.products["brand"].iloc[qi])
        if min_rating_number > 0:
            keep &= self.rating_number >= min_rating_number
        if not include_renewed:
            keep &= ~self.products["is_renewed"].to_numpy(dtype=bool)

        if apply_guard:
            keep &= affinity >= min_segment_affinity
            keep &= np.abs(log_ratio) <= log_band
            rel_gap = np.clip(np.abs(log_ratio) / log_band, 0.0, 1.0)
            score = score - segment_penalty_weight * (1.0 - affinity) - price_penalty_weight * rel_gap

        idx = np.flatnonzero(keep)
        if idx.size == 0:
            return self.products.head(0).assign(
                price_effective=[], price_is_estimated=[], text_sim=[], spec_sim=[],
                segment_affinity=[], price_ratio=[], score=[],
            )[_RESULT_COLUMNS]

        k = int(min(max(k, 1), idx.size))
        if max_per_brand is None or max_per_brand <= 0:
            top_local = np.argpartition(-score[idx], k - 1)[:k]
            top = idx[top_local]
            top = top[np.argsort(-score[top])]
        else:
            top = self._select_diverse(idx, score, k, int(max_per_brand))

        out = self.products.iloc[top].copy()
        out["price_effective"] = self.price_eff[top]
        out["price_is_estimated"] = self.price_estimated[top]
        out["text_sim"] = text_sim[top]
        out["spec_sim"] = spec_sim[top]
        out["segment_affinity"] = affinity[top]
        out["price_ratio"] = price_ratio[top]
        out["score"] = score[top]
        return out[_RESULT_COLUMNS].reset_index(drop=True)

    def _select_diverse(
        self, idx: np.ndarray, score: np.ndarray, k: int, max_per_brand: int
    ) -> np.ndarray:
        """Greedy top-k over ``idx`` with a per-brand cap, relaxed if it starves the list.

        Only a bounded pool of the highest scoring candidates is materialised, so the cap
        costs a sort over a few hundred rows rather than the whole corpus.
        """
        pool_size = int(min(idx.size, max(400, k * 60)))
        pool_local = np.argpartition(-score[idx], pool_size - 1)[:pool_size] \
            if pool_size < idx.size else np.arange(idx.size)
        pool = idx[pool_local]
        pool = pool[np.argsort(-score[pool])]

        brands = self.products["brand"].astype(str).to_numpy()
        chosen: list[int] = []
        overflow: list[int] = []
        counts: dict[str, int] = {}
        for i in pool:
            b = brands[i]
            if counts.get(b, 0) < max_per_brand:
                counts[b] = counts.get(b, 0) + 1
                chosen.append(i)
                if len(chosen) == k:
                    break
            else:
                overflow.append(i)
        if len(chosen) < k:  # not enough distinct brands -> relax the cap
            chosen.extend(overflow[: k - len(chosen)])
        out = np.array(chosen[:k], dtype=int)
        return out[np.argsort(-score[out])]

    # -- convenience -------------------------------------------------------------------

    def sample_products(
        self, segment: str, n: int = 1, min_reviews: int = 50, require_price: bool = True,
        max_price: float | None = None, min_price: float | None = None,
    ) -> pd.DataFrame:
        """Pick popular, well-populated example products from a segment (for demos/tests)."""
        df = self.products
        sel = df[df["segment"] == segment]
        if require_price:
            sel = sel[~sel["price_is_missing"]]
        if max_price is not None:
            sel = sel[sel["price"] <= max_price]
        if min_price is not None:
            sel = sel[sel["price"] >= min_price]
        sel = sel[sel["n_reviews"] >= min_reviews]
        return sel.nlargest(n, "n_reviews")


@lru_cache(maxsize=1)
def get_matcher() -> CompetitorMatcher:
    """Process-wide cached matcher (the Streamlit app should use this)."""
    return CompetitorMatcher.load(show_progress=False)


def find_competitors(parent_asin: str, k: int = 10, **kwargs: Any) -> pd.DataFrame:
    """Module-level shortcut around :meth:`CompetitorMatcher.find_competitors`.

    Uses a process-wide cached matcher so repeated calls pay the parquet/embedding load
    only once.
    """
    return get_matcher().find_competitors(parent_asin, k=k, **kwargs)


# --------------------------------------------------------------------------------------
# 6. Self-test
# --------------------------------------------------------------------------------------

_SHOW = [
    PRODUCT_KEY, "brand", "segment", "cpu_family", "ram_gb", "storage_gb", "screen_in",
    "gpu_model", "price_effective", "price_is_estimated", "score",
]


def _fmt_row(r: pd.Series, rank: int | str = "Q") -> str:
    price = r.get("price_effective", r.get("price"))
    est = bool(r.get("price_is_estimated", False))
    price_s = "n/a" if price is None or (isinstance(price, float) and math.isnan(price)) \
        else f"${price:,.0f}{'~' if est else ' '}"
    specs = (
        f"{str(r['cpu_family'])[:12]:<12} "
        f"{('%.0fGB' % r['ram_gb']) if pd.notna(r['ram_gb']) else '  ?GB':>6} "
        f"{('%.0fGB' % r['storage_gb']) if pd.notna(r['storage_gb']) else '  ?GB':>7} "
        f"{('%.1f\"' % r['screen_in']) if pd.notna(r['screen_in']) else '  ?"':>6} "
        f"{str(r['gpu_model'])[:16]:<16}"
    )
    score = f"{r['score']:.3f}" if "score" in r and pd.notna(r.get("score")) else "  -  "
    return (
        f"  {str(rank):>2} {score} {str(r['segment']):<11} {price_s:>9} "
        f"{str(r['brand'])[:14]:<14} {specs} {str(r['title'])[:64]}"
    )


def _demo(matcher: CompetitorMatcher, label: str, row: pd.Series, k: int = 10, **kwargs: Any) -> pd.DataFrame:
    """Print the query product followed by its ranked competitors."""
    print("\n" + "=" * 150)
    print(f"### {label}")
    q = row.copy()
    i = matcher.index[row[PRODUCT_KEY]]
    q["price_effective"] = matcher.price_eff[i]
    q["price_is_estimated"] = matcher.price_estimated[i]
    print(_fmt_row(q, "Q"))
    print("-" * 150)
    res = matcher.find_competitors(row[PRODUCT_KEY], k=k, **kwargs)
    for rank, (_, r) in enumerate(res.iterrows(), 1):
        print(_fmt_row(r, rank))
    return res


def _self_test(matcher: CompetitorMatcher) -> None:
    """Eyeball test: real products from each segment plus the Chromebook-guard check."""
    picks = [
        ("GAMING", matcher.sample_products("gaming", 1, min_reviews=100)),
        ("ULTRABOOK", matcher.sample_products("ultrabook", 1, min_reviews=100)),
        ("BUSINESS", matcher.sample_products("business", 1, min_reviews=50)),
        ("BUDGET", matcher.sample_products("budget", 1, min_reviews=100)),
        ("CHROMEBOOK", matcher.sample_products("chromebook", 1, min_reviews=100)),
    ]
    for label, sel in picks:
        if sel.empty:
            print(f"[self-test] no example found for {label}")
            continue
        _demo(matcher, f"{label} query", sel.iloc[0], k=10, min_rating_number=10)

    # ---- headline guard: a cheap / discounted business laptop --------------------------
    # Pick, deterministically, the low-priced business laptop whose *unguarded* top-10 is
    # most polluted by Chromebooks - i.e. the hardest case for the guard.
    cheap = matcher.products[
        (matcher.products["segment"] == "business")
        & (~matcher.products["price_is_missing"])
        & (matcher.products["price"] <= 400)
        & (matcher.products["n_reviews"] >= 5)
    ].nlargest(150, "n_reviews")
    if cheap.empty:
        print("[self-test] no cheap business laptop found for the guard test")
        return
    worst_asin, worst_leaks = cheap.iloc[0][PRODUCT_KEY], (-1, -1)
    for _, r in cheap.iterrows():
        off = matcher.find_competitors(r[PRODUCT_KEY], k=10, apply_guard=False)
        # rank candidates by Chromebook pollution first, then by budget-netbook pollution
        leaks = (int((off["segment"] == "chromebook").sum()),
                 int((off["segment"] == "budget").sum()))
        if leaks > worst_leaks:
            worst_asin, worst_leaks = r[PRODUCT_KEY], leaks
    q = matcher.row(worst_asin)

    guarded = _demo(matcher, "GUARD ON  - cheap business laptop (no Chromebooks expected)",
                    q, k=10)
    unguarded = _demo(matcher, "GUARD OFF - same query, pure hybrid similarity (ablation)",
                      q, k=10, apply_guard=False)

    n_chrome_on = int((guarded["segment"] == "chromebook").sum())
    n_chrome_off = int((unguarded["segment"] == "chromebook").sum())
    print("\n" + "=" * 150)
    print(f"[guard check] {worst_asin}: chromebooks in top-10 guard ON = {n_chrome_on} (must be 0), "
          f"guard OFF = {n_chrome_off}")
    assert n_chrome_on == 0, "GUARD FAILED: a chromebook leaked into a business laptop's competitors"

    # ---- corpus-wide leakage sweep -----------------------------------------------------
    pop = matcher.products[matcher.products["n_reviews"] >= 20]
    sample = pop.sample(n=min(200, len(pop)), random_state=0)
    stats = {}
    for mode, guard in (("ON", True), ("OFF", False)):
        cross = price_out = total = 0
        for _, r in sample.iterrows():
            res = matcher.find_competitors(r[PRODUCT_KEY], k=10, apply_guard=guard)
            total += len(res)
            cross += int((res["segment_affinity"] < MIN_SEGMENT_AFFINITY).sum())
            ratio = res["price_ratio"].to_numpy()
            price_out += int(((ratio > 3.0) | (ratio < 1 / 3.0)).sum())
        stats[mode] = (total, cross, price_out)
        print(f"[sweep] guard {mode:<3}: {total} recommendations over {len(sample)} queries -> "
              f"{cross} cross-class (affinity < {MIN_SEGMENT_AFFINITY}), "
              f"{price_out} beyond a 3x price gap")
    # cross-class leakage must be exactly zero; the 3x price metric is only a report, since
    # the guard legitimately allows a wider band when a price had to be estimated.
    assert stats["ON"][1] == 0, "GUARD FAILED: cross-class competitors survived the guard"
    assert stats["ON"][2] <= stats["OFF"][2], "GUARD FAILED: price spread got worse with the guard on"
    print("[self-test] OK")


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point: build/refresh the embedding index and run the self-test."""
    ap = argparse.ArgumentParser(description="Build the competitor-matching index and self-test it.")
    ap.add_argument("--rebuild", action="store_true", help="force re-embedding of all products")
    ap.add_argument("--device", default="cpu", help="torch device for embedding (default: cpu)")
    ap.add_argument("--asin", default=None, help="print competitors for one parent_asin and exit")
    ap.add_argument("-k", type=int, default=10, help="number of competitors to show")
    ap.add_argument("--no-guard", action="store_true", help="disable the segment/price guard")
    args = ap.parse_args(list(argv) if argv is not None else None)

    products = load_products()
    print(f"[matching] products: {len(products):,} rows")
    matcher = CompetitorMatcher.load(products, force_rebuild=args.rebuild, device=args.device)

    if args.asin:
        _demo(matcher, f"query {args.asin}", matcher.row(args.asin), k=args.k,
              apply_guard=not args.no_guard)
        return 0

    _self_test(matcher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
