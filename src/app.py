"""Streamlit front-end for the laptop competitor-intelligence system.

Five sections, all driven by one *focus product* held in ``st.session_state``:

1. **Explorer**   - filter the 25.8k-row catalogue and pick the focus product.
2. **Competitors** - :func:`matching.find_competitors` as a table + a plotly scatter.
3. **Pricing**    - :mod:`pricing` positioning inside segment and brand.
4. **Reviews**    - :mod:`sentiment` aspect profile, focus vs competitors, verbatims.
5. **Ask**        - :mod:`rag` natural-language Q&A rendered with its citations.

Design rules enforced throughout
--------------------------------
*Never show a market statistic without its denominator.*  Every median/percentile in
the UI is printed with the ``n`` priced products it came from and the coverage of the
group it describes - only 31% of the catalogue carries a price.

*Never render a missing price as blank or 0.*  :func:`fmt_price` returns the literal
string ``"price not listed"`` (``"not listed"`` inside dense tables, where the column
header already says *Price*), and unpriced rows are dropped from price charts with an
on-screen count of how many were dropped.

*The LLM is optional.*  The catalogue, matching, pricing and sentiment tabs never touch
torch.  The agent's weights load lazily on the first question; if that load fails the
tab shows the error, offers the retrieval-only view (which needs no GPU) and the rest
of the app keeps working.

Run with::

    streamlit run src/app.py
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Environment guard - MUST run before pandas (and therefore pyarrow) is imported.
#
# pyarrow 25 defaults to the mimalloc allocator, whose per-thread heap is torn down when
# the thread that first touched it exits.  Streamlit runs every script run in a *fresh*
# ScriptRunner thread, so the second run - i.e. the first widget interaction - segfaults
# the whole interpreter inside any pyarrow call: ``df.parent_asin == asin`` on a
# pandas-3 arrow-backed string column, the ``take`` behind boolean indexing, or
# ``st.dataframe``'s arrow serialisation.  Reproduced on a two-line app
# (``st.dataframe(pd.DataFrame({"b": ["x"]}))``), so it is an environment property, not
# an app bug.  Selecting Arrow's plain system allocator removes the thread-local heap and
# the crash with it; the cost is negligible at this data size.
# --------------------------------------------------------------------------------------
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
_ARROW_PREIMPORTED = "pyarrow" in sys.modules

# Streamlit executes this file as ``__main__`` from an arbitrary CWD; make the sibling
# modules importable before anything else.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ASPECTS, PRODUCT_KEY, SEGMENTS


LOG = logging.getLogger("app")


def arrow_pool_backend() -> str:
    """Name of the Arrow allocator actually in use (``'system'`` is the safe one)."""
    try:
        import pyarrow as pa

        return str(pa.default_memory_pool().backend_name)
    except Exception:  # pragma: no cover - pyarrow always present alongside pandas 3
        return "unavailable"

# ======================================================================================
# 0. Look and feel
# ======================================================================================

st.set_page_config(
    page_title="Laptop Competitor Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Validated categorical slots (light surface).  Scatter/heatmap forms use at most the
# first three, which are the slots that clear the all-pairs CVD floors.
C_PEER = "#2a78d6"      # slot 1 - blue    : peers / the population
C_FOCUS = "#eb6834"     # slot 2 - orange  : the focus product, everywhere
C_POS = "#2a78d6"       # diverging + pole
C_NEG = "#e34948"       # diverging - pole
C_MID = "#f0efec"       # diverging neutral midpoint
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e2"

MISSING_PRICE_TEXT = "price not listed"
MISSING_PRICE_CELL = "not listed"
UNKNOWN = "unknown"
POS_LABEL = "positive"
NEG_LABEL = "negative"

ASPECT_NAMES = list(ASPECTS)

EXAMPLE_QUESTIONS = [
    "Compare the Acer Predator Helios 300 gaming laptop with the ASUS TUF Gaming A15 - "
    "which one is the better buy on specs and price?",
    "Which gaming laptops under $1200 do reviewers rate best on thermals and fan noise?",
    "Who are the main competitors to the Acer Aspire 3 slim laptop, and how are they "
    "positioned on price?",
    "What is the typical price of a gaming laptop compared with a chromebook in this "
    "market, and how much does a discrete GPU add?",
    "What is the best thin and light Windows ultrabook under $900 with 16GB of RAM?",
]


# ======================================================================================
# 1. Formatting helpers
# ======================================================================================


def fmt_price(value: Any, missing: str = MISSING_PRICE_TEXT) -> str:
    """Format a price, or say so explicitly when it is absent.

    A missing price is never rendered as an empty cell or as ``$0`` - the catalogue
    only prices ~31% of its rows and the difference between "cheap" and "unknown"
    is the whole point of the pricing module.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return missing
    return f"${float(value):,.0f}"


def fmt_num(value: Any, unit: str = "", decimals: int = 0, missing: str = UNKNOWN) -> str:
    """Format a possibly-missing number with an optional unit suffix."""
    if value is None or pd.isna(value):
        return missing
    return f"{float(value):,.{decimals}f}{unit}"


def fmt_text(value: Any, missing: str = UNKNOWN) -> str:
    """Normalise the catalogue's string sentinels ('Unknown', '', None) to one token."""
    if value is None or pd.isna(value):
        return missing
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "unknown", "other"}:
        return missing
    return text


def fmt_pct(value: Any, decimals: int = 0, missing: str = UNKNOWN) -> str:
    """Format a 0-1 fraction as a percentage."""
    if value is None or pd.isna(value):
        return missing
    return f"{100.0 * float(value):.{decimals}f}%"


def fmt_signed_pct(value: Any, missing: str = UNKNOWN) -> str:
    """Format an already-scaled percentage delta with an explicit sign."""
    if value is None or pd.isna(value):
        return missing
    return f"{float(value):+.0f}%"


def short_title(text: Any, max_chars: int = 70) -> str:
    """Trim a marketing title to something that fits in a chart label or a selectbox."""
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= max_chars else clean[: max_chars - 1].rstrip() + "…"


def spec_line(row: pd.Series) -> str:
    """One-line spec summary used in tables, hovers and headers."""
    bits = [
        fmt_text(row.get("cpu_family")),
        f"{fmt_num(row.get('ram_gb'), ' GB')} RAM",
        f"{fmt_num(row.get('storage_gb'), ' GB')} {fmt_text(row.get('storage_type'), '')}".strip(),
        f"{fmt_num(row.get('screen_in'), '\"', 1)}",
    ]
    gpu = fmt_text(row.get("gpu_model"))
    if gpu != UNKNOWN:
        bits.append(gpu)
    return " | ".join(b for b in bits if b and b != UNKNOWN)


def n_caption(n: int, n_total: int, what: str = "priced") -> str:
    """Render the denominator caption that must accompany every market statistic."""
    cov = (n / n_total) if n_total else 0.0
    return f"n={n:,} {what} of {n_total:,} listings ({cov:.0%} coverage)"


# ======================================================================================
# 2. Cached loaders
# ======================================================================================


@st.cache_resource(show_spinner=False)
def startup_banner() -> str:
    """Log the environment facts worth knowing exactly once per server process."""
    msg = (f"arrow memory pool backend: {arrow_pool_backend()} "
           f"(pyarrow pre-imported: {_ARROW_PREIMPORTED}); "
           f"{len(load_products()):,} products, {len(sentiment_ids()):,} with review sentiment")
    LOG.warning("[app] %s", msg)
    return msg


@st.cache_resource(show_spinner="Loading catalogue and embedding index…")
def load_matcher():
    """Process-wide competitor matcher (products.parquet + cached MiniLM embeddings).

    ``cache_resource`` rather than ``cache_data`` because the matcher holds a 40 MB
    float32 embedding matrix that must not be pickled/copied per session.
    """
    import matching

    return matching.get_matcher()


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    """The catalogue frame, in the exact row order the embedding index uses."""
    return load_matcher().products.copy()


@st.cache_data(show_spinner=False)
def brand_options() -> list[str]:
    """Brands ordered by catalogue size (most listings first)."""
    return load_products()["brand"].value_counts().index.tolist()


@st.cache_data(show_spinner=False)
def os_options() -> list[str]:
    """Distinct OS families present in the catalogue."""
    return sorted(load_products()["os_family"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def competitors_for(
    asin: str,
    k: int,
    apply_guard: bool,
    exclude_same_brand: bool,
    max_per_brand: int,
    include_renewed: bool,
) -> pd.DataFrame:
    """Cached wrapper around :meth:`matching.CompetitorMatcher.find_competitors`."""
    return load_matcher().find_competitors(
        asin,
        k=k,
        apply_guard=apply_guard,
        exclude_same_brand=exclude_same_brand,
        max_per_brand=max_per_brand,
        include_renewed=include_renewed,
    )


@st.cache_data(show_spinner=False)
def segment_prices() -> pd.DataFrame:
    """Per-segment price table with coverage and reliability flags."""
    import pricing

    return pricing.segment_price_table(load_products())


@st.cache_data(show_spinner=False)
def brand_prices(min_products: int = 25, top: int = 30) -> pd.DataFrame:
    """Per-brand price table restricted to brands with a meaningful catalogue size."""
    import pricing

    return pricing.brand_price_table(load_products(), min_products=min_products, top=top)


@st.cache_data(show_spinner=False)
def price_position_for(asin: str) -> dict:
    """Cached :func:`pricing.price_position` for one product."""
    import pricing

    return pricing.price_position(asin, load_products())


@st.cache_data(show_spinner=False)
def coverage_report() -> dict:
    """Cached :func:`pricing.coverage_bias` - the app's headline price caveat."""
    import pricing

    return pricing.coverage_bias(load_products())


@st.cache_data(show_spinner=False)
def compare_frame(asins: tuple[str, ...]) -> pd.DataFrame:
    """Cached :func:`pricing.compare_products` for a tuple of ASINs (order preserved)."""
    import pricing

    out = pricing.compare_products(list(asins), load_products())
    # ``attrs`` does not survive the cache pickle round-trip; the callers recompute the
    # priced/unpriced counts from the frame itself.
    return out


@st.cache_data(show_spinner=False)
def sentiment_ids() -> set[str]:
    """ASINs that have a mined review-sentiment profile (a minority of the catalogue)."""
    import sentiment

    try:
        return set(sentiment.load_product_sentiment().index)
    except FileNotFoundError:
        return set()


@st.cache_data(show_spinner=False)
def product_sentiment(asin: str) -> dict | None:
    """Cached :func:`sentiment.get_product_sentiment`."""
    import sentiment

    try:
        return sentiment.get_product_sentiment(asin)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def verbatims(asin: str, negative: bool, k: int, aspect: str | None) -> list[dict]:
    """Cached top praise/complaint snippets for a product."""
    import sentiment

    try:
        if negative:
            return sentiment.top_complaints(asin, k=k, aspect=aspect)
        return sentiment.top_praises(asin, k=k, aspect=aspect)
    except FileNotFoundError:
        return []


@st.cache_resource(show_spinner="Preparing the retrieval index…")
def load_agent():
    """The RAG agent (retriever eagerly, LLM weights lazily on the first question).

    Constructing :class:`rag.RagAgent` only *handles* the model; the 4-bit weights are
    materialised inside ``LocalLLM.load()`` on the first :meth:`chat` call, so this
    stays cheap and CPU-only until the user actually asks something.
    """
    import rag

    return rag.get_agent()


@st.cache_data(show_spinner=False, max_entries=64)
def retrieve_only(question: str) -> dict:
    """Run retrieval without the LLM - always available, never touches the GPU."""
    agent = load_agent()
    spec, evidence, notes, stats = agent.retrieve(question)
    return {
        "spec": spec.to_dict(),
        "evidence": [e.to_dict() for e in evidence],
        "notes": notes,
        "stats": stats,
    }


@st.cache_data(show_spinner=False, max_entries=32)
def generate_answer(question: str, max_new_tokens: int) -> dict:
    """Full RAG answer as a plain dict (greedy decoding, so caching is sound)."""
    _prepare_torch_for_streamlit()
    return load_agent().answer(question, max_new_tokens=max_new_tokens).to_dict()


def _prepare_torch_for_streamlit() -> None:
    """Neutralise the ``torch.classes`` / Streamlit module-watcher interaction.

    Streamlit's local source watcher walks ``sys.modules`` and touches
    ``module.__path__._path``; ``torch.classes`` raises a custom error from that
    attribute, which surfaces as a spurious exception in the browser.  Emptying the
    attribute before the model loads keeps the watcher quiet.
    """
    try:
        import torch

        torch.classes.__path__ = []  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - torch missing or already patched
        pass


# ======================================================================================
# 3. Plot helpers
# ======================================================================================


def style_fig(fig: go.Figure, height: int = 420, showlegend: bool = True) -> go.Figure:
    """Apply the shared chart styling: recessive grid, text-token ink, top legend."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, size=13),
        title_font=dict(color=INK, size=15),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="",
                    font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID,
                     title_font=dict(color=INK_2), tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID,
                     title_font=dict(color=INK_2), tickfont=dict(color=INK_2))
    return fig


def marker_sizes(rating_number: pd.Series, lo: float = 9.0, hi: float = 26.0) -> np.ndarray:
    """Map review counts to marker areas on a log scale, floored at a legible 9px."""
    n = pd.to_numeric(rating_number, errors="coerce").fillna(0).to_numpy(dtype=float)
    logs = np.log1p(np.clip(n, 0, None))
    span = logs.max() - logs.min()
    if span <= 0:
        return np.full(len(logs), lo)
    return lo + (hi - lo) * (logs - logs.min()) / span


def diverging_colors(values: np.ndarray) -> list[str]:
    """Two-pole colouring for polarity bars: blue above zero, red below."""
    return [C_POS if v >= 0 else C_NEG for v in values]


def polarity_colorscale() -> list[list]:
    """Blue-gray-red diverging scale for the -1..+1 polarity heatmap."""
    return [[0.0, C_NEG], [0.5, C_MID], [1.0, C_POS]]


# ======================================================================================
# 4. Catalogue filtering
# ======================================================================================


def apply_filters(df: pd.DataFrame, f: dict[str, Any]) -> pd.DataFrame:
    """Apply the sidebar filter dict to the catalogue and return the surviving rows."""
    mask = pd.Series(True, index=df.index)

    if f["query"]:
        haystack = df["title"].fillna("").str.lower() + " " + df["brand"].fillna("").str.lower()
        for token in f["query"].lower().split():
            mask &= haystack.str.contains(token, regex=False)
    if f["brands"]:
        mask &= df["brand"].isin(f["brands"])
    if f["segments"]:
        mask &= df["segment"].isin(f["segments"])
    if f["os"]:
        mask &= df["os_family"].isin(f["os"])

    lo, hi = f["price_range"]
    in_band = df["price"].between(lo, hi)
    mask &= in_band | (df["price"].isna() if f["include_unpriced"] else False)

    if f["min_ram"]:
        mask &= df["ram_gb"].fillna(-1) >= f["min_ram"]
    if f["min_storage"]:
        mask &= df["storage_gb"].fillna(-1) >= f["min_storage"]
    if f["min_cpu_tier"]:
        mask &= df["cpu_tier"].fillna(-1) >= f["min_cpu_tier"]

    s_lo, s_hi = f["screen_range"]
    if (s_lo, s_hi) != f["screen_bounds"]:
        mask &= df["screen_in"].between(s_lo, s_hi)

    if f["gpu"] == "Discrete only":
        mask &= df["is_discrete_gpu"]
    elif f["gpu"] == "Integrated only":
        mask &= ~df["is_discrete_gpu"]

    if f["min_rating"] > 0:
        mask &= df["average_rating"].fillna(0) >= f["min_rating"]
    if f["min_ratings_count"] > 0:
        mask &= df["rating_number"].fillna(0) >= f["min_ratings_count"]
    if f["exclude_renewed"]:
        mask &= ~df["is_renewed"]
    if f["only_with_reviews"]:
        mask &= df[PRODUCT_KEY].isin(sentiment_ids())

    return df[mask]


SORTS: dict[str, tuple[str, bool]] = {
    "Most reviewed": ("rating_number", False),
    "Highest rated": ("average_rating", False),
    "Price: low to high": ("price", True),
    "Price: high to low": ("price", False),
    "Most RAM": ("ram_gb", False),
    "Most storage": ("storage_gb", False),
}


def sort_catalogue(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Sort the result set, always pushing missing values to the bottom."""
    col, asc = SORTS[how]
    return df.sort_values(col, ascending=asc, na_position="last", kind="stable")


def sidebar_filters(df: pd.DataFrame) -> dict[str, Any]:
    """Render the sidebar controls and return the chosen filter values."""
    st.sidebar.header("Catalogue filters")

    price_bounds = (0, int(np.ceil(df["price"].max() / 100.0) * 100))
    screen_bounds = (float(np.floor(df["screen_in"].min())), float(np.ceil(df["screen_in"].max())))

    f: dict[str, Any] = {}
    f["query"] = st.sidebar.text_input("Search title / brand", key="flt_query",
                                       placeholder="e.g. thinkpad x1 carbon")
    f["brands"] = st.sidebar.multiselect("Brand", brand_options(), key="flt_brands")
    f["segments"] = st.sidebar.multiselect("Segment", SEGMENTS, key="flt_segments")

    with st.sidebar.expander("Price", expanded=True):
        f["price_range"] = st.slider(
            "Listed price (USD)", price_bounds[0], price_bounds[1], price_bounds, step=50,
            key="flt_price",
        )
        f["include_unpriced"] = st.checkbox(
            "Include products with no listed price",
            value=True, key="flt_unpriced",
            help=f"Only {coverage_report()['coverage']:.0%} of the catalogue carries a price. "
                 "Unpriced products cannot be filtered by budget - keep them visible or "
                 "you are silently looking at a third of the market.",
        )

    with st.sidebar.expander("Specs"):
        f["min_ram"] = st.selectbox("Minimum RAM (GB)", [0, 4, 8, 16, 32, 64], key="flt_ram",
                                    format_func=lambda v: "any" if not v else f"{v} GB+")
        f["min_storage"] = st.selectbox("Minimum storage (GB)", [0, 128, 256, 512, 1024, 2048],
                                        key="flt_storage",
                                        format_func=lambda v: "any" if not v else f"{v} GB+")
        f["min_cpu_tier"] = st.selectbox(
            "Minimum CPU tier", [0, 3, 5, 7, 9], key="flt_cpu_tier",
            format_func=lambda v: "any" if not v else f"i{v} / Ryzen {v} or better",
        )
        f["screen_bounds"] = screen_bounds
        f["screen_range"] = st.slider("Screen size (inches)", screen_bounds[0], screen_bounds[1],
                                      screen_bounds, step=0.5, key="flt_screen")
        f["gpu"] = st.radio("Graphics", ["Any", "Discrete only", "Integrated only"], key="flt_gpu")
        f["os"] = st.multiselect("Operating system", os_options(), key="flt_os")

    with st.sidebar.expander("Quality"):
        f["min_rating"] = st.slider("Minimum average rating", 0.0, 5.0, 0.0, step=0.1,
                                    key="flt_min_rating")
        f["min_ratings_count"] = st.number_input("Minimum number of Amazon ratings", min_value=0,
                                                 value=0, step=10, key="flt_min_ratings_count")
        f["exclude_renewed"] = st.checkbox("Exclude renewed / refurbished", value=False,
                                           key="flt_exclude_renewed")
        f["only_with_reviews"] = st.checkbox(
            "Only products with mined review sentiment", value=False, key="flt_only_reviews",
            help=f"{len(sentiment_ids()):,} of {len(df):,} products have review text scored "
                 "by the sentiment pass.",
        )
    return f


# ======================================================================================
# 5. Focus product
# ======================================================================================


def default_focus(df: pd.DataFrame) -> str:
    """Pick a sensible landing product: the most-reviewed priced laptop with sentiment."""
    ids = sentiment_ids()
    pool = df[df["price"].notna() & df[PRODUCT_KEY].isin(ids)] if ids else df[df["price"].notna()]
    if pool.empty:
        pool = df
    return str(pool.nlargest(1, "n_reviews").iloc[0][PRODUCT_KEY])


def focus_row(df: pd.DataFrame) -> pd.Series:
    """The currently selected product as a catalogue row."""
    asin = st.session_state.get("focus_asin")
    hit = df[df[PRODUCT_KEY] == asin]
    if hit.empty:
        asin = default_focus(df)
        st.session_state["focus_asin"] = asin
        hit = df[df[PRODUCT_KEY] == asin]
    return hit.iloc[0]


def set_focus(asin: str) -> None:
    """Point the whole app at a different product."""
    st.session_state["focus_asin"] = str(asin)


def focus_header(row: pd.Series) -> None:
    """Render the persistent focus-product banner shown above every tab."""
    st.markdown(f"**Focus product** · `{row[PRODUCT_KEY]}`")
    st.markdown(f"### {short_title(row['title'], 130)}")
    cols = st.columns(6)
    cols[0].metric("Price", fmt_price(row["price"]))
    cols[1].metric("Brand", fmt_text(row["brand"]))
    cols[2].metric("Segment", fmt_text(row["segment"]))
    cols[3].metric("Rating", fmt_num(row["average_rating"], "", 1),
                   delta=f"{int(row['rating_number']):,} ratings",
                   delta_color="off", delta_arrow="off")
    cols[4].metric("Reviews kept", f"{int(row['n_reviews']):,}")
    cols[5].metric("Review sentiment",
                   "available" if row[PRODUCT_KEY] in sentiment_ids() else "not mined")
    st.caption(
        f"{spec_line(row)} · {fmt_text(row['os_family'])} · "
        f"{fmt_num(row['weight_lb'], ' lb', 1)}"
        + ("  · **renewed / refurbished listing**" if bool(row["is_renewed"]) else "")
    )


# ======================================================================================
# 6. Tab: product explorer
# ======================================================================================


TABLE_COLUMNS = {
    "title": st.column_config.TextColumn("Title", width="large"),
    "brand": st.column_config.TextColumn("Brand"),
    "segment": st.column_config.TextColumn("Segment"),
    "price_text": st.column_config.TextColumn("Price", help=MISSING_PRICE_TEXT),
    "cpu": st.column_config.TextColumn("CPU"),
    "ram": st.column_config.TextColumn("RAM"),
    "storage": st.column_config.TextColumn("Storage"),
    "screen": st.column_config.TextColumn("Screen"),
    "gpu": st.column_config.TextColumn("GPU"),
    "os_family": st.column_config.TextColumn("OS"),
    "rating": st.column_config.TextColumn("Rating"),
    "rating_number": st.column_config.NumberColumn("# ratings", format="%d"),
    "n_reviews": st.column_config.NumberColumn("# reviews", format="%d"),
    "is_renewed": st.column_config.CheckboxColumn("Renewed"),
    PRODUCT_KEY: st.column_config.TextColumn("ASIN"),
}


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Project catalogue rows onto display-ready columns with explicit missing tokens."""
    out = pd.DataFrame(index=df.index)
    out["title"] = df["title"].map(lambda t: short_title(t, 90))
    out["brand"] = df["brand"].map(fmt_text)
    out["segment"] = df["segment"].map(fmt_text)
    out["price_text"] = df["price"].map(lambda p: fmt_price(p, MISSING_PRICE_CELL))
    out["cpu"] = df["cpu_family"].map(fmt_text)
    out["ram"] = df["ram_gb"].map(lambda v: fmt_num(v, " GB"))
    out["storage"] = [
        f"{fmt_num(g, ' GB')} {fmt_text(t, '')}".strip() if pd.notna(g) else UNKNOWN
        for g, t in zip(df["storage_gb"], df["storage_type"])
    ]
    out["screen"] = df["screen_in"].map(lambda v: fmt_num(v, '"', 1))
    out["gpu"] = df["gpu_model"].map(fmt_text)
    out["os_family"] = df["os_family"].map(fmt_text)
    out["rating"] = df["average_rating"].map(lambda v: fmt_num(v, "", 1))
    out["rating_number"] = df["rating_number"]
    out["n_reviews"] = df["n_reviews"]
    out["is_renewed"] = df["is_renewed"]
    out[PRODUCT_KEY] = df[PRODUCT_KEY]
    return out


def tab_explorer(df: pd.DataFrame, results: pd.DataFrame, focus: pd.Series) -> None:
    """Filtered catalogue browser and focus-product picker."""
    st.subheader("Product explorer")

    n_priced = int(results["price"].notna().sum())
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"**{len(results):,}** of {len(df):,} listings match the current filters "
            f"· {n_caption(n_priced, len(results))}."
        )
    with right:
        sort_by = st.selectbox("Sort by", list(SORTS), index=0, key="explorer_sort")

    if results.empty:
        st.warning("No products match these filters. Loosen them in the sidebar.")
        return

    ordered = sort_catalogue(results, sort_by)
    shown = ordered.head(500)

    picker_options = shown[PRODUCT_KEY].tolist()
    labels = {
        r[PRODUCT_KEY]: f"{short_title(r['title'], 80)}  —  {fmt_price(r['price'])}"
        for _, r in shown.iterrows()
    }
    current = st.session_state.get("focus_asin")
    index = picker_options.index(current) if current in picker_options else 0
    chosen = st.selectbox(
        "Focus product (type to filter; the other tabs all follow this choice)",
        picker_options,
        index=index,
        format_func=lambda a: labels.get(a, a),
        key="explorer_focus_picker",
    )
    picked, reset = st.columns([1, 5])
    if picked.button("Set as focus", type="primary", key="explorer_set_focus"):
        set_focus(chosen)
        st.rerun()
    if current not in picker_options:
        reset.caption(
            f"The current focus product (`{current}`) is outside the active filters; "
            "it stays selected until you pick a new one."
        )

    st.dataframe(
        display_table(shown),
        column_config=TABLE_COLUMNS,
        column_order=list(TABLE_COLUMNS),
        hide_index=True,
        height=430,
    )
    st.caption(
        f"Showing the first {len(shown):,} of {len(ordered):,} matches. "
        f"'{MISSING_PRICE_CELL}' means the source listing carries no price - it is not $0."
    )

    with st.expander("Full record for the focus product"):
        record = {
            "ASIN": str(focus[PRODUCT_KEY]),
            "Title": str(focus["title"]),
            "Brand / store": f"{fmt_text(focus['brand'])} ({fmt_text(focus['store'])})",
            "Segment": fmt_text(focus["segment"]),
            "Price": fmt_price(focus["price"]),
            "CPU": f"{fmt_text(focus['cpu_brand'])} {fmt_text(focus['cpu_family'], '')} "
                   f"{fmt_num(focus['cpu_ghz'], ' GHz', 1)}".strip(),
            "RAM": f"{fmt_num(focus['ram_gb'], ' GB')} {fmt_text(focus['ram_type'], '')}".strip(),
            "Storage": f"{fmt_num(focus['storage_gb'], ' GB')} {fmt_text(focus['storage_type'], '')}".strip(),
            "Screen": f"{fmt_num(focus['screen_in'], ' in', 1)} "
                      f"({fmt_num(focus['screen_w'])}x{fmt_num(focus['screen_h'])})",
            "GPU": f"{fmt_text(focus['gpu_brand'])} {fmt_text(focus['gpu_model'], '')}".strip()
                   + (" (discrete)" if bool(focus["is_discrete_gpu"]) else " (integrated)"),
            "OS": fmt_text(focus["os_family"]),
            "Weight": fmt_num(focus["weight_lb"], " lb", 2),
            "Renewed": "yes" if bool(focus["is_renewed"]) else "no",
            "Amazon rating": f"{fmt_num(focus['average_rating'], '', 1)} "
                             f"from {int(focus['rating_number']):,} ratings",
            "Reviews retained": f"{int(focus['n_reviews']):,}",
            "Variants merged": f"{int(focus['n_variants']):,}",
        }
        st.dataframe(
            pd.DataFrame({"field": list(record), "value": list(record.values())}),
            hide_index=True, height=min(560, 36 * len(record)),
        )


# ======================================================================================
# 7. Tab: competitors
# ======================================================================================


SCATTER_METRICS = {
    "Average rating": ("average_rating", "Amazon average rating", 1, ""),
    "RAM (GB)": ("ram_gb", "RAM (GB)", 0, " GB"),
    "Storage (GB)": ("storage_gb", "Storage (GB)", 0, " GB"),
    "Screen (inches)": ("screen_in", "Screen size (inches)", 1, '"'),
    "CPU clock (GHz)": ("cpu_ghz", "CPU base clock (GHz)", 1, " GHz"),
    "Weight (lb)": ("weight_lb", "Weight (lb)", 1, " lb"),
}


def competitor_scatter(focus: pd.Series, comps: pd.DataFrame, metric_label: str) -> go.Figure | None:
    """Price vs a chosen metric, focus highlighted; unpriced rows are excluded upstream."""
    col, axis_title, decimals, unit = SCATTER_METRICS[metric_label]
    frame = pd.concat([focus.to_frame().T, comps], ignore_index=True)
    frame["is_focus"] = [True] + [False] * len(comps)
    # concat with a transposed Series yields object dtype columns; force the numerics back
    for numeric in {"price", "rating_number", col}:
        frame[numeric] = pd.to_numeric(frame[numeric], errors="coerce")
    plot = frame[frame["price"].notna() & frame[col].notna()]
    if plot.empty:
        return None

    sizes = marker_sizes(plot["rating_number"])
    fig = go.Figure()
    for is_focus, name, color in ((False, "Competitors", C_PEER), (True, "Focus product", C_FOCUS)):
        sel = plot["is_focus"] == is_focus
        if not sel.any():
            continue
        sub = plot[sel]
        fig.add_trace(
            go.Scatter(
                x=sub["price"],
                y=sub[col],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=sizes[sel.to_numpy()],
                    line=dict(color=SURFACE, width=2),  # 2px surface ring on overlap
                    symbol="diamond" if is_focus else "circle",
                ),
                customdata=np.stack(
                    [
                        sub["title"].map(lambda t: short_title(t, 60)),
                        sub["brand"].map(fmt_text),
                        sub["segment"].map(fmt_text),
                        sub["rating_number"].fillna(0).astype(int).map("{:,}".format),
                        [spec_line(r) for _, r in sub.iterrows()],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}"
                    "<br>Price $%{x:,.0f}<br>" + axis_title + ": %{y:,." + str(decimals) + "f}"
                    "<br>%{customdata[3]} ratings<br>%{customdata[4]}<extra></extra>"
                ),
            )
        )
    if bool((plot["is_focus"]).any()):
        f = plot[plot["is_focus"]].iloc[0]
        fig.add_annotation(
            x=float(f["price"]), y=float(f[col]),
            text=short_title(f["title"], 34), showarrow=True, arrowhead=0, arrowcolor=C_FOCUS,
            ax=0, ay=-34, font=dict(color=INK, size=12), bgcolor="rgba(255,255,255,0.85)",
        )
    fig.update_layout(title=f"Listed price vs {axis_title.lower()} (marker size = # ratings)")
    fig.update_xaxes(title_text="Listed price (USD)", tickprefix="$", separatethousands=True)
    fig.update_yaxes(title_text=axis_title, ticksuffix=unit)
    return style_fig(fig)


def tab_competitors(focus: pd.Series, comps: pd.DataFrame, opts: dict[str, Any]) -> None:
    """Competitive set for the focus product: table + scatter."""
    st.subheader("Competitive set")
    st.caption(
        "Ranked by a hybrid of MiniLM title/spec-document similarity and structured spec "
        "similarity, then guarded by segment affinity and a price band."
    )

    if comps.empty:
        st.warning(
            "No competitors survived the guard for this product. Turn the segment/price "
            "guard off in the controls above to see the raw similarity ranking."
        )
        return

    asins = [str(focus[PRODUCT_KEY])] + comps[PRODUCT_KEY].astype(str).tolist()
    table = compare_frame(tuple(asins))
    score = comps.set_index(PRODUCT_KEY)["score"].to_dict()
    text_sim = comps.set_index(PRODUCT_KEY)["text_sim"].to_dict()
    spec_sim = comps.set_index(PRODUCT_KEY)["spec_sim"].to_dict()

    view = pd.DataFrame({
        "role": ["FOCUS"] + [f"#{i}" for i in range(1, len(comps) + 1)],
        "title": table["title"].map(lambda t: short_title(t, 80)),
        "brand": table["brand"].map(fmt_text),
        "segment": table["segment"].map(fmt_text),
        "price_text": table["price"].map(lambda p: fmt_price(p, MISSING_PRICE_CELL)),
        "vs_segment": [
            f"{fmt_signed_pct(d)} vs median" if pd.notna(d) else UNKNOWN
            for d in table["price_vs_segment_median_pct"]
        ],
        "cpu": table["cpu_family"].map(fmt_text),
        "ram": table["ram_gb"].map(lambda v: fmt_num(v, " GB")),
        "storage": [
            f"{fmt_num(g, ' GB')} {fmt_text(t, '')}".strip() if pd.notna(g) else UNKNOWN
            for g, t in zip(table["storage_gb"], table["storage_type"])
        ],
        "screen": table["screen_in"].map(lambda v: fmt_num(v, '"', 1)),
        "gpu": table["gpu_model"].map(fmt_text),
        "rating": [
            f"{fmt_num(r, '', 1)} ({int(n):,})" for r, n in
            zip(table["average_rating"], table["rating_number"].fillna(0))
        ],
        "similarity": [""] + [f"{score.get(a, float('nan')):.3f}" for a in asins[1:]],
        "text/spec": [""] + [
            f"{text_sim.get(a, float('nan')):.2f} / {spec_sim.get(a, float('nan')):.2f}"
            for a in asins[1:]
        ],
        PRODUCT_KEY: table[PRODUCT_KEY],
    })

    st.dataframe(
        view, hide_index=True, height=min(560, 42 + 36 * len(view)),
        column_config={
            "role": st.column_config.TextColumn("Rank", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "brand": st.column_config.TextColumn("Brand"),
            "segment": st.column_config.TextColumn("Segment"),
            "price_text": st.column_config.TextColumn("Price", help=MISSING_PRICE_TEXT),
            "vs_segment": st.column_config.TextColumn("Price vs segment median"),
            "cpu": st.column_config.TextColumn("CPU"),
            "ram": st.column_config.TextColumn("RAM"),
            "storage": st.column_config.TextColumn("Storage"),
            "screen": st.column_config.TextColumn("Screen"),
            "gpu": st.column_config.TextColumn("GPU"),
            "rating": st.column_config.TextColumn("Rating (# ratings)"),
            "similarity": st.column_config.TextColumn("Match score"),
            "text/spec": st.column_config.TextColumn("text / spec sim"),
            PRODUCT_KEY: st.column_config.TextColumn("ASIN"),
        },
    )

    n_priced = int(table["price"].notna().sum())
    if n_priced < len(table):
        st.info(
            f"{len(table) - n_priced} of {len(table)} products in this comparison have no "
            f"listed price ({MISSING_PRICE_TEXT}); their price cells are blank rather than "
            "zero and they are omitted from the chart below."
        )

    chart_col, pick_col = st.columns([4, 1])
    with pick_col:
        metric = st.selectbox("Y axis", list(SCATTER_METRICS), key="comp_metric")
        st.caption("Only priced products can appear on a price axis.")
    with chart_col:
        fig = competitor_scatter(focus, comps, metric)
        if fig is None:
            st.warning(
                "Neither the focus product nor its competitors have both a listed price "
                f"and a value for {metric.lower()}, so this chart cannot be drawn."
            )
        else:
            st.plotly_chart(fig, theme=None, key="comp_scatter")

    st.markdown("**Jump to a competitor**")
    cols = st.columns(min(5, len(comps)))
    for i, (_, r) in enumerate(comps.head(len(cols)).iterrows()):
        with cols[i]:
            st.caption(f"{short_title(r['title'], 46)}\n\n{fmt_price(r['price'])}")
            if st.button("Focus", key=f"focus_comp_{r[PRODUCT_KEY]}"):
                set_focus(str(r[PRODUCT_KEY]))
                st.rerun()

    with st.expander("What the guard does"):
        st.markdown(
            "- **Segment affinity** drops candidates whose segment is incompatible with the "
            "focus product's (a chromebook is not a competitor to a gaming laptop) and "
            "penalises weak-but-allowed pairs.\n"
            "- **Price band** keeps candidates within a x2.25 band of the focus price; when "
            "either price is missing the matcher falls back to a spec-based price estimate "
            "and widens the band, which is why some rows show "
            f"'{MISSING_PRICE_CELL}' but still rank.\n"
            f"- Guard currently **{'on' if opts['apply_guard'] else 'off'}**, max "
            f"{opts['max_per_brand']} listings per brand, renewed listings "
            f"{'included' if opts['include_renewed'] else 'excluded'}."
        )


# ======================================================================================
# 8. Tab: pricing
# ======================================================================================


def segment_median_chart(seg_table: pd.DataFrame, focus_segment: str) -> go.Figure:
    """Median listed price per segment, focus segment highlighted, every bar labelled n."""
    tbl = seg_table.dropna(subset=["median"]).sort_values("median")
    is_focus = tbl["segment"] == focus_segment
    fig = go.Figure()
    for flag, name, color in ((False, "Other segments", C_PEER), (True, "Focus segment", C_FOCUS)):
        sub = tbl[is_focus == flag]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["segment"], y=sub["median"], name=name,
                marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                text=[f"${m:,.0f}<br>n={int(n):,}" for m, n in zip(sub["median"], sub["n"])],
                textposition="outside", textfont=dict(color=INK_2, size=11),
                customdata=np.column_stack([
                    pd.to_numeric(sub[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    for c in ("n", "n_products", "coverage", "p25", "p75")
                ]),
                hovertemplate=(
                    "<b>%{x}</b><br>median $%{y:,.0f}"
                    "<br>IQR $%{customdata[3]:,.0f} - $%{customdata[4]:,.0f}"
                    "<br>n=%{customdata[0]:,.0f} priced of %{customdata[1]:,.0f} listings"
                    "<br>coverage %{customdata[2]:.0%}<extra></extra>"
                ),
            )
        )
    fig.update_layout(title="Median listed price by segment (n = priced listings behind each bar)",
                      bargap=0.35)
    fig.update_yaxes(title_text="Median listed price (USD)", tickprefix="$", separatethousands=True)
    fig.update_xaxes(title_text="")
    return style_fig(fig, height=380)


def segment_distribution_chart(prices: pd.Series, segment: str, focus_price: float | None,
                               stats: dict) -> go.Figure:
    """Price histogram for the focus product's segment with quartile and focus markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=prices, nbinsx=40, name=f"'{segment}' priced listings",
            marker=dict(color=C_PEER, line=dict(color=SURFACE, width=1)),
            hovertemplate="$%{x}<br>%{y} listings<extra></extra>",
        )
    )
    for key, label, dash in (("p25", "p25", "dot"), ("median", "median", "dash"), ("p75", "p75", "dot")):
        value = stats.get(key)
        if value:
            fig.add_vline(x=value, line=dict(color=INK_2, width=1, dash=dash),
                          annotation_text=f"{label} ${value:,.0f}",
                          annotation_font=dict(color=INK_2, size=11))
    if focus_price is not None:
        fig.add_vline(x=focus_price, line=dict(color=C_FOCUS, width=3),
                      annotation_text=f"focus ${focus_price:,.0f}",
                      annotation_font=dict(color=C_FOCUS, size=12))
    fig.update_layout(
        title=f"Where the focus product sits in the '{segment}' price distribution "
              f"({n_caption(stats['n'], stats['n_total'])})"
    )
    fig.update_xaxes(title_text="Listed price (USD)", tickprefix="$", separatethousands=True)
    fig.update_yaxes(title_text="Listings")
    return style_fig(fig, height=380, showlegend=False)


def tab_pricing(df: pd.DataFrame, focus: pd.Series) -> None:
    """Price positioning of the focus product inside its segment and brand."""
    st.subheader("Price positioning")

    cov = coverage_report()
    st.warning(
        f"**Coverage caveat.** Only **{cov['n_priced']:,} of "
        f"{cov['n_priced'] + cov['n_unpriced']:,}** listings ({cov['coverage']:.1%}) carry a "
        f"price in the source dump, and coverage is not uniform - it varies by "
        f"{cov['segment_coverage_spread']:.0%} across segments. Priced listings also skew "
        f"more-reviewed (median {cov['median_rating_number_priced']:,.0f} vs "
        f"{cov['median_rating_number_unpriced']:,.0f} ratings). Every statistic below is "
        "printed with the n it was computed from; treat them as describing the *priced* "
        "subset, not the whole market.",
    )

    pos = price_position_for(str(focus[PRODUCT_KEY]))
    seg, brand = pos["vs_segment"], pos["vs_brand"]

    cols = st.columns(4)
    cols[0].metric("Focus price", fmt_price(pos["price"]))
    cols[1].metric(
        f"'{pos['segment']}' median", fmt_price(seg["median"], UNKNOWN),
        delta=fmt_signed_pct(seg["delta_pct"], "n/a") if seg["delta_pct"] is not None else None,
        delta_color="off",
        help=n_caption(seg["n"], seg["n_total"]),
    )
    cols[2].metric(
        f"{pos['brand']} median", fmt_price(brand["median"], UNKNOWN),
        delta=fmt_signed_pct(brand["delta_pct"], "n/a") if brand["delta_pct"] is not None else None,
        delta_color="off",
        help=n_caption(brand["n"], brand["n_total"]),
    )
    cols[3].metric(
        "Segment percentile",
        f"{seg['percentile']:.0f}th" if seg["percentile"] is not None else UNKNOWN,
        help="Percentile among *priced* peers only.",
    )

    a, b = st.columns(2)
    a.caption(f"Segment benchmark: {n_caption(seg['n'], seg['n_total'])}"
              + ("" if seg["reliable"] else "  —  **below the reliability floor**"))
    b.caption(f"Brand benchmark: {n_caption(brand['n'], brand['n_total'])}"
              + ("" if brand["reliable"] else "  —  **below the reliability floor**"))

    if pos["price_available"]:
        st.markdown(
            f"This listing is **{seg['label']}** relative to its segment "
            f"({fmt_price(pos['price'])} vs a median of {fmt_price(seg['median'], UNKNOWN)}, "
            f"{fmt_signed_pct(seg['delta_pct'])}) and **{brand['label']}** relative to the rest "
            f"of {pos['brand']} ({fmt_price(brand['median'], UNKNOWN)} median)."
        )
    else:
        st.info(
            f"This product has **{MISSING_PRICE_TEXT}** in the source data, so no position can "
            f"be computed for it. Its segment's priced peers have a median of "
            f"{fmt_price(seg['median'], UNKNOWN)} ({n_caption(seg['n'], seg['n_total'])})."
        )

    for note in pos["notes"]:
        st.caption(f"• {note}")

    st.divider()
    seg_table = segment_prices()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(segment_median_chart(seg_table, str(focus["segment"])),
                        theme=None, key="seg_median_chart")
    with right:
        seg_prices = df.loc[(df["segment"] == focus["segment"]) & df["price"].notna(), "price"]
        st.plotly_chart(
            segment_distribution_chart(
                seg_prices, str(focus["segment"]),
                float(pos["price"]) if pos["price_available"] else None,
                {"n": seg["n"], "n_total": seg["n_total"], "p25": seg["p25"],
                 "median": seg["median"], "p75": seg["p75"]},
            ),
            theme=None, key="seg_dist_chart",
        )

    st.markdown("**Segment price table**")
    st.dataframe(
        pd.DataFrame({
            "segment": seg_table["segment"],
            "listings": seg_table["n_products"],
            "priced": seg_table["n"],
            "coverage": seg_table["coverage"],
            "median": seg_table["median"].map(lambda v: fmt_price(v, UNKNOWN)),
            "p25 - p75": [
                f"{fmt_price(a, UNKNOWN)} - {fmt_price(b, UNKNOWN)}"
                for a, b in zip(seg_table["p25"], seg_table["p75"])
            ],
            "mean rating": seg_table["mean_rating"].map(lambda v: fmt_num(v, "", 2)),
            "reliable": seg_table["reliable"],
        }),
        hide_index=True,
        column_config={
            "coverage": st.column_config.ProgressColumn("price coverage", min_value=0.0,
                                                        max_value=1.0, format="percent"),
            "reliable": st.column_config.CheckboxColumn("n >= 5"),
        },
        height=min(420, 42 + 36 * len(seg_table)),
    )

    st.markdown(f"**Brand price table** · brands with >= 25 listings, {pos['brand']} highlighted")
    b_table = brand_prices()
    if not b_table.empty:
        b_view = pd.DataFrame({
            "brand": [
                f"* {b}" if b == pos["brand"] else b for b in b_table["brand"]
            ],
            "listings": b_table["n_products"],
            "priced": b_table["n"],
            "coverage": b_table["coverage"],
            "median": b_table["median"].map(lambda v: fmt_price(v, UNKNOWN)),
            "p25 - p75": [
                f"{fmt_price(a, UNKNOWN)} - {fmt_price(c, UNKNOWN)}"
                for a, c in zip(b_table["p25"], b_table["p75"])
            ],
            "renewed": b_table["n_renewed"],
            "mean rating": b_table["mean_rating"].map(lambda v: fmt_num(v, "", 2)),
            "reliable": b_table["reliable"],
        })
        st.dataframe(
            b_view, hide_index=True, height=380,
            column_config={
                "coverage": st.column_config.ProgressColumn("price coverage", min_value=0.0,
                                                            max_value=1.0, format="percent"),
                "reliable": st.column_config.CheckboxColumn("n >= 5"),
            },
        )
        st.caption(
            "'priced' is the denominator behind every median in this table; brands whose "
            "coverage bar is short have medians built on a handful of listings."
        )


# ======================================================================================
# 9. Tab: reviews
# ======================================================================================


def aspect_bar(profile: dict) -> go.Figure:
    """Diverging bar of aspect polarity for one product, labelled with mention counts."""
    aspects = [a for a in ASPECT_NAMES if a in profile["aspects"]]
    values = np.array([profile["aspects"][a]["polarity"] or 0.0 for a in aspects])
    mentions = [profile["aspects"][a]["mentions"] for a in aspects]
    order = np.argsort(values)
    aspects = [aspects[i].replace("_", " / ") for i in order]
    values, mentions = values[order], [mentions[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=values, y=aspects, orientation="h",
            marker=dict(color=diverging_colors(values), line=dict(color=SURFACE, width=2)),
            text=[f"{v:+.2f} ({m} mentions)" for v, m in zip(values, mentions)],
            textposition="outside", textfont=dict(color=INK_2, size=11),
            customdata=np.array(mentions),
            hovertemplate="<b>%{y}</b><br>polarity %{x:+.2f}"
                          "<br>%{customdata:,} review clauses<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=INK_2, width=1))
    fig.update_layout(title="Aspect polarity (-1 negative → +1 positive)")
    fig.update_xaxes(title_text="mean clause polarity", range=[-1.35, 1.35])
    fig.update_yaxes(title_text="")
    return style_fig(fig, height=60 + 38 * max(len(aspects), 3), showlegend=False)


def aspect_heatmap(rows: list[tuple[str, dict]]) -> go.Figure:
    """Focus vs competitors: aspect polarity heatmap on the diverging blue-red scale."""
    labels = [label for label, _ in rows]
    aspects = [a for a in ASPECT_NAMES if any(a in prof["aspects"] for _, prof in rows)]
    z, mentions, text = [], [], []
    for _, prof in rows:
        z.append([prof["aspects"].get(a, {}).get("polarity") for a in aspects])
        mentions.append([prof["aspects"].get(a, {}).get("mentions", 0) for a in aspects])
        text.append([
            "" if prof["aspects"].get(a) is None else f"{prof['aspects'][a]['polarity']:+.2f}"
            for a in aspects
        ])
    fig = go.Figure(
        go.Heatmap(
            z=z, x=[a.replace("_", " / ") for a in aspects], y=labels,
            zmin=-1, zmax=1, colorscale=polarity_colorscale(),
            xgap=2, ygap=2,  # 2px surface gap between cells
            text=text, texttemplate="%{text}", textfont=dict(size=11, color=INK),
            customdata=mentions,
            hovertemplate="<b>%{y}</b><br>%{x}<br>polarity %{z:+.2f}"
                          "<br>%{customdata:,} mentions<extra></extra>",
            colorbar=dict(title=dict(text="polarity", font=dict(color=INK_2)),
                          tickfont=dict(color=INK_2), thickness=12),
        )
    )
    fig.update_layout(title="Aspect polarity: focus product vs competitors "
                            "(blank = the aspect is never mentioned)")
    return style_fig(fig, height=110 + 46 * len(labels), showlegend=False)


def render_snippets(items: list[dict], negative: bool) -> None:
    """Render verbatim review snippets with their aspect, rating and helpfulness."""
    if not items:
        st.caption("No qualifying snippets for this product/aspect.")
        return
    for item in items:
        tone = C_NEG if negative else C_POS
        meta = " · ".join(filter(None, [
            item["aspect"].replace("_", " / "),
            f"polarity {item['polarity']:+.2f}",
            f"{item['rating']:.0f}/5 stars" if item.get("rating") is not None else "",
            f"{item['helpful_vote']} helpful" if item.get("helpful_vote") else "",
            str(item["review_year"]) if item.get("review_year") else "",
            "verified" if item.get("verified_purchase") else "unverified",
        ]))
        st.markdown(
            f"<div style='border-left:3px solid {tone};padding:2px 0 2px 10px;margin:8px 0'>"
            f"<span style='color:{INK}'>“{item['snippet']}”</span><br>"
            f"<span style='color:{INK_2};font-size:0.82em'>{meta}</span></div>",
            unsafe_allow_html=True,
        )


def tab_reviews(focus: pd.Series, comps: pd.DataFrame) -> None:
    """Sentiment summary, aspect breakdown and verbatims for the focus product."""
    st.subheader("Review sentiment")

    asin = str(focus[PRODUCT_KEY])
    profile = product_sentiment(asin)
    if profile is None:
        st.info(
            f"No mined review sentiment for this product. The sentiment pass scored a sample "
            f"of the review corpus, so only **{len(sentiment_ids()):,}** of "
            f"**{len(load_products()):,}** catalogue products have an aspect profile "
            f"(this one has {int(focus['n_reviews']):,} reviews retained in the corpus). "
            "Tick *Only products with mined review sentiment* in the sidebar to browse the "
            "products that do."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Reviews scored", f"{profile['n_reviews_scored']:,}")
    cols[1].metric("Overall polarity", f"{profile['overall_polarity']:+.2f}",
                   help="Mean clause polarity, -1 (negative) to +1 (positive).")
    cols[2].metric("Positive share", fmt_pct(profile["overall_pos_share"]))
    cols[3].metric("Mean review rating", fmt_num(profile["mean_rating"], "", 2))
    st.caption(
        f"All figures below come from n={profile['n_reviews_scored']:,} scored reviews for "
        f"this listing (Amazon reports {int(focus['rating_number']):,} ratings overall)."
    )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(aspect_bar(profile), theme=None, key="aspect_bar")
    with right:
        rows = [(f"FOCUS · {short_title(focus['title'], 34)}", profile)]
        for _, r in comps.iterrows():
            prof = product_sentiment(str(r[PRODUCT_KEY]))
            if prof is not None:
                rows.append((short_title(r["title"], 34), prof))
            if len(rows) >= 7:
                break
        if len(rows) == 1:
            st.info(
                "None of the current competitors have mined review sentiment, so there is "
                "nothing to compare against. Widen the competitor set or pick a more "
                "reviewed focus product."
            )
        else:
            st.plotly_chart(aspect_heatmap(rows), theme=None, key="aspect_heatmap")
            st.caption(
                "Rows: the focus product and the "
                f"{len(rows) - 1} competitor(s) that have scored reviews out of "
                f"{len(comps)} in the competitive set."
            )

    st.divider()
    ctl_a, ctl_b = st.columns([1, 3])
    with ctl_a:
        aspect_choice = st.selectbox(
            "Aspect", ["all aspects"] + ASPECT_NAMES,
            format_func=lambda a: a.replace("_", " / "), key="verbatim_aspect",
        )
        k = st.slider("Snippets per side", 1, 8, 4, key="verbatim_k")
    aspect = None if aspect_choice == "all aspects" else aspect_choice
    with ctl_b:
        st.markdown("**Representative verbatims**")
        st.caption(
            "Ranked by opinion strength x helpful votes, de-duplicated, verified purchases "
            "up-weighted. These are quotes from real reviews of this listing."
        )

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown(f"**What reviewers praise** ({POS_LABEL})")
        render_snippets(verbatims(asin, False, k, aspect), negative=False)
    with neg_col:
        st.markdown(f"**What reviewers complain about** ({NEG_LABEL})")
        render_snippets(verbatims(asin, True, k, aspect), negative=True)


# ======================================================================================
# 10. Tab: ask the agent
# ======================================================================================


def render_evidence(evidence: list[dict], expanded: bool = False) -> None:
    """Render the numbered evidence blocks the model was given."""
    for ev in evidence:
        meta = ev.get("meta", {})
        title = meta.get("title") or meta.get("label") or ev["kind"]
        with st.expander(f"{ev['marker']}  {ev['kind']}  ·  {short_title(title, 80)}",
                         expanded=expanded):
            st.code(ev["text"], language=None, wrap_lines=True)
            asin = meta.get("parent_asin")
            if asin:
                cols = st.columns([1, 4])
                if cols[0].button("Focus this product", key=f"ev_focus_{ev['marker']}_{asin}"):
                    set_focus(str(asin))
                    st.rerun()
                cols[1].caption(
                    f"`{asin}` · {fmt_price(meta.get('price'))}"
                    + (f" · rating {meta['average_rating']:.1f}"
                       if meta.get("average_rating") else "")
                )


def render_answer(result: dict) -> None:
    """Render a RAG answer with its citations and the grounding audit."""
    st.markdown("#### Answer")
    st.markdown(result["answer"])
    if result.get("truncated"):
        st.warning("The answer hit the token cap and is cut off - raise 'max new tokens'.")

    citations = result.get("citations", [])
    st.markdown("#### Citations")
    if not citations:
        st.warning(
            "The model produced no citation markers, so nothing in this answer is traceable "
            "to the retrieved evidence. Treat it as unsupported."
        )
    else:
        for c in citations:
            bits = [f"**{c['marker']}**", f"_{c['kind']}_", short_title(c.get("title", ""), 90)]
            if c.get("price") is not None:
                bits.append(fmt_price(c["price"]))
            elif c["kind"] == "product":
                bits.append(MISSING_PRICE_TEXT)
            if c.get("aspect"):
                bits.append(f"aspect: {c['aspect'].replace('_', ' / ')}")
            if c.get("parent_asin"):
                bits.append(f"`{c['parent_asin']}`")
            st.markdown(" · ".join(bits))

    audit = st.columns(4)
    audit[0].metric("Grounded", "yes" if result["grounded"] else "no",
                    help="No invented markers, no unverified numbers, no cross-attributed reviews.")
    audit[1].metric("Citation rate", fmt_pct(result["citation_rate"]),
                    help="Share of answer sentences carrying at least one valid marker.")
    timings = result.get("timings", {})
    audit[2].metric("Retrieval", f"{timings.get('retrieval_s', 0):.1f}s")
    audit[3].metric("Generation", f"{timings.get('generation_s', 0):.1f}s",
                    delta=f"{timings.get('tokens_per_second', 0):.0f} tok/s",
                    delta_color="off", delta_arrow="off")

    if result["unsupported_markers"]:
        st.error("Invented citation markers: " + ", ".join(result["unsupported_markers"]))
    if result["unverified_numbers"]:
        st.error("Numbers not present in the evidence: " + ", ".join(result["unverified_numbers"]))
    if result["misattributed_reviews"]:
        st.error(
            "Review quotes attributed to the wrong product: "
            + "; ".join(f"{m['review']} belongs to {m['belongs_to']} but was cited with "
                        f"{m['cited_with']}" for m in result["misattributed_reviews"])
        )
    if result["uncited_sentences"]:
        with st.expander(f"{len(result['uncited_sentences'])} sentence(s) without a citation"):
            for s in result["uncited_sentences"]:
                st.markdown(f"- {s}")


def tab_ask() -> None:
    """Natural-language Q&A over the catalogue, rendered with its citations."""
    st.subheader("Ask the agent")
    st.caption(
        "Retrieval is hybrid dense + lexical over the same MiniLM index the matcher uses; "
        "generation is a local 4-bit Qwen2.5-7B-Instruct on the GPU. Every answer is audited "
        "against the evidence it was given."
    )

    example = st.selectbox("Example questions", ["(write my own)"] + EXAMPLE_QUESTIONS,
                           format_func=lambda q: short_title(q, 110), key="ask_example")
    default = "" if example == "(write my own)" else example
    question = st.text_area("Question", value=default, height=90, key="ask_question",
                            placeholder="e.g. Which business laptops under $800 have the "
                                        "fewest keyboard complaints?")

    controls = st.columns([1, 1, 2])
    retrieval_only = controls[0].toggle(
        "Retrieval only (no LLM)", value=False, key="ask_retrieval_only",
        help="Show the evidence the agent would use without loading the language model. "
             "Runs on CPU in about a second.",
    )
    max_new_tokens = controls[1].number_input("Max new tokens", 128, 1200, 600, step=64,
                                              key="ask_max_tokens")
    run = controls[2].button("Ask", type="primary", key="ask_run")

    if not run:
        st.info(
            "Pick or type a question and press **Ask**. The first LLM question loads ~5.6 GB "
            "of 4-bit weights onto the GPU and takes roughly a minute; later questions answer "
            "in a few seconds."
        )
        return
    if not question.strip():
        st.warning("Type a question first.")
        return

    if retrieval_only:
        with st.spinner("Retrieving evidence…"):
            try:
                out = retrieve_only(question.strip())
            except Exception as exc:  # pragma: no cover - defensive UI path
                st.error(f"Retrieval failed: {exc}")
                st.expander("Traceback").code(traceback.format_exc())
                return
        st.success(
            f"Retrieved {len(out['evidence'])} evidence blocks "
            f"(question type: **{out['spec']['question_type']}**). No model was loaded."
        )
        for note in out["notes"]:
            st.caption(f"• {note}")
        render_evidence(out["evidence"], expanded=False)
        with st.expander("Parsed query spec"):
            st.json(out["spec"])
        return

    try:
        with st.spinner("Retrieving, then generating on the GPU…"):
            result = generate_answer(question.strip(), int(max_new_tokens))
    except Exception as exc:  # noqa: BLE001 - any model/CUDA failure must degrade, not crash
        st.error(
            "**The language model could not run, so no answer was generated.** "
            "Everything else in this app - the explorer, the competitive set, the pricing "
            "tables and the review sentiment - is unaffected and still works.\n\n"
            f"Reason: `{type(exc).__name__}: {exc}`"
        )
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.markdown(
            "Common causes: the GPU is busy or out of memory (another process holds the "
            "4-bit weights), CUDA is unavailable, or the model files are not in the local "
            "HuggingFace cache. **You can still use the retrieval-only mode below**, which "
            "runs on CPU and shows the exact evidence the agent would have cited."
        )
        try:
            out = retrieve_only(question.strip())
            st.markdown("#### Retrieved evidence (no generation)")
            for note in out["notes"]:
                st.caption(f"• {note}")
            render_evidence(out["evidence"])
        except Exception as inner:  # pragma: no cover - defensive UI path
            st.warning(f"Retrieval fallback also failed: {inner}")
        return

    render_answer(result)
    st.markdown("#### Evidence the model was given")
    st.caption(
        f"{len(result['evidence'])} blocks, {result['prompt_chars']:,} prompt characters. "
        "[P#] = product card, [R#] = verbatim review snippet, [S#] = market statistic."
    )
    render_evidence(result["evidence"])
    with st.expander("Parsed query spec and retrieval diagnostics"):
        st.json({"query_spec": result["query_spec"], "retrieval": result["retrieval"],
                 "timings": result["timings"]})


# ======================================================================================
# 11. Main
# ======================================================================================


def main() -> None:
    """Assemble the sidebar, the focus header and the five tabs."""
    startup_banner()
    st.title("Laptop competitor intelligence")

    if _ARROW_PREIMPORTED and arrow_pool_backend() != "system":
        st.error(
            "pyarrow was imported before this script could select Arrow's system "
            f"allocator (current backend: `{arrow_pool_backend()}`). In this environment "
            "the mimalloc backend crashes the interpreter on the second script run. "
            "Restart with `ARROW_DEFAULT_MEMORY_POOL=system streamlit run src/app.py`."
        )

    try:
        products = load_products()
    except FileNotFoundError as exc:
        st.error(
            "The processed catalogue is missing. Run the pipeline first:\n\n"
            "`python src/pipeline.py`\n\n"
            f"({exc})"
        )
        return

    filters = sidebar_filters(products)
    results = apply_filters(products, filters)

    if "focus_asin" not in st.session_state:
        st.session_state["focus_asin"] = default_focus(products)
    focus = focus_row(products)

    focus_header(focus)

    st.sidebar.divider()
    st.sidebar.header("Competitor search")
    opts = {
        "k": st.sidebar.slider("Competitors to return", 3, 25, 10, key="comp_k"),
        "apply_guard": st.sidebar.checkbox(
            "Apply segment + price guard", value=True, key="comp_guard",
            help="Off = raw hybrid similarity ranking (the ablation that shows why the "
                 "guard exists).",
        ),
        "exclude_same_brand": st.sidebar.checkbox("Exclude the same brand", value=False,
                                                  key="comp_xbrand"),
        "max_per_brand": st.sidebar.slider("Max listings per brand", 1, 10, 3, key="comp_maxbrand"),
        "include_renewed": st.sidebar.checkbox("Include renewed listings", value=True,
                                               key="comp_renewed"),
    }
    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(products):,} de-duplicated laptop listings · "
        f"{int(products['price'].notna().sum()):,} priced · "
        f"{len(sentiment_ids()):,} with mined review sentiment."
    )

    try:
        comps = competitors_for(str(focus[PRODUCT_KEY]), **opts)
    except Exception as exc:  # noqa: BLE001 - a matcher failure must not blank the app
        st.error(f"Competitor search failed for `{focus[PRODUCT_KEY]}`: {exc}")
        comps = pd.DataFrame(columns=list(products.columns) + ["score", "text_sim", "spec_sim"])

    tabs = st.tabs(["Explorer", "Competitors", "Pricing", "Reviews", "Ask the agent"])
    with tabs[0]:
        tab_explorer(products, results, focus)
    with tabs[1]:
        tab_competitors(focus, comps, opts)
    with tabs[2]:
        tab_pricing(products, focus)
    with tabs[3]:
        tab_reviews(focus, comps)
    with tabs[4]:
        tab_ask()


main()
