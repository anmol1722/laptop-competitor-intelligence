"""Retrieval-augmented competitor-intelligence agent for the laptop corpus.

The agent answers natural-language questions about the 25,841-product catalogue by

1. **parsing** the question into an intent (spec comparison / recommendation /
   competitor lookup / market question) plus hard constraints (brand, segment,
   budget, RAM, GPU, aspect ...) with deterministic rules -- no LLM call, so the
   retrieval stage is fast, debuggable and reproducible;
2. **retrieving** evidence from the artifacts the other modules already built --
   the MiniLM product index from :mod:`matching` (reused, never rebuilt), the
   verbatim aspect snippets from :mod:`sentiment`, and the coverage-aware price
   aggregates from :mod:`pricing`;
3. **grounding** the generation: every evidence item is rendered as a numbered
   block with a citation marker (``[P3]`` product, ``[R7]`` review, ``[S1]``
   market statistic), and the system prompt forbids any claim that is not in the
   context and requires a marker on every factual sentence;
4. **auditing** the answer after generation: markers are resolved back to
   evidence, invented markers are flagged, uncited sentences are counted, and
   every dollar / GB / inch figure in the answer is checked against the context.
   The audit is returned with the answer, so the UI can render citations *and*
   show when the model drifted off its evidence.

The local LLM is ``config.LLM_MODEL`` (Qwen2.5-7B-Instruct, pre-quantized 4-bit)
loaded once behind a process-wide singleton, so importing this module is cheap
and Streamlit reruns do not reload 5.5 GB of weights.

Typical use::

    from rag import answer
    result = answer("Which gaming laptops under $1200 have the best battery reviews?")
    print(result.answer)
    for cite in result.citations:
        print(cite["marker"], cite["kind"], cite["title"])

CLI::

    python src/rag.py --selftest                 # 4 real questions end to end
    python src/rag.py -q "your question"         # one question
    python src/rag.py -q "..." --retrieval-only  # dump the context, no generation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402  (path bootstrap must run first)
    LLM_MODEL,
    PRODUCT_KEY,
    SEGMENTS,
)
import matching  # noqa: E402
import pricing  # noqa: E402
import sentiment  # noqa: E402

LOG = logging.getLogger("rag")

# --------------------------------------------------------------------------------------
# 1. Tunables
# --------------------------------------------------------------------------------------

#: Blend between the MiniLM cosine and the lexical TF-IDF score when ranking products.
#: Dense alone is weak on exact model names ("Predator Helios 300"), lexical alone is
#: weak on paraphrases ("thin and light for travel"), so both are always used.
DENSE_WEIGHT = 0.62

#: Small additive priors (on the 0-1 hybrid score) that steer retrieval towards listings
#: we can actually say something about, instead of one-review ghost listings.
POPULARITY_PRIOR = 0.07      # log-scaled Amazon rating count
REVIEW_EVIDENCE_PRIOR = 0.06  # product has mined review sentiment
RATING_PRIOR = 0.03          # average_rating, only when >= MIN_RATINGS_FOR_PRIOR ratings
MIN_RATINGS_FOR_PRIOR = 20

#: Diversity cap when listing recommendations (otherwise one brand fills the list).
MAX_PER_BRAND = 2

#: Evidence budget.  ~4 chars/token, so 11k chars is roughly 2.8k prompt tokens; on the
#: 8 GB RTX 4060 that prefills in ~3 s and leaves plenty of head-room for the KV cache.
MAX_CONTEXT_CHARS = 11_000
MAX_PRODUCTS = 8
MAX_REVIEWS_PER_PRODUCT = 3
MAX_REVIEW_EVIDENCE = 12
MAX_STATS = 8

#: Generation.  Greedy by default: this is an extraction/synthesis task over supplied
#: context, where sampling only buys hallucination risk.
DEFAULT_MAX_NEW_TOKENS = 600
DEFAULT_TEMPERATURE = 0.0

QUESTION_TYPES = ("spec_compare", "recommend", "competitor", "market", "product_lookup")

# --------------------------------------------------------------------------------------
# 2. Question understanding (deterministic; no LLM in the retrieval path)
# --------------------------------------------------------------------------------------

_COMPARE_RE = re.compile(
    r"\b(compare|versus|vs\.?|difference between|better than|which is better|head to head)\b",
    re.IGNORECASE,
)
_COMPETITOR_RE = re.compile(
    r"\b(competitor|competitors|compete|competing|rival|rivals|alternative|alternatives|"
    r"similar to|comparable to|instead of|up against|who else sells)\b",
    re.IGNORECASE,
)
_RECOMMEND_RE = re.compile(
    r"\b(recommend|recommendation|best|should i buy|which laptop|what laptop|suggest|"
    r"looking for|good for|worth buying|top pick|pick for)\b",
    re.IGNORECASE,
)
_MARKET_RE = re.compile(
    r"\b(market|segment|segments|category|categories|overall|across the|typical|average price|"
    r"median price|price range|how much do|premium|landscape|distribution|brands? charge|"
    r"which brand|price gap|positioning)\b",
    re.IGNORECASE,
)

#: "compare X with Y" / "X vs Y" splitters, longest first so "compared with" wins over "with".
_SPLIT_RE = re.compile(
    r"\s+(?:versus|vs\.?|compares?\s+(?:with|to|against)|compared\s+(?:with|to|against)|"
    r"against|or|and)\s+",
    re.IGNORECASE,
)

_SEGMENT_HINTS: dict[str, tuple[str, ...]] = {
    "gaming": ("gaming", "gamer", "game", "esports", "fps", "rtx", "gtx"),
    "ultrabook": ("ultrabook", "thin and light", "thin-and-light", "lightweight", "portable",
                  "travel", "ultraportable", "premium thin"),
    "business": ("business", "work", "office", "enterprise", "corporate", "professional",
                 "productivity"),
    "budget": ("budget", "cheap", "cheapest", "affordable", "entry level", "entry-level",
               "inexpensive", "low cost", "low-cost"),
    "chromebook": ("chromebook", "chrome os", "chromeos"),
    "mainstream": ("mainstream", "everyday", "general purpose", "family", "all round",
                   "all-round"),
}
assert set(_SEGMENT_HINTS) == set(SEGMENTS), "segment hints drifted from config.SEGMENTS"

_OS_HINTS = {
    "Windows": ("windows", "win 10", "win 11", "windows 10", "windows 11"),
    "macOS": ("macos", "mac os", "macbook", "mac "),
    "ChromeOS": ("chromeos", "chrome os", "chromebook"),
    "Linux": ("linux", "ubuntu"),
}

_PRICE_MAX_RE = re.compile(
    r"(?:under|below|less than|cheaper than|up to|max(?:imum)?|no more than|within|budget of)"
    r"\s*(?:usd|\$)?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?",
    re.IGNORECASE,
)
_PRICE_MIN_RE = re.compile(
    r"(?:over|above|more than|at least|starting at|from)\s*(?:usd|\$)?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?",
    re.IGNORECASE,
)
_PRICE_BETWEEN_RE = re.compile(
    r"between\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_PRICE_AROUND_RE = re.compile(
    r"(?:around|about|near|roughly|circa|~)\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?", re.IGNORECASE
)
_RAM_RE = re.compile(r"\b(\d{1,3})\s*gb\s*(?:of\s*)?(?:ram|memory)\b", re.IGNORECASE)
_STORAGE_RE = re.compile(r"\b(\d{3,4})\s*gb\s*(?:ssd|storage|nvme)\b|\b(\d(?:\.\d)?)\s*tb\b",
                         re.IGNORECASE)
_SCREEN_RE = re.compile(r"\b(1[0-9](?:\.\d)?)\s*(?:\"|inch|in\b|-inch)", re.IGNORECASE)
_CPU_RE = re.compile(r"\b(?:core\s*)?i([3579])\b|\bryzen\s*([3579])\b|\b(celeron|pentium|"
                     r"m1|m2|m3|snapdragon)\b", re.IGNORECASE)
_GPU_RE = re.compile(r"\b(rtx\s*\d{4}|gtx\s*\d{3,4}|discrete gpu|dedicated graphics|"
                     r"dedicated gpu|graphics card)\b", re.IGNORECASE)
_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b")

#: Superlative questions ("the cheapest business laptop with 32 GB") cannot be answered by
#: relevance ranking alone - the honest answer needs the extreme of the *filtered set*, not
#: the extreme of whatever ten listings the embedding surfaced.  Each entry maps a phrase
#: to ``(column, ascending, human label)``.
_SUPERLATIVES: list[tuple[re.Pattern, tuple[str, bool, str]]] = [
    (re.compile(r"\b(cheapest|least expensive|lowest priced?|most affordable)\b", re.I),
     ("price", True, "lowest price")),
    (re.compile(r"\b(most expensive|priciest|highest priced?|dearest)\b", re.I),
     ("price", False, "highest price")),
    (re.compile(r"\b(best|highest|top)[- ]rated\b|\bbest reviewed\b", re.I),
     ("average_rating", False, "highest Amazon rating")),
    (re.compile(r"\b(lightest|least heavy)\b", re.I), ("weight_lb", True, "lowest weight")),
    (re.compile(r"\bmost ram\b|\bmost memory\b", re.I), ("ram_gb", False, "most RAM")),
    (re.compile(r"\b(biggest|largest) (screen|display)\b", re.I),
     ("screen_in", False, "largest screen")),
    (re.compile(r"\bmost storage\b|\bbiggest (ssd|drive|storage)\b", re.I),
     ("storage_gb", False, "most storage")),
]

#: Words that are never a product name on their own, so they must not seed a lookup.
_STOP_PHRASES = {
    "laptop", "laptops", "notebook", "notebooks", "computer", "computers", "machine",
    "machines", "one", "it", "them", "this", "that", "these", "those", "model", "models",
}

#: Connectives that end a product-name phrase ("the ASUS ROG laptop - which one is ...").
_PHRASE_STOP_RE = re.compile(
    r"\b(?:with|vs|versus|against|compares?|compared|comparison|and|or|which|who|that|for|is|"
    r"are|was|were|has|have|do|does|should|under|below|over|about|than|but|instead|to)\b",
    re.IGNORECASE,
)


@dataclass
class QuerySpec:
    """The deterministic parse of a user question that drives retrieval."""

    question: str
    question_type: str
    brands: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)
    os_family: str | None = None
    price_max: float | None = None
    price_min: float | None = None
    ram_min: float | None = None
    storage_min: float | None = None
    screen_in: float | None = None
    cpu_tier: float | None = None
    needs_discrete_gpu: bool = False
    asins: list[str] = field(default_factory=list)
    entity_phrases: list[str] = field(default_factory=list)
    wants_reviews: bool = True
    wants_price_stats: bool = False
    #: ``(column, ascending, label)`` when the question asks for an extreme value.
    sort_by: tuple[str, bool, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (for the UI's 'why did it retrieve this?' panel)."""
        return {
            "question": self.question,
            "question_type": self.question_type,
            "brands": self.brands,
            "segments": self.segments,
            "aspects": self.aspects,
            "os_family": self.os_family,
            "price_max": self.price_max,
            "price_min": self.price_min,
            "ram_min": self.ram_min,
            "storage_min": self.storage_min,
            "screen_in": self.screen_in,
            "cpu_tier": self.cpu_tier,
            "needs_discrete_gpu": self.needs_discrete_gpu,
            "asins": self.asins,
            "entity_phrases": self.entity_phrases,
            "wants_reviews": self.wants_reviews,
            "wants_price_stats": self.wants_price_stats,
            "sort_by": list(self.sort_by) if self.sort_by else None,
        }


def _to_usd(raw: str, k_suffix: str | None) -> float:
    """Parse ``'1,200'`` / ``'1.2'`` + optional ``k`` suffix into dollars."""
    value = float(raw.replace(",", ""))
    if k_suffix:
        value *= 1000.0
    elif value <= 15:            # "under 1.5" almost always means $1,500
        value *= 1000.0
    return value


def classify_question(question: str) -> str:
    """Label a question with one of :data:`QUESTION_TYPES`.

    The order matters: an explicit comparison beats a competitor lookup ("compare the
    Helios with its rivals" is still a comparison), and market wording only wins when no
    specific product intent is present.
    """
    q = question.lower()
    if _COMPARE_RE.search(q):
        return "spec_compare"
    if _COMPETITOR_RE.search(q):
        return "competitor"
    if _MARKET_RE.search(q) and not _RECOMMEND_RE.search(q):
        return "market"
    if _RECOMMEND_RE.search(q):
        return "recommend"
    if _MARKET_RE.search(q):
        return "market"
    return "product_lookup"


def _extract_entity_phrases(question: str, brands: Sequence[str]) -> list[str]:
    """Pull candidate product-name phrases out of a question.

    Anything that looks like a model designation is kept: a known brand plus the words
    that follow it, capitalised runs, and alphanumeric model tokens ("G15", "E5-575").
    The phrases are only *seeds* -- they are resolved against the index by embedding +
    lexical search, so a partial or misspelled name still lands on a real product.
    """
    phrases: list[str] = []
    lowered = question.lower()

    for brand in brands:
        for m in re.finditer(rf"\b{re.escape(brand.lower())}\b", lowered):
            tail = question[m.start(): m.start() + 80]
            # keep the brand plus the words that follow it, stopping at punctuation ...
            chunk = re.split(r"[,.;?!]", tail, maxsplit=1)[0]
            # ... and at the first connective, so "ASUS ROG laptop - which one is" -> "ASUS ROG laptop"
            stop = _PHRASE_STOP_RE.search(chunk, len(brand))
            if stop:
                chunk = chunk[: stop.start()]
            words = chunk.replace("-", " ").split()[:7]
            if len(words) > 1:
                phrases.append(" ".join(words).strip())

    # capitalised / alphanumeric model runs, e.g. "Predator Helios 300", "ROG Strix G15"
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:[ -][A-Z0-9][A-Za-z0-9]*){1,4})\b", question):
        cand = m.group(1).strip()
        if cand.lower() in _STOP_PHRASES or len(cand) < 4:
            continue
        phrases.append(cand)

    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(phrases, key=len, reverse=True):
        key = p.lower()
        if key in _STOP_PHRASES or key in seen:
            continue
        if any(key in s for s in seen):        # already covered by a longer phrase
            continue
        seen.add(key)
        out.append(p)
    return out[:4]


def parse_question(question: str, known_brands: Sequence[str]) -> QuerySpec:
    """Parse a natural-language question into a :class:`QuerySpec`.

    Parameters
    ----------
    question:
        The raw user question.
    known_brands:
        Normalised brand strings from ``products.parquet`` -- matching against the real
        corpus vocabulary avoids inventing filters for brands nobody sells.
    """
    q = question.lower()
    qtype = classify_question(question)

    brands = sorted(
        {b for b in known_brands if b and len(b) > 1 and re.search(rf"\b{re.escape(b.lower())}\b", q)},
        key=len,
        reverse=True,
    )
    segments = [seg for seg, hints in _SEGMENT_HINTS.items() if any(h in q for h in hints)]
    aspects = sentiment.detect_aspects(question)

    os_family = None
    for fam, hints in _OS_HINTS.items():
        if any(h in q for h in hints):
            os_family = fam
            break

    price_min = price_max = None
    m = _PRICE_BETWEEN_RE.search(q)
    if m:
        lo, hi = _to_usd(m.group(1), None), _to_usd(m.group(2), None)
        price_min, price_max = min(lo, hi), max(lo, hi)
    else:
        m = _PRICE_MAX_RE.search(q)
        if m:
            price_max = _to_usd(m.group(1), m.group(2))
        m = _PRICE_MIN_RE.search(q)
        if m:
            price_min = _to_usd(m.group(1), m.group(2))
        if price_min is None and price_max is None:
            m = _PRICE_AROUND_RE.search(q)
            if m:
                mid = _to_usd(m.group(1), m.group(2))
                price_min, price_max = 0.75 * mid, 1.25 * mid

    ram_min = float(_RAM_RE.search(q).group(1)) if _RAM_RE.search(q) else None

    storage_min = None
    m = _STORAGE_RE.search(q)
    if m:
        storage_min = float(m.group(1)) if m.group(1) else float(m.group(2)) * 1024.0

    screen_in = float(_SCREEN_RE.search(q).group(1)) if _SCREEN_RE.search(q) else None

    cpu_tier = None
    m = _CPU_RE.search(q)
    if m and (m.group(1) or m.group(2)):
        cpu_tier = float(m.group(1) or m.group(2))

    needs_gpu = bool(_GPU_RE.search(q)) or "gaming" in segments

    spec = QuerySpec(
        question=question.strip(),
        question_type=qtype,
        brands=brands,
        segments=segments,
        aspects=aspects,
        os_family=os_family,
        price_max=price_max,
        price_min=price_min,
        ram_min=ram_min,
        storage_min=storage_min,
        screen_in=screen_in,
        cpu_tier=cpu_tier,
        needs_discrete_gpu=needs_gpu,
        asins=_ASIN_RE.findall(question),
        entity_phrases=_extract_entity_phrases(question, brands),
        sort_by=next((target for pat, target in _SUPERLATIVES if pat.search(question)), None),
        wants_reviews=qtype != "market" or bool(aspects),
        wants_price_stats=(
            qtype == "market"
            or price_max is not None
            or price_min is not None
            or "price" in q or "cost" in q or "value" in q or "expensive" in q or "cheap" in q
        ),
    )
    return spec


def split_comparison(question: str) -> list[str]:
    """Split a comparison question into its two (or more) sides.

    ``"How does the Acer Predator Helios 300 compare with the ASUS ROG Strix G15?"``
    -> ``["the Acer Predator Helios 300", "the ASUS ROG Strix G15?"]``
    """
    body = re.sub(r"^\s*(how does|how do|compare|which is better[,:]?|what is the difference "
                  r"between|whats the difference between)\s+", "", question.strip(),
                  flags=re.IGNORECASE)
    parts = [p.strip(" ?.,") for p in _SPLIT_RE.split(body) if p.strip(" ?.,")]
    return [p for p in parts if len(p) > 3][:3]


def _phrase_brands(phrase: str, brands: Sequence[str]) -> list[str]:
    """Brands mentioned in a phrase (used to keep comparison sides distinct)."""
    low = phrase.lower()
    return [b for b in brands if re.search(rf"\b{re.escape(b.lower())}\b", low)]


# --------------------------------------------------------------------------------------
# 3. Evidence containers
# --------------------------------------------------------------------------------------


@dataclass
class Evidence:
    """One citable context block.

    Attributes
    ----------
    marker:
        The inline citation token the model must use, e.g. ``"[P3]"``.
    kind:
        ``"product"``, ``"review"`` or ``"stat"``.
    text:
        The rendered block that goes into the prompt.
    meta:
        Structured payload for the UI (asin, title, price, snippet, aspect, ...).
    """

    marker: str
    kind: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {"marker": self.marker, "kind": self.kind, "text": self.text, "meta": self.meta}


@dataclass
class RagAnswer:
    """The agent's response: grounded answer text plus its full audit trail."""

    question: str
    question_type: str
    answer: str
    evidence: list[Evidence]
    citations: list[dict[str, Any]]
    unsupported_markers: list[str]
    uncited_sentences: list[str]
    unverified_numbers: list[str]
    query_spec: dict[str, Any]
    retrieval: dict[str, Any]
    timings: dict[str, float]
    prompt_chars: int
    truncated: bool = False
    misattributed_reviews: list[dict[str, str]] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True when every marker resolves, no number was invented and no review was
        attributed to a product other than the one its block names."""
        return not (self.unsupported_markers or self.unverified_numbers
                    or self.misattributed_reviews)

    @property
    def citation_rate(self) -> float:
        """Share of answer sentences carrying at least one valid citation marker."""
        total = len(_sentences(self.answer))
        if total == 0:
            return 0.0
        return 1.0 - len(self.uncited_sentences) / total

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the Streamlit layer."""
        return {
            "question": self.question,
            "question_type": self.question_type,
            "answer": self.answer,
            "evidence": [e.to_dict() for e in self.evidence],
            "citations": self.citations,
            "unsupported_markers": self.unsupported_markers,
            "uncited_sentences": self.uncited_sentences,
            "unverified_numbers": self.unverified_numbers,
            "misattributed_reviews": self.misattributed_reviews,
            "grounded": self.grounded,
            "truncated": self.truncated,
            "citation_rate": self.citation_rate,
            "query_spec": self.query_spec,
            "retrieval": self.retrieval,
            "timings": self.timings,
            "prompt_chars": self.prompt_chars,
        }


# --------------------------------------------------------------------------------------
# 4. Rendering helpers
# --------------------------------------------------------------------------------------


def _num(value: Any, fmt: str = "{:.0f}", suffix: str = "") -> str | None:
    """Format a possibly-missing numeric cell, returning ``None`` when unusable."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return fmt.format(float(value)) + suffix


_MISSING_TOKENS = {"", "nan", "none", "null", "unknown", "other"}


def _txt(value: Any) -> str:
    """Normalise a categorical cell to a display string, or ``''`` when it is missing.

    ``products.parquet`` encodes "we could not parse this" three different ways depending
    on the column (``NaN``, the literal string ``'Unknown'``, empty string); rendering any
    of them verbatim would put ``RAM=4 GB nan`` in front of the model.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return "" if s.lower() in _MISSING_TOKENS else s


def _title(text: str, max_chars: int = 110) -> str:
    """Trim a marketing-bloated Amazon title down to something quotable."""
    t = re.sub(r"\s+", " ", str(text)).strip()
    return t if len(t) <= max_chars else t[: max_chars - 1].rstrip() + "…"


def render_product(row: pd.Series, marker: str, sent: dict | None = None,
                   extra: Sequence[str] = ()) -> str:
    """Render one product as a compact, fully-attributed context block.

    Missing values are printed as ``unknown`` rather than dropped: the model has to be
    able to *see* that a price or a spec is absent, otherwise it fills the gap itself.
    """
    specs: list[str] = []
    cpu = " ".join(x for x in [_txt(row.get("cpu_brand")), _txt(row.get("cpu_family"))] if x)
    ghz = _num(row.get("cpu_ghz"), "{:.1f}", " GHz")
    specs.append(f"CPU={cpu or 'unknown'}" + (f" {ghz}" if ghz else ""))
    ram = _num(row.get("ram_gb"), "{:.0f}", " GB")
    ram_type = _txt(row.get("ram_type"))
    specs.append("RAM=" + (f"{ram}{' ' + ram_type if ram_type else ''}" if ram else "unknown"))
    stor = _num(row.get("storage_gb"), "{:.0f}", " GB")
    stor_type = _txt(row.get("storage_type"))
    specs.append("Storage=" + (f"{stor}{' ' + stor_type if stor_type else ''}" if stor else "unknown"))
    scr = _num(row.get("screen_in"), "{:.1f}", '"')
    res = None
    if pd.notna(row.get("screen_w")) and pd.notna(row.get("screen_h")):
        res = f"{int(row['screen_w'])}x{int(row['screen_h'])}"
    specs.append("Screen=" + ((scr or "unknown") + (f" {res}" if res else "")))
    gpu_txt = _txt(row.get("gpu_model"))
    gpu_brand = _txt(row.get("gpu_brand"))
    if gpu_brand and gpu_brand.lower() not in gpu_txt.lower():
        gpu_txt = f"{gpu_brand} {gpu_txt}".strip()
    specs.append(f"GPU={gpu_txt or 'unknown'} "
                 f"({'discrete' if bool(row.get('is_discrete_gpu')) else 'integrated'})")
    specs.append(f"OS={_txt(row.get('os_family')) or 'unknown'}")
    wt = _num(row.get("weight_lb"), "{:.1f}", " lb")
    if wt:
        specs.append(f"Weight={wt}")

    price = row.get("price")
    price_txt = (f"${float(price):,.2f} (listed price)" if pd.notna(price)
                 else "not listed in the source data (unknown)")
    rating = _num(row.get("average_rating"), "{:.1f}")
    n_ratings = int(row.get("rating_number") or 0)
    rating_txt = (f"{rating}/5 from {n_ratings:,} Amazon ratings" if rating else "no rating")

    lines = [
        f"{marker} {_title(row.get('title', ''))}",
        f"    brand={_txt(row.get('brand')) or 'unknown'} | "
        f"segment={_txt(row.get('segment')) or 'unknown'} | "
        f"asin={row.get(PRODUCT_KEY)}" + (" | RENEWED/refurbished" if bool(row.get("is_renewed")) else ""),
        "    " + " | ".join(specs),
        f"    price={price_txt} | rating={rating_txt}",
    ]
    if sent:
        pos = sent.get("overall_pos_share")
        bits = []
        if pos is not None:
            bits.append(f"{pos:.0%} of {sent['n_reviews_scored']} analysed reviews are positive")
        ranked = sorted(
            ((a, d) for a, d in sent.get("aspects", {}).items() if d.get("polarity") is not None),
            key=lambda kv: kv[1]["mentions"], reverse=True,
        )[:4]
        for aspect, d in ranked:
            bits.append(f"{aspect} {d['label']} ({d['mentions']} mentions, "
                        f"{d['pos_share']:.0%} positive)")
        if bits:
            lines.append("    review sentiment: " + "; ".join(bits))
    for line in extra:
        lines.append(f"    {line}")
    return "\n".join(lines)


def render_review(snippet: dict, marker: str, product_marker: str, product_title: str) -> str:
    """Render one verbatim review snippet, tied to the product block it supports."""
    meta = [f"about {product_marker} ({_title(product_title, 60)})", f"aspect={snippet['aspect']}",
            f"tone={'negative' if snippet['polarity'] < 0 else 'positive'}"]
    if snippet.get("rating") is not None:
        meta.append(f"{snippet['rating']:.0f}-star review")
    if snippet.get("review_year"):
        meta.append(str(snippet["review_year"]))
    if snippet.get("helpful_vote"):
        meta.append(f"{snippet['helpful_vote']} helpful votes")
    if snippet.get("verified_purchase"):
        meta.append("verified purchase")
    text = re.sub(r"\s+", " ", snippet["snippet"]).strip()
    return f'{marker} customer review — {", ".join(meta)}\n    "{text}"'


def _stat_line(row: pd.Series, group_col: str, label: str) -> str:
    """One row of a pricing table rendered as a sentence with its coverage caveat."""
    med = _num(row.get("median"), "${:,.0f}")
    p25 = _num(row.get("p25"), "${:,.0f}")
    p75 = _num(row.get("p75"), "${:,.0f}")
    txt = (f"{label} '{row[group_col]}': median price {med or 'unknown'} "
           f"(p25 {p25 or 'n/a'} - p75 {p75 or 'n/a'}), "
           f"{int(row['n']):,} priced listings out of {int(row['n_products']):,} "
           f"({float(row['coverage']):.0%} price coverage)")
    mr = _num(row.get("mean_rating"), "{:.2f}")
    if mr:
        txt += f", mean Amazon rating {mr}"
    if not bool(row.get("reliable", True)):
        txt += " [UNRELIABLE: fewer than 5 priced listings]"
    return txt


# --------------------------------------------------------------------------------------
# 5. Retrieval
# --------------------------------------------------------------------------------------


class EvidenceRetriever:
    """Hybrid dense + lexical product search with review and market-stat evidence.

    The dense side *reuses* the MiniLM index that :mod:`matching` already built and
    cached (``product_embeddings.npy``); this class never builds a second embedding
    model -- it only loads the same ``config.EMBED_MODEL`` to encode the incoming
    question, which is a single sentence on CPU (~10 ms) and leaves the GPU free for
    the LLM.

    Examples
    --------
    >>> r = EvidenceRetriever()                                    # doctest: +SKIP
    >>> spec = parse_question("best gaming laptop under $1200", r.brands)  # doctest: +SKIP
    >>> r.search(spec, k=5)[["title", "price", "score"]]           # doctest: +SKIP
    """

    def __init__(self) -> None:
        self.matcher = matching.get_matcher()
        self.products: pd.DataFrame = self.matcher.products
        self.embeddings: np.ndarray = self.matcher.embeddings
        self.brands: list[str] = sorted(
            {str(b) for b in self.products["brand"].dropna().unique() if str(b) != "Unknown"}
        )

        # popularity / quality priors, computed once
        n_rat = pd.to_numeric(self.products["rating_number"], errors="coerce").fillna(0).to_numpy()
        self._pop = np.log1p(n_rat) / max(np.log1p(n_rat).max(), 1.0)
        rating = pd.to_numeric(self.products["average_rating"], errors="coerce").to_numpy()
        self._rating = np.where(np.isnan(rating), 0.0, (rating - 3.0) / 2.0).clip(0.0, 1.0)
        self._rating[n_rat < MIN_RATINGS_FOR_PRIOR] = 0.0

        try:
            sent_ids = set(sentiment.load_product_sentiment().index)
        except FileNotFoundError:          # sentiment pass not run yet
            sent_ids = set()
            LOG.warning("product_sentiment.parquet missing; review evidence disabled")
        self.sentiment_ids = sent_ids
        self._has_sent = self.products[PRODUCT_KEY].isin(sent_ids).to_numpy()

        self._encoder = None
        self._tfidf = None
        self._tfidf_matrix = None

    # -- lazily built query-side machinery ----------------------------------------------

    @property
    def encoder(self):
        """The same sentence-transformer the product index was built with (CPU)."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(matching.EMBED_MODEL, device="cpu")
        return self._encoder

    def _ensure_lexical(self) -> None:
        """Build the TF-IDF title index (~3 s) the first time a lexical score is needed."""
        if self._tfidf is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = (
            self.products["brand"].fillna("").astype(str) + " "
            + self.products["title"].fillna("").astype(str) + " "
            + self.products["segment"].fillna("").astype(str) + " "
            + self.products["gpu_model"].fillna("").astype(str) + " "
            + self.products["cpu_family"].fillna("").astype(str)
        ).tolist()
        vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
                              min_df=2, max_features=300_000, strip_accents="unicode")
        self._tfidf_matrix = vec.fit_transform(corpus)
        self._tfidf = vec

    def encode_query(self, text: str) -> np.ndarray:
        """L2-normalised MiniLM embedding of a query string."""
        vec = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                                  show_progress_bar=False)
        return np.ascontiguousarray(vec.astype(np.float32))[0]

    # -- scoring ------------------------------------------------------------------------

    def hybrid_scores(self, text: str) -> np.ndarray:
        """Dense cosine blended with lexical TF-IDF cosine over the whole catalogue."""
        dense = self.embeddings @ self.encode_query(text)
        self._ensure_lexical()
        qv = self._tfidf.transform([text])
        lex = np.asarray((self._tfidf_matrix @ qv.T).todense()).ravel()
        return DENSE_WEIGHT * dense + (1.0 - DENSE_WEIGHT) * lex

    def constraint_mask(self, spec: QuerySpec, use_price: bool = True) -> np.ndarray:
        """Boolean mask of listings satisfying the question's hard constraints.

        A budget constraint also drops price-less listings: with 69 % of the catalogue
        unpriced we cannot honestly claim an unpriced machine is "under $1,000".
        """
        df = self.products
        mask = np.ones(len(df), dtype=bool)
        if spec.brands:
            mask &= df["brand"].astype(str).str.lower().isin([b.lower() for b in spec.brands]).to_numpy()
        if spec.segments:
            mask &= df["segment"].astype(str).isin(spec.segments).to_numpy()
        # An OS filter is only safe when the question is about one kind of machine:
        # "gaming laptops vs chromebooks" mentions ChromeOS but must not exclude Windows.
        if spec.os_family and len(spec.segments) <= 1:
            mask &= (df["os_family"].astype(str) == spec.os_family).to_numpy()
        if use_price and (spec.price_max is not None or spec.price_min is not None):
            price = pd.to_numeric(df["price"], errors="coerce").to_numpy()
            ok = ~np.isnan(price)
            if spec.price_max is not None:
                ok &= price <= spec.price_max
            if spec.price_min is not None:
                ok &= price >= spec.price_min
            mask &= ok
        if spec.ram_min:
            ram = pd.to_numeric(df["ram_gb"], errors="coerce").to_numpy()
            mask &= ~np.isnan(ram) & (ram >= spec.ram_min)
        if spec.storage_min:
            st = pd.to_numeric(df["storage_gb"], errors="coerce").to_numpy()
            mask &= ~np.isnan(st) & (st >= spec.storage_min)
        if spec.screen_in:
            sc = pd.to_numeric(df["screen_in"], errors="coerce").to_numpy()
            mask &= ~np.isnan(sc) & (np.abs(sc - spec.screen_in) <= 1.0)
        if spec.cpu_tier:
            tier = pd.to_numeric(df["cpu_tier"], errors="coerce").to_numpy()
            mask &= ~np.isnan(tier) & (tier >= spec.cpu_tier)
        if spec.needs_discrete_gpu and spec.question_type in ("recommend", "product_lookup"):
            mask &= df["is_discrete_gpu"].to_numpy(dtype=bool)
        return mask

    def search(self, spec: QuerySpec, k: int = 6, text: str | None = None,
               diversify: bool = True, require_reviews: bool = False) -> pd.DataFrame:
        """Rank catalogue listings for a parsed question.

        Falls back through progressively weaker constraint sets (full -> without price ->
        none) rather than returning nothing, and records which relaxation was used in
        ``frame.attrs['relaxation']`` so the answer can disclose it.
        """
        score = self.hybrid_scores(text or spec.question)
        score = (score
                 + POPULARITY_PRIOR * self._pop
                 + RATING_PRIOR * self._rating
                 + REVIEW_EVIDENCE_PRIOR * self._has_sent.astype(np.float32))

        attempts = [("all constraints", self.constraint_mask(spec, use_price=True))]
        if spec.price_max is not None or spec.price_min is not None:
            attempts.append(("price constraint dropped", self.constraint_mask(spec, use_price=False)))
        attempts.append(("no constraints", np.ones(len(self.products), dtype=bool)))

        chosen_mask, relaxation = attempts[-1][1], attempts[-1][0]
        for label, mask in attempts:
            m = mask & self._has_sent if require_reviews else mask
            if m.sum() >= max(k, 3):
                chosen_mask, relaxation = m, label
                break

        idx = np.flatnonzero(chosen_mask)
        if idx.size == 0:
            out = self.products.head(0).copy()
            out["score"] = []
            out.attrs["relaxation"] = "no candidates"
            out.attrs["n_candidates"] = 0
            out.attrs["ordering"] = "relevance"
            return out

        ordering = "relevance"
        if spec.sort_by is not None and spec.sort_by[0] in self.products.columns:
            # "the cheapest business laptop with 32 GB" is a question about the extreme of
            # the *filtered set*, not about the top of a relevance list, so the whole
            # candidate set is ordered by the requested column and the switch is disclosed.
            col, ascending, label = spec.sort_by
            values = pd.to_numeric(self.products[col], errors="coerce").to_numpy()
            usable = idx[~np.isnan(values[idx])]
            if usable.size >= 1:
                keys = values[usable]
                order = usable[np.argsort(keys if ascending else -keys, kind="stable")]
                ordering = f"sorted by {label} over all {usable.size:,} matching listings"
            else:
                order = idx[np.argsort(-score[idx])]
                ordering = f"relevance ({label} is unknown for every candidate)"
        else:
            order = idx[np.argsort(-score[idx])]
        picked: list[int] = []
        counts: dict[str, int] = {}
        brands = self.products["brand"].astype(str).to_numpy()
        for i in order[: max(400, k * 40)]:
            if diversify and counts.get(brands[i], 0) >= MAX_PER_BRAND:
                continue
            counts[brands[i]] = counts.get(brands[i], 0) + 1
            picked.append(int(i))
            if len(picked) >= k:
                break
        if len(picked) < k:                     # not enough distinct brands -> relax the cap
            for i in order:
                if int(i) not in picked:
                    picked.append(int(i))
                if len(picked) >= k:
                    break

        out = self.products.iloc[picked].copy()
        out["score"] = score[picked]
        out.attrs["relaxation"] = relaxation
        out.attrs["n_candidates"] = int(idx.size)
        out.attrs["ordering"] = ordering
        return out.reset_index(drop=True)

    def resolve_product(self, phrase: str, spec: QuerySpec) -> pd.Series | None:
        """Resolve a free-text product mention to a single catalogue row.

        Used for comparison and competitor questions, where the anchor must be a real
        listing.  Returns ``None`` when nothing scores above a floor, which the agent
        surfaces as "that product is not in the catalogue" instead of guessing.
        """
        if not phrase.strip():
            return None
        raw = self.hybrid_scores(phrase) + POPULARITY_PRIOR * self._pop \
            + REVIEW_EVIDENCE_PRIOR * self._has_sent.astype(np.float32)
        mask = np.ones(len(self.products), dtype=bool)
        brand_hits = [b for b in self.brands
                      if re.search(rf"\b{re.escape(b.lower())}\b", phrase.lower())]
        if brand_hits:
            mask &= self.products["brand"].astype(str).isin(brand_hits).to_numpy()
        if spec.segments:
            seg_mask = self.products["segment"].astype(str).isin(spec.segments).to_numpy()
            if (mask & seg_mask).sum() >= 1:
                mask &= seg_mask
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            idx = np.arange(len(self.products))
        best = idx[int(np.argmax(raw[idx]))]
        if raw[best] < 0.20:
            return None
        row = self.products.iloc[best].copy()
        row["score"] = float(raw[best])
        return row

    # -- review + market evidence -------------------------------------------------------

    def review_evidence(self, parent_asin: str, aspects: Sequence[str], k: int = 3) -> list[dict]:
        """Balanced praise/complaint snippets for one product.

        When the question named aspects (battery, thermals, ...) those are queried first
        so the evidence actually addresses what was asked; the remainder is filled with
        the product's strongest overall opinions.
        """
        if parent_asin not in self.sentiment_ids:
            return []
        out: list[dict] = []
        seen: set[str] = set()

        def _take(items: list[dict], limit: int) -> None:
            for it in items:
                key = it["snippet"][:80].lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(it)
                if len([o for o in out]) >= limit:
                    return

        wanted = [a for a in aspects if a in sentiment.ASPECT_NAMES]
        for aspect in wanted:
            _take(sentiment.top_praises(parent_asin, k=1, aspect=aspect), k)
            _take(sentiment.top_complaints(parent_asin, k=1, aspect=aspect), k)
        if len(out) < k:
            _take(sentiment.top_praises(parent_asin, k=2), k)
        if len(out) < k:
            _take(sentiment.top_complaints(parent_asin, k=2), k)
        return out[:k]

    def market_stats(self, spec: QuerySpec, rows: pd.DataFrame | None = None) -> list[str]:
        """Coverage-aware market statistics relevant to the question.

        Everything here comes from :mod:`pricing`, so the "n priced of n listings" caveat
        travels with each number into the prompt.
        """
        segs = list(spec.segments)
        if not segs and rows is not None and len(rows):
            segs = list(pd.unique(rows["segment"].astype(str)))[:3]

        segment_lines: list[str] = []
        seg_tbl = pricing.segment_price_table()
        sel = seg_tbl[seg_tbl["segment"].isin(segs)] if segs else seg_tbl
        for _, r in sel.iterrows():
            segment_lines.append(_stat_line(r, "segment", "Segment"))

        # The GPU premium is a *derived* figure the model cannot compute correctly from
        # segment medians (it tried, and invented one), so it is ranked above brand rows.
        gpu_lines: list[str] = []
        ql = spec.question.lower()
        if spec.needs_discrete_gpu or "gpu" in ql or "graphics" in ql:
            prem = pricing.discrete_gpu_premium()
            keep = prem[prem["segment"].isin(segs + ["ALL"])] if segs else prem[prem["segment"] == "ALL"]
            for _, r in keep.head(3).iterrows():
                pu, pp = _num(r.get("premium_usd"), "${:,.0f}"), _num(r.get("premium_pct"), "{:.0f}%")
                if pu is None:
                    continue
                gpu_lines.append(
                    f"Discrete-GPU price premium in '{r['segment']}': {pu} ({pp}) - median "
                    f"{_num(r.get('median_discrete'), '${:,.0f}')} across {int(r['n_discrete'])} "
                    f"priced discrete-GPU listings vs {_num(r.get('median_integrated'), '${:,.0f}')} "
                    f"across {int(r['n_integrated'])} priced integrated-GPU listings"
                    + ("" if bool(r.get("reliable")) else " [UNRELIABLE: small sample]")
                )

        brand_lines: list[str] = []
        if spec.brands or spec.question_type == "market":
            brand_tbl = pricing.brand_price_table(min_products=25, top=40)
            bsel = (brand_tbl[brand_tbl["brand"].isin(spec.brands)] if spec.brands
                    else brand_tbl.nlargest(4, "n_products"))
            for _, r in bsel.head(4).iterrows():
                brand_lines.append(_stat_line(r, "brand", "Brand"))

        return (segment_lines + gpu_lines + brand_lines)[:MAX_STATS]


@lru_cache(maxsize=1)
def get_retriever() -> EvidenceRetriever:
    """Process-wide cached retriever (matcher + priors + lexical index)."""
    return EvidenceRetriever()


# --------------------------------------------------------------------------------------
# 6. Local LLM singleton
# --------------------------------------------------------------------------------------


def _disable_triton_native_ops() -> str:
    """Route torch's Triton-backed native op overrides back to ATen.

    torch 2.13 dispatches a few eager ops (e.g. ``bmm_outer_product``, used by Qwen's
    RoPE) to Triton kernels.  Triton's CUDA driver shim is compiled on first use with
    ``gcc`` and ``Python.h``; this machine has no Python development headers, so that
    build fails and *any* generation raises ``CalledProcessError``.  Deregistering the
    Triton DSL makes those ops fall back to the ATen kernels, which cost a few percent
    of decode speed and always work.  Returns a short status string for logging.
    """
    try:
        import torch._native as torch_native

        torch_native.registry.deregister_op_overrides(disable_dsl_names="triton")
        return "triton native-op overrides disabled (ATen fallback)"
    except Exception as exc:                    # pragma: no cover - future torch versions
        return f"triton override deregistration skipped ({exc})"


class LocalLLM:
    """Lazy singleton wrapper around the pre-quantized 4-bit Qwen2.5-7B-Instruct.

    Weights are loaded on the first :meth:`chat` call, never at import time, so
    ``import rag`` stays sub-second and Streamlit reruns reuse the resident model.
    The 4-bit checkpoint carries its own ``BitsAndBytesConfig``; no quantization config
    is passed at load time, only the compute dtype and the SDPA attention kernel.
    Measured on the RTX 4060 Laptop 8 GB: 5.6 GB VRAM resident, ~31 tokens/s decode.
    """

    def __init__(self, model_name: str = LLM_MODEL, device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_seconds: float | None = None
        self.notes: list[str] = []

    def load(self) -> None:
        """Materialise tokenizer + weights (idempotent)."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.notes.append(_disable_triton_native_ops())
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; a 4-bit 7B model is unusably slow on CPU. "
                "Fix the CUDA install or run rag.py with --retrieval-only."
            )
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,          # compute dtype; weights stay 4-bit NF4
            device_map={"": 0} if self.device == "cuda" else self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        # the checkpoint ships max_length=32768, which shadows max_new_tokens warnings
        self.model.generation_config.max_length = None
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.load_seconds = time.time() - t0
        LOG.info("loaded %s in %.1fs (%s)", self.model_name, self.load_seconds,
                 "; ".join(self.notes))

    def chat(self, messages: Sequence[dict[str, str]],
             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
             temperature: float = DEFAULT_TEMPERATURE) -> tuple[str, dict[str, float]]:
        """Run one chat completion and return ``(text, stats)``.

        ``temperature <= 0`` selects greedy decoding, which is the default: the task is
        synthesis over supplied evidence, where sampling only adds hallucination risk.
        """
        self.load()
        import torch

        enc = self.tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        n_prompt = int(enc["input_ids"].shape[1])

        kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            kwargs.update(do_sample=False)

        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        dt = time.time() - t0
        new_tokens = int(out.shape[1]) - n_prompt
        text = self.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        return text, {
            "prompt_tokens": float(n_prompt),
            "new_tokens": float(new_tokens),
            "seconds": dt,
            "tokens_per_second": new_tokens / dt if dt > 0 else 0.0,
            # hit the cap without emitting a stop token -> the answer is cut off
            "truncated": float(int(out[0, -1]) not in eos_ids),
        }


@lru_cache(maxsize=1)
def get_llm(device: str = "cuda") -> LocalLLM:
    """Process-wide cached LLM handle (weights still load lazily on first use)."""
    return LocalLLM(device=device)


# --------------------------------------------------------------------------------------
# 7. Prompting
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a competitor-intelligence analyst for the laptop market. \
You work ONLY from the CONTEXT block you are given, which was retrieved from an Amazon \
laptop catalogue with mined customer-review sentiment and price statistics.

Hard rules:
1. Use ONLY facts that appear in the CONTEXT. Never add specs, prices, models, brands or \
review opinions from your own knowledge, even if you are confident they are true.
2. Cite every factual sentence with the marker(s) of the evidence it came from, written \
inline in square brackets: [P2] for a product block, [R5] for a customer review, [S1] for \
a market statistic. Multiple markers are fine: [P1][R3].
3. Never invent a marker. Only use markers that appear in the CONTEXT.
4. If the CONTEXT does not contain what is needed, say plainly what is missing, e.g. \
"The retrieved data does not include the battery life of [P3]." Do not guess, and do not \
fall back on general knowledge.
5. Price is missing for most listings. If a block says the price is not listed, say the \
price is unknown - never estimate it. Quote every number exactly as it appears.
6. Review sentiment percentages describe only the reviews that were analysed; report them \
as such and keep the counts attached.
7. Do not derive new numbers by arithmetic on the context (differences, premiums, averages, \
percentages). If a derived figure is not stated in the CONTEXT, say it is not available.
8. Every sentence that states a fact must carry its marker(s); a bullet with several \
sentences needs a marker on each of them.
9. Each customer-review block names the ONE product it is about ("about [P3]"). Never use \
a review as evidence for any other product.
10. Be concise and decision-useful: at most 6 short bullets or 200 words, no preamble, no \
restating of these instructions, and finish your last sentence."""

_TASK_HINTS = {
    "spec_compare": (
        "Compare the products in at most 5 bullets, covering only the specs that matter for "
        "the question, then close with one sentence naming the better buy and why. Call out "
        "any spec that is unknown for either side instead of skipping it, and never state a "
        "spec (VRAM, refresh rate, battery hours) that no block lists."
    ),
    "recommend": (
        "Recommend at most 3 of the retrieved products, best first. For each, give the one "
        "or two reasons it fits the request, grounded in its specs, price and review "
        "evidence, plus the strongest caveat visible in the reviews."
    ),
    "competitor": (
        "Identify the closest competitors to the anchor product from the retrieved "
        "candidates, and for each say how it is positioned against the anchor on specs, "
        "price and review sentiment."
    ),
    "market": (
        "Answer at the market level using the statistics blocks. Always report the sample "
        "size and price coverage behind any figure you quote, and flag any statistic marked "
        "UNRELIABLE."
    ),
    "product_lookup": (
        "Answer directly from the retrieved product and review evidence, and say what the "
        "data does not cover."
    ),
}


#: A style exemplar (not data).  A 7B model reliably puts one marker at the end of a whole
#: bullet unless it is shown line-by-line citation; with this exemplar the measured
#: sentence-level citation rate roughly doubles.
CITATION_STYLE_EXAMPLE = """CITATION STYLE - copy this shape exactly (the products below are \
invented placeholders, not data):
- Example Brand X15 at $429.00 [P2] sits below the mainstream median of $649.99 [S1].
  Buyers praise the keyboard [R4], but two reviewers report loud fans under load [R5].
- Example Brand Y14 costs $612.00 [P3] and its price is 43% above the [P2] listing [P3].
  The retrieved data does not list its battery life [P3].
Every line carries at least one marker."""


def build_prompt(question: str, question_type: str, evidence: Sequence[Evidence],
                 notes: Sequence[str] = ()) -> list[dict[str, str]]:
    """Assemble the chat messages: grounding system prompt + context + task."""
    blocks = "\n\n".join(e.text for e in evidence)
    markers = ", ".join(e.marker for e in evidence)
    note_txt = ("\nRETRIEVAL NOTES (mention them if they affect the answer):\n"
                + "\n".join(f"- {n}" for n in notes)) if notes else ""
    user = (
        f"CONTEXT (the only facts you may use; {len(evidence)} blocks, markers: {markers})\n"
        f"{'=' * 78}\n{blocks}\n{'=' * 78}\n{note_txt}\n\n"
        f"QUESTION: {question}\n\n"
        f"TASK: {_TASK_HINTS.get(question_type, _TASK_HINTS['product_lookup'])}\n"
        f"{CITATION_STYLE_EXAMPLE}\n"
        f"Now answer the question. If the context cannot answer part of it, say so explicitly."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------------------
# 8. Post-hoc grounding audit
# --------------------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\[([PRS]\d+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n+")
#: ``-`` is captured but dropped: the context prints a negative premium as ``$-74`` while
#: the model writes "cheaper by $74", and those are the same fact.
_MONEY_RE = re.compile(r"\$\s?-?([\d,]+(?:\.\d{1,2})?)")
_SPEC_NUM_RE = re.compile(r"\b(\d{1,4}(?:\.\d)?)\s?(gb|tb|ghz|inch|\"|hz|lb)\b", re.IGNORECASE)
_PCT_RE = re.compile(r"\b(\d{1,3}(?:\.\d)?)\s?%")


def _sentences(text: str) -> list[str]:
    """Split answer text into sentences/bullets for the citation audit."""
    parts = [p.strip(" -*•\t") for p in _SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if len(p) > 25]


def _canon_money(raw: str) -> float:
    """Parse a captured money string (``'1,499.99'``) into a float for comparison."""
    return float(raw.replace(",", "").rstrip("."))


def audit_answer(answer: str, evidence: Sequence[Evidence]) -> dict[str, Any]:
    """Check an answer against the context it was supposed to be grounded in.

    Four independent checks, all cheap and all reported to the caller:

    * **marker resolution** - every ``[P#]/[R#]/[S#]`` must exist in the evidence;
      invented markers are returned in ``unsupported_markers``;
    * **citation coverage** - sentences longer than a clause with no marker at all are
      returned in ``uncited_sentences``;
    * **number provenance** - every dollar amount, every ``GB/TB/GHz/inch/Hz/lb`` figure
      and every percentage in the answer must appear in the context; the rest are
      returned in ``unverified_numbers``.  This catches the classic failure where the
      model keeps the citation but rounds or invents the value next to it.
    * **cross-attribution** - a review block belongs to exactly one product.  When a
      sentence cites review ``[R8]`` (which the context ties to ``[P3]``) next to a
      *different* product marker, that is an attribution hallucination and is returned in
      ``misattributed_reviews``.
    """
    by_marker = {e.marker.strip("[]"): e for e in evidence}
    context = "\n".join(e.text for e in evidence)

    used, unsupported = [], []
    for m in dict.fromkeys(_MARKER_RE.findall(answer or "")):
        (used if m in by_marker else unsupported).append(m)

    citations = []
    for m in used:
        ev = by_marker[m]
        citations.append({
            "marker": f"[{m}]",
            "kind": ev.kind,
            "title": ev.meta.get("title") or ev.meta.get("label") or ev.kind,
            **{k: v for k, v in ev.meta.items() if k != "title"},
        })

    uncited = [s for s in _sentences(answer) if not _MARKER_RE.search(s)]

    ctx_money = {_canon_money(x) for x in _MONEY_RE.findall(context)}
    ctx_specs = {(float(v), u.lower().replace('"', "inch")) for v, u in _SPEC_NUM_RE.findall(context)}
    ctx_pct = {float(p) for p in _PCT_RE.findall(context)}
    unverified: list[str] = []
    for raw in _MONEY_RE.findall(answer or ""):
        val = _canon_money(raw)
        if not any(abs(val - c) <= max(1.0, 0.01 * c) for c in ctx_money):
            unverified.append(f"${raw}")
    for val, unit in _SPEC_NUM_RE.findall(answer or ""):
        key = (float(val), unit.lower().replace('"', "inch"))
        if key not in ctx_specs:
            unverified.append(f"{val} {unit}")
    for raw in _PCT_RE.findall(answer or ""):
        val = float(raw)
        if not any(abs(val - c) <= 1.0 for c in ctx_pct):
            unverified.append(f"{raw}%")

    # a review may only be quoted alongside the product its block names
    owner = {e.marker.strip("[]"): e.meta.get("product_marker") for e in evidence
             if e.kind == "review"}
    misattributed: list[dict[str, str]] = []
    for sent in _sentences(answer):
        marks = _MARKER_RE.findall(sent)
        products = {m for m in marks if m.startswith("P")}
        if not products:
            continue
        for m in marks:
            home = (owner.get(m) or "").strip("[]")
            if home and home not in products:
                misattributed.append({
                    "review": f"[{m}]", "belongs_to": f"[{home}]",
                    "cited_with": ", ".join(f"[{p}]" for p in sorted(products)),
                    "sentence": sent[:160],
                })

    return {
        "citations": citations,
        "unsupported_markers": unsupported,
        "uncited_sentences": uncited,
        "unverified_numbers": sorted(set(unverified)),
        "misattributed_reviews": misattributed,
    }


# --------------------------------------------------------------------------------------
# 9. The agent
# --------------------------------------------------------------------------------------

_NO_EVIDENCE_ANSWER = (
    "I could not find anything in the retrieved catalogue that matches this question, so I "
    "cannot answer it from the data. Try naming a brand, a segment "
    f"({', '.join(SEGMENTS)}) or a budget."
)


class RagAgent:
    """Retrieval-augmented question answering over the laptop competitive set.

    The agent is stateless per question: :meth:`retrieve` builds the evidence set,
    :meth:`answer` generates on top of it and audits the result.  Both are public so the
    Streamlit app can show the retrieved evidence while the model is still decoding, and
    so retrieval can be tested without a GPU.

    Examples
    --------
    >>> agent = get_agent()                                            # doctest: +SKIP
    >>> res = agent.answer("Who competes with the Acer Predator Helios 300?")  # doctest: +SKIP
    >>> res.citation_rate, res.grounded                                # doctest: +SKIP
    """

    def __init__(self, retriever: EvidenceRetriever | None = None,
                 llm: LocalLLM | None = None) -> None:
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm()

    # -- retrieval ----------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[QuerySpec, list[Evidence], list[str], dict[str, Any]]:
        """Build the evidence set for a question.

        Returns ``(spec, evidence, notes, stats)`` where ``notes`` are retrieval caveats
        that get shown to the model (e.g. a dropped price filter) and ``stats`` is a
        diagnostics payload for the UI.
        """
        spec = parse_question(question, self.retriever.brands)
        notes: list[str] = []
        products: pd.DataFrame
        anchors: list[str] = []

        if spec.question_type == "spec_compare":
            products, anchors, notes = self._retrieve_comparison(spec, notes)
        elif spec.question_type == "competitor":
            products, anchors, notes = self._retrieve_competitors(spec, notes)
        elif spec.question_type == "market":
            products = self._retrieve_market_examples(spec)
        else:
            products = self.retriever.search(
                spec, k=MAX_PRODUCTS if spec.question_type == "recommend" else 5,
                require_reviews=spec.wants_reviews and bool(spec.aspects),
            )

        ordering = products.attrs.get("ordering", "relevance")
        if ordering.startswith("sorted by"):
            notes.append(
                f"The product blocks are {ordering} that satisfy the parsed filters, in order, "
                f"so [P1] really is the extreme one among them. Any listing whose value is "
                f"missing in the source data was excluded from that ranking."
            )
        elif spec.sort_by is not None:
            notes.append(
                f"The question asks for the {spec.sort_by[2]}, but the retrieved blocks are "
                f"ordered by relevance, not by that value - do not claim a superlative."
            )

        relaxation = products.attrs.get("relaxation")
        if relaxation and relaxation != "all constraints":
            notes.append(
                f"Not enough listings satisfied every constraint, so retrieval fell back: "
                f"{relaxation}. Say so if it changes the answer."
            )
        if (spec.price_max or spec.price_min) and relaxation == "all constraints":
            notes.append(
                "Only listings with a real price in the source data were eligible for the "
                "budget filter; 69% of the catalogue has no price and was therefore excluded."
            )

        evidence = self._assemble(spec, products, anchors, notes)
        stats = {
            "n_products_retrieved": int(len(products)),
            "n_candidates": int(products.attrs.get("n_candidates", 0)),
            "relaxation": relaxation,
            "ordering": ordering,
            "anchors": anchors,
            "n_evidence": len(evidence),
            "kinds": {k: sum(1 for e in evidence if e.kind == k) for k in ("product", "review", "stat")},
        }
        return spec, evidence, notes, stats

    def _retrieve_comparison(self, spec: QuerySpec, notes: list[str]
                             ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Resolve both sides of a comparison to real listings.

        Brand-anchored entity phrases win over a naive "X vs Y" split, because real
        questions rarely split cleanly ("compare the Acer Predator with the ASUS ROG -
        which one is the better buy?" has no separator the splitter can trust).
        """
        brands = self.retriever.brands
        anchored = [p for p in spec.entity_phrases if _phrase_brands(p, brands)]
        distinct = {tuple(_phrase_brands(p, brands)) for p in anchored}
        if len(anchored) >= 2 and len(distinct) >= 2:
            sides = anchored
        else:
            split = [p for p in split_comparison(spec.question)
                     if _phrase_brands(p, brands) or re.search(r"[A-Z][a-z]+\s+[A-Z0-9]", p)]
            sides = split if len(split) >= 2 else (spec.entity_phrases or split)
        rows: list[pd.Series] = []
        seen: set[str] = set()
        for side in sides[:3]:
            row = self.retriever.resolve_product(side, spec)
            if row is None:
                notes.append(f"No catalogue listing matched '{side}'.")
                continue
            if row[PRODUCT_KEY] in seen:
                continue
            seen.add(row[PRODUCT_KEY])
            rows.append(row)
        if len(rows) < 2:
            extra = self.retriever.search(spec, k=4)
            for _, r in extra.iterrows():
                if r[PRODUCT_KEY] not in seen:
                    seen.add(r[PRODUCT_KEY])
                    rows.append(r)
                if len(rows) >= 2:
                    break
            notes.append("Fewer than two named products could be resolved; the closest "
                         "catalogue matches were used instead.")
        df = pd.DataFrame(rows).reset_index(drop=True)
        df.attrs["relaxation"] = "all constraints"
        df.attrs["n_candidates"] = len(df)
        return df, [str(r[PRODUCT_KEY]) for r in rows], notes

    def _retrieve_market_examples(self, spec: QuerySpec) -> pd.DataFrame:
        """Illustrative listings for a market question.

        When the question contrasts several segments ("gaming vs chromebook") the sample
        is drawn per segment, so the model can see a concrete example of each side rather
        than four listings from whichever segment the embedding happened to favour.
        """
        if len(spec.segments) <= 1:
            return self.retriever.search(spec, k=4, require_reviews=False)
        frames = []
        for seg in spec.segments[:3]:
            sub = replace(spec, segments=[seg])
            frames.append(self.retriever.search(sub, k=2, require_reviews=False))
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[PRODUCT_KEY])
        out.attrs["relaxation"] = "all constraints"
        out.attrs["n_candidates"] = sum(int(f.attrs.get("n_candidates", 0)) for f in frames)
        return out.reset_index(drop=True)

    def _retrieve_competitors(self, spec: QuerySpec, notes: list[str]
                              ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Resolve the anchor product and pull its competitive set from :mod:`matching`."""
        phrase = spec.entity_phrases[0] if spec.entity_phrases else spec.question
        anchor = self.retriever.resolve_product(phrase, spec)
        if anchor is None:
            notes.append("No anchor product could be resolved; falling back to a plain search.")
            df = self.retriever.search(spec, k=6)
            return df, [], notes

        asin = str(anchor[PRODUCT_KEY])
        try:
            comp = self.retriever.matcher.find_competitors(asin, k=5, max_per_brand=2)
        except KeyError:                       # pragma: no cover - anchor comes from the frame
            comp = self.retriever.products.head(0)
        frame = pd.concat([anchor.to_frame().T, comp], ignore_index=True)
        frame = frame.drop_duplicates(subset=[PRODUCT_KEY]).reset_index(drop=True)
        frame.attrs["relaxation"] = "all constraints"
        frame.attrs["n_candidates"] = len(frame)
        notes.append(
            f"The competitive set was produced by the hybrid text+spec matcher with its "
            f"segment and price-band guard, anchored on {asin}."
        )
        return frame, [asin], notes

    def _assemble(self, spec: QuerySpec, products: pd.DataFrame, anchors: Sequence[str],
                  notes: list[str]) -> list[Evidence]:
        """Render products, reviews and market stats into a budgeted evidence list.

        Each product's review snippets are emitted *immediately after* that product's
        block rather than in a separate section at the end.  Measured effect: with the
        reviews pooled at the end the model attributed snippets to the wrong product
        (``[R10]`` about ``[P5]`` quoted as evidence for ``[P4]``); interleaving them
        removed those cross-attributions from the audit.
        """
        evidence: list[Evidence] = []
        budget = MAX_CONTEXT_CHARS
        review_count = 0

        # Price gaps against the anchor are computed here, not left to the model: rule 7 of
        # the system prompt forbids it from doing arithmetic, and 7B models get the sign of
        # "$345 vs $349" wrong often enough to matter.
        anchor_price = None
        if anchors:
            anchor_rows = products[products[PRODUCT_KEY] == anchors[0]]
            if len(anchor_rows) and pd.notna(anchor_rows.iloc[0].get("price")):
                anchor_price = float(anchor_rows.iloc[0]["price"])

        for i, (_, row) in enumerate(products.head(MAX_PRODUCTS).iterrows(), start=1):
            asin = str(row[PRODUCT_KEY])
            marker = f"[P{i}]"
            sent = sentiment.get_product_sentiment(asin) if asin in self.retriever.sentiment_ids else None
            extra: list[str] = []
            if anchors and asin == anchors[0] and spec.question_type == "competitor":
                extra.append("role: ANCHOR product for this competitor lookup")
            elif anchors and asin in anchors:
                extra.append("role: named in the question")
            elif anchor_price and pd.notna(row.get("price")):
                delta = float(row["price"]) - anchor_price
                word = "cheaper than" if delta < 0 else ("the same price as" if delta == 0
                                                         else "more expensive than")
                extra.append(f"price vs the anchor [P1]: ${abs(delta):,.2f} {word} the anchor "
                             f"({abs(delta) / anchor_price:.0%} difference)")
            if spec.wants_price_stats and pd.notna(row.get("price")):
                try:
                    pos = pricing.price_position(asin)
                    seg = pos["vs_segment"]
                    if seg.get("median"):
                        # the delta is spelled out so the model never has to compute it
                        delta = (f", {abs(seg['delta_pct']):.0f}% "
                                 f"{'above' if seg['delta_pct'] >= 0 else 'below'} that median"
                                 if seg.get("delta_pct") is not None else "")
                        extra.append(
                            f"price position: {seg['label']} vs the '{pos['segment']}' segment "
                            f"median ${seg['median']:,.0f}{delta} (n={seg['n']} priced peers"
                            + (f", percentile {seg['percentile']:.0f} of 100"
                               if seg.get("percentile") is not None else "")
                            + ")"
                        )
                except KeyError:               # pragma: no cover
                    pass
            block = render_product(row, marker, sent, extra)
            if len(block) > budget:
                break
            budget -= len(block) + 2
            title = _title(row.get("title", ""), 140)
            evidence.append(Evidence(marker, "product", block, {
                "parent_asin": asin,
                "title": title,
                "brand": str(row.get("brand", "")),
                "segment": str(row.get("segment", "")),
                "price": None if pd.isna(row.get("price")) else float(row["price"]),
                "average_rating": None if pd.isna(row.get("average_rating")) else float(row["average_rating"]),
                "rating_number": int(row.get("rating_number") or 0),
                "has_review_evidence": sent is not None,
                "score": float(row["score"]) if "score" in row and pd.notna(row.get("score")) else None,
            }))

            if not spec.wants_reviews or review_count >= MAX_REVIEW_EVIDENCE:
                continue
            for snip in self.retriever.review_evidence(asin, spec.aspects,
                                                       k=MAX_REVIEWS_PER_PRODUCT):
                if review_count >= MAX_REVIEW_EVIDENCE:
                    break
                rmarker = f"[R{review_count + 1}]"
                rblock = render_review(snip, rmarker, marker, title)
                if len(rblock) > budget:
                    break
                budget -= len(rblock) + 2
                review_count += 1
                evidence.append(Evidence(rmarker, "review", rblock, {
                    "parent_asin": asin,
                    "product_marker": marker,
                    "title": _title(title, 90),
                    "aspect": snip["aspect"],
                    "polarity": round(float(snip["polarity"]), 3),
                    "rating": snip.get("rating"),
                    "helpful_vote": snip.get("helpful_vote"),
                    "verified_purchase": snip.get("verified_purchase"),
                    "review_year": snip.get("review_year"),
                    "review_id": snip.get("review_id"),
                    "snippet": snip["snippet"],
                }))

        if spec.wants_price_stats or spec.question_type == "market":
            for j, line in enumerate(self.retriever.market_stats(spec, products), start=1):
                marker = f"[S{j}]"
                block = f"{marker} market statistic — {line}"
                if len(block) > budget:
                    break
                budget -= len(block) + 2
                evidence.append(Evidence(marker, "stat", block,
                                         {"label": line[:90], "source": "pricing.py"}))

        if not evidence:
            notes.append("No evidence could be retrieved for this question.")
        return evidence

    # -- generation ---------------------------------------------------------------------

    def answer(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
               temperature: float = DEFAULT_TEMPERATURE) -> RagAnswer:
        """Answer a question end to end: retrieve, generate, audit.

        Parameters
        ----------
        question:
            Natural-language question about the catalogue.
        max_new_tokens:
            Generation cap (default 420 - long enough for a 3-item recommendation with
            citations, short enough to stay a few seconds on the 4060).
        temperature:
            ``0`` (default) means greedy decoding.

        Returns
        -------
        RagAnswer
            The answer text plus every piece of evidence, the resolved citations and the
            grounding audit.
        """
        t0 = time.time()
        spec, evidence, notes, stats = self.retrieve(question)
        t_retrieval = time.time() - t0

        if not evidence:
            return RagAnswer(
                question=question, question_type=spec.question_type,
                answer=_NO_EVIDENCE_ANSWER, evidence=[], citations=[], unsupported_markers=[],
                uncited_sentences=[], unverified_numbers=[], query_spec=spec.to_dict(),
                retrieval=stats, timings={"retrieval_s": t_retrieval, "generation_s": 0.0},
                prompt_chars=0,
            )

        messages = build_prompt(question, spec.question_type, evidence, notes)
        prompt_chars = sum(len(m["content"]) for m in messages)
        text, gen_stats = self.llm.chat(messages, max_new_tokens=max_new_tokens,
                                        temperature=temperature)
        audit = audit_answer(text, evidence)

        return RagAnswer(
            question=question,
            question_type=spec.question_type,
            answer=text,
            evidence=evidence,
            citations=audit["citations"],
            unsupported_markers=audit["unsupported_markers"],
            uncited_sentences=audit["uncited_sentences"],
            unverified_numbers=audit["unverified_numbers"],
            misattributed_reviews=audit["misattributed_reviews"],
            query_spec=spec.to_dict(),
            retrieval=stats,
            timings={
                "retrieval_s": round(t_retrieval, 3),
                "generation_s": round(gen_stats["seconds"], 3),
                "tokens_per_second": round(gen_stats["tokens_per_second"], 1),
                "prompt_tokens": gen_stats["prompt_tokens"],
                "new_tokens": gen_stats["new_tokens"],
            },
            prompt_chars=prompt_chars,
            truncated=bool(gen_stats.get("truncated")),
        )


@lru_cache(maxsize=1)
def get_agent() -> RagAgent:
    """Process-wide cached agent (the Streamlit app should use this)."""
    return RagAgent()


def answer(question: str, **kwargs: Any) -> RagAnswer:
    """Module-level shortcut around :meth:`RagAgent.answer` using the cached agent."""
    return get_agent().answer(question, **kwargs)


# --------------------------------------------------------------------------------------
# 10. CLI / self-test
# --------------------------------------------------------------------------------------

SELFTEST_QUESTIONS = [
    "Compare the Acer Predator Helios 300 gaming laptop with the ASUS TUF Gaming A15 - "
    "which one is the better buy on specs and price?",
    "Which gaming laptops under $1200 do reviewers rate best on thermals and fan noise?",
    "Who are the main competitors to the Acer Aspire 3 slim laptop, and how are they "
    "positioned on price?",
    "What is the typical price of a gaming laptop compared with a chromebook in this market, "
    "and how much does a discrete GPU add?",
    "How long does the battery last on the Dell XPS 17 with the 4K OLED touchscreen?",
]


def _print_answer(res: RagAnswer, show_context: bool = False) -> None:
    """Pretty-print an answer with its citations and grounding audit."""
    print("\n" + "=" * 88)
    print(f"Q [{res.question_type}] {res.question}")
    print("=" * 88)
    if show_context:
        print("--- CONTEXT " + "-" * 76)
        for e in res.evidence:
            print(e.text)
        print("-" * 88)
    print(res.answer)
    print("-" * 88)
    print(f"retrieval: {res.retrieval['n_evidence']} evidence blocks "
          f"{res.retrieval['kinds']} from {res.retrieval['n_candidates']:,} candidates "
          f"({res.retrieval['relaxation']}; {res.retrieval.get('ordering', 'relevance')})")
    print(f"timing: retrieval {res.timings['retrieval_s']:.2f}s | generation "
          f"{res.timings.get('generation_s', 0):.1f}s "
          f"({res.timings.get('tokens_per_second', 0):.1f} tok/s, "
          f"{int(res.timings.get('prompt_tokens', 0))} prompt tokens)")
    print(f"grounding: citation rate {res.citation_rate:.0%} | "
          f"invented markers {res.unsupported_markers or 'none'} | "
          f"unverified numbers {res.unverified_numbers or 'none'} | "
          f"misattributed reviews "
          f"{[m['review'] + '->' + m['cited_with'] for m in res.misattributed_reviews] or 'none'}"
          + (" | TRUNCATED at the token cap" if res.truncated else ""))
    print("citations:")
    for c in res.citations:
        detail = ""
        if c["kind"] == "review":
            detail = f" | aspect={c.get('aspect')} polarity={c.get('polarity')} -> {c.get('product_marker')}"
        elif c["kind"] == "product":
            price = c.get("price")
            detail = f" | {c.get('brand')}/{c.get('segment')} | " + (
                f"${price:,.2f}" if price else "price unknown")
        print(f"  {c['marker']:<5} {c['kind']:<8} {c['title'][:78]}{detail}")


def _selftest(questions: Sequence[str], show_context: bool, max_new_tokens: int) -> int:
    """Run the self-test battery and return a process exit code."""
    agent = get_agent()
    results = []
    for q in questions:
        res = agent.answer(q, max_new_tokens=max_new_tokens)
        _print_answer(res, show_context=show_context)
        results.append(res)

    print("\n" + "=" * 88)
    print("SELF-TEST SUMMARY")
    print("=" * 88)
    ok = True
    for res in results:
        flag = "OK " if (res.citation_rate >= 0.6 and not res.unsupported_markers
                         and not res.misattributed_reviews and not res.truncated) else "CHECK"
        ok &= flag == "OK "
        print(f"{flag} [{res.question_type:<13}] cites={res.citation_rate:5.0%} "
              f"blocks={res.retrieval['n_evidence']:>2} "
              f"invented={len(res.unsupported_markers)} "
              f"unverified_nums={len(res.unverified_numbers)} "
              f"misattrib={len(res.misattributed_reviews)} "
              f"trunc={int(res.truncated)} "
              f"gen={res.timings.get('generation_s', 0):5.1f}s :: {res.question[:52]}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Grounded RAG agent over the laptop catalogue")
    ap.add_argument("-q", "--question", help="ask a single question")
    ap.add_argument("--selftest", action="store_true", help="run the built-in question battery")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="print the retrieved context without loading the LLM")
    ap.add_argument("--show-context", action="store_true", help="also print the full context")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--json", action="store_true", help="emit the answer as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.retrieval_only:
        questions = [args.question] if args.question else list(SELFTEST_QUESTIONS)
        retriever = get_retriever()
        agent = RagAgent(retriever=retriever, llm=LocalLLM())   # LLM never loaded
        for q in questions:
            spec, evidence, notes, stats = agent.retrieve(q)
            print("\n" + "=" * 88)
            print(f"Q [{spec.question_type}] {q}")
            print(f"parse: {json.dumps({k: v for k, v in spec.to_dict().items() if v not in (None, [], False)})}")
            print(f"stats: {stats}")
            print("=" * 88)
            for e in evidence:
                print(e.text)
            for n in notes:
                print(f"NOTE: {n}")
        return 0

    if args.selftest or not args.question:
        return _selftest(SELFTEST_QUESTIONS, args.show_context, args.max_new_tokens)

    res = get_agent().answer(args.question, max_new_tokens=args.max_new_tokens)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    else:
        _print_answer(res, show_context=args.show_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
