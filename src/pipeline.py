"""Data cleaning and spec-parsing pipeline for the laptop competitor-intelligence system.

Reads the raw Amazon Reviews 2023 (McAuley Lab) laptop dumps, filters out lingering
accessories / non-laptops, normalizes brands, parses free-text spec fields into the
typed schema documented in ``config.py``, collapses configuration variants of the same
model, assigns a market segment, cleans the review text and writes:

    data/processed/products.parquet
    data/processed/reviews.parquet

Run as a script::

    /home/anmol/project/.venv/bin/python /home/anmol/project/src/pipeline.py

The individual parsers (:func:`parse_cpu`, :func:`parse_ram`, :func:`parse_storage`,
:func:`parse_gpu`, :func:`parse_screen`, :func:`normalize_brand`, :func:`assign_segment`)
are importable and reusable by the other modules and by tests.
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BUSINESS_SERIES_HINTS,
    GAMING_BRAND_HINTS,
    PRODUCTS_PARQUET,
    PRODUCT_KEY,
    RAW_META,
    RAW_REVIEWS,
    REVIEWS_PARQUET,
    SEGMENTS,
    ULTRABOOK_SERIES_HINTS,
)

# --------------------------------------------------------------------------------------
# 1. Accessory / non-laptop filtering
# --------------------------------------------------------------------------------------

#: Product types that are definitely NOT a laptop (accessories, spare parts, other goods).
ACCESSORY_RE = re.compile(
    r"(carrying case|laptop case\b|laptop sleeve|\bsleeve\b|backpack|messenger bag"
    r"|laptop bag\b|skin decal|decal sticker|vinyl decal|\bdecal\b|screen protector"
    r"|privacy (?:screen|filter)|keyboard cover|silicone keyboard|palm ?rest|cooling pad"
    r"|laptop stand\b|docking station|\blap ?dock\b|\bac adapter\b|power adapter|\bcharger\b"
    r"|replacement (?:battery|keyboard|screen|lcd|palmrest|hinge|motherboard|fan|adapter)"
    r"|\bmotherboard\b|so-?dimm|memory module|ram upgrade kit|hdd caddy|mouse pad|dry erase"
    r"|\bcapri\b|playstation|\bps5\b|\bxbox\b|all-in-one (?:computer|desktop|pc)"
    r"|desktop (?:computer|pc|tower)|operating system -)",
    re.I,
)

#: Other form factors that are not laptops unless the title also reads like a laptop.
FORM_FACTOR_RE = re.compile(
    r"(\btablet\b|\be-?reader\b|smartphone|\bcell ?phone\b|\bmonitor\b|\bhdtv\b|\btelevision\b)",
    re.I,
)

#: Desktop / all-in-one giveaways. Only fire when no portable form-factor word is present
#: (``'GPD Pocket3 ... Mini Pc Notebook Laptop'`` and ``'Aspire One ... Mini PC'`` are laptops).
DESKTOP_RE = re.compile(
    r"(all[- ]?in[- ]?one\b|all in one\b|\baio\b|elitedesk|prodesk|thinkcentre|optiplex"
    r"|\btower\b|micro[- ]tower|mini[- ]tower|\bsff\b|small form factor)",
    re.I,
)
PORTABLE_RE = re.compile(r"(laptop|notebook|ultrabook|chromebook|netbook|portable|macbook)", re.I)

#: Evidence that the listing really is a notebook computer.
LAPTOP_RE = re.compile(
    r"(laptop|notebook|chromebook|ultrabook|macbook|netbook|thinkpad|ideapad|pavilion"
    r"|inspiron|latitude|elitebook|probook|aspire|vivobook|zenbook|omen|nitro|predator"
    r"|legion|toughbook|vaio|satellite|travelmate|\bxps\b|spectre|envy|yoga|swift|\bgram\b"
    r"|surface (?:laptop|book|pro)|2-?in-?1|2 in 1|convertible|clamshell)",
    re.I,
)

#: "... Laptop **with** carrying case" -> a bundle, still a laptop.
BUNDLE_CTX_RE = re.compile(r"(with|w/|includes?|including|bundle[ds]?|plus|\+|and)\s*$", re.I)


def is_accessory(title: str, categories: Iterable[str] | None = None) -> tuple[bool, str]:
    """Decide whether a listing is an accessory / non-laptop rather than a real laptop.

    Parameters
    ----------
    title:
        Raw product title.
    categories:
        Raw Amazon category breadcrumb (used as a weak extra signal).

    Returns
    -------
    (drop, reason)
        ``drop`` is True when the listing should be removed; ``reason`` is a short
        machine-friendly tag ('accessory', 'other_form_factor', 'desktop_or_aio',
        'no_title', '').
    """
    t = (title or "").strip()
    if not t:
        return True, "no_title"

    cat_text = " ".join(categories or []).lower()
    if "accessories & peripherals" in cat_text or "laptop bags" in cat_text:
        return True, "accessory"

    lap = LAPTOP_RE.search(t)
    acc = ACCESSORY_RE.search(t)
    if acc is not None:
        preceding = t[max(0, acc.start() - 14) : acc.start()]
        is_bundle = bool(BUNDLE_CTX_RE.search(preceding))
        if not is_bundle and (lap is None or acc.start() < lap.start()):
            return True, "accessory"

    form = FORM_FACTOR_RE.search(t)
    if form is not None and lap is None:
        return True, "other_form_factor"

    if DESKTOP_RE.search(t) and PORTABLE_RE.search(t) is None:
        return True, "desktop_or_aio"

    return False, ""


# --------------------------------------------------------------------------------------
# 2. Brand normalization
# --------------------------------------------------------------------------------------

#: Resellers / refurbishers that appear in ``store`` but are not laptop manufacturers.
RESELLER_STORES = {
    "amazon renewed",
    "amazon",
    "renewed",
    "computer upgrade king",
    "cuk",
    "oemgenuine",
    "excaliberpc",
    "hidevolution",
    "me2 michaelelectronics2",
    "ist computers",
    "tech data",
    "genuine",
    "quality refurbished computers",
    "discount electronics",
    "pc wholesale",
    "amazon.com",
    "electronics-salon",
}

#: Canonical brand spellings keyed by a lowercase alias.
BRAND_ALIASES = {
    "hp": "HP",
    "hewlett packard": "HP",
    "hewlett-packard": "HP",
    "hewlett packard enterprise": "HP",
    "hp inc": "HP",
    "compaq": "HP",
    "dell": "Dell",
    "dell computer": "Dell",
    "dell technologies": "Dell",
    "alienware": "Alienware",
    "lenovo": "Lenovo",
    "ibm": "Lenovo",
    "thinkpad": "Lenovo",
    "asus": "ASUS",
    "asustek": "ASUS",
    "asus computer international": "ASUS",
    "rog": "ASUS",
    "acer": "Acer",
    "gateway": "Gateway",
    "msi": "MSI",
    "micro-star international": "MSI",
    "microsoft": "Microsoft",
    "microsoft surface": "Microsoft",
    "apple": "Apple",
    "samsung": "Samsung",
    "samsung electronics": "Samsung",
    "sony": "Sony",
    "vaio": "VAIO",
    "toshiba": "Toshiba",
    "dynabook": "Toshiba",
    "lg": "LG",
    "lg electronics": "LG",
    "razer": "Razer",
    "gigabyte": "GIGABYTE",
    "aorus": "GIGABYTE",
    "sager": "Sager",
    "clevo": "Sager",
    "panasonic": "Panasonic",
    "fujitsu": "Fujitsu",
    "eluktronics": "Eluktronics",
    "google": "Google",
    "huawei": "Huawei",
    "xiaomi": "Xiaomi",
    "chuwi": "CHUWI",
    "jumper": "Jumper",
    "evoo": "EVOO",
    "rca": "RCA",
    "system76": "System76",
    "framework": "Framework",
    "prostar": "Prostar",
    "averatec": "Averatec",
    "sgin": "SGIN",
    "nec": "NEC",
    "everex": "Everex",
    "emachines": "eMachines",
    "packard bell": "Packard Bell",
    "raspberry pi": "Raspberry Pi",
}

#: Brands searched for inside the title when store/details are useless.
TITLE_BRAND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(hewlett[- ]packard|hp)\b", re.I), "HP"),
    (re.compile(r"\balienware\b", re.I), "Alienware"),
    (re.compile(r"\bdell\b", re.I), "Dell"),
    (re.compile(r"\b(lenovo|thinkpad|ideapad|thinkbook)\b", re.I), "Lenovo"),
    (re.compile(r"\b(asus|zenbook|vivobook|rog\b)", re.I), "ASUS"),
    (re.compile(r"\b(acer|predator|nitro)\b", re.I), "Acer"),
    (re.compile(r"\b(msi)\b", re.I), "MSI"),
    (re.compile(r"\b(apple|macbook)\b", re.I), "Apple"),
    (re.compile(r"\b(microsoft|surface (?:laptop|book|pro|go))\b", re.I), "Microsoft"),
    (re.compile(r"\bsamsung\b", re.I), "Samsung"),
    (re.compile(r"\b(sony|vaio)\b", re.I), "Sony"),
    (re.compile(r"\b(toshiba|dynabook|tecra|satellite)\b", re.I), "Toshiba"),
    (re.compile(r"\brazer\b", re.I), "Razer"),
    (re.compile(r"\b(gigabyte|aorus)\b", re.I), "GIGABYTE"),
    (re.compile(r"\b(lg gram|lg)\b", re.I), "LG"),
    (re.compile(r"\bpanasonic|toughbook\b", re.I), "Panasonic"),
    (re.compile(r"\bfujitsu|lifebook\b", re.I), "Fujitsu"),
    (re.compile(r"\bgateway\b", re.I), "Gateway"),
    (re.compile(r"\bcompaq\b", re.I), "HP"),
    (re.compile(r"\bsager\b", re.I), "Sager"),
    (re.compile(r"\beluktronics\b", re.I), "Eluktronics"),
    (re.compile(r"\bhuawei|matebook\b", re.I), "Huawei"),
    (re.compile(r"\bgoogle|pixelbook\b", re.I), "Google"),
    (re.compile(r"\bchuwi\b", re.I), "CHUWI"),
    (re.compile(r"\bsystem76\b", re.I), "System76"),
    (re.compile(r"\bibm\b", re.I), "Lenovo"),
]

RENEWED_RE = re.compile(
    r"(renewed|refurbish|refurb\b|certified pre[- ]?owned|\bcpo\b|pre[- ]owned|off[- ]lease)",
    re.I,
)


def _clean_brand_token(value: str | None) -> str | None:
    """Return a trimmed brand token, or None when the value is unusable."""
    if value is None:
        return None
    v = str(value).strip().strip(",;|")
    v = re.sub(r"\s+", " ", v)
    if not v or v.lower() in {"unknown", "n/a", "na", "none", "generic", "-", "brand"}:
        return None
    if len(v) > 40:
        return None
    return v


def normalize_brand(store: str | None, details: dict | None = None, title: str = "") -> str:
    """Return the canonical manufacturer brand for a listing.

    ``store`` is dirty (case variants, ``'HEWLETT PACKARD'``) and for ~6.4k records it is
    a reseller such as ``'Amazon Renewed'``. In that case the true brand is recovered from
    ``details['Brand']`` and finally from the title.

    Parameters
    ----------
    store:
        Raw ``store`` field from the meta record.
    details:
        Raw ``details`` dict of the meta record.
    title:
        Raw product title (last-resort source).

    Returns
    -------
    str
        Canonical brand, e.g. ``'HP'``, ``'Lenovo'``, ``'ASUS'``; ``'Unknown'`` when
        nothing can be recovered.
    """
    details = details or {}
    candidates: list[str] = []

    s = _clean_brand_token(store)
    if s is not None and s.lower() not in RESELLER_STORES and not RENEWED_RE.search(s):
        candidates.append(s)

    for key in ("Brand", "Manufacturer"):
        d = _clean_brand_token(details.get(key))
        if d is not None and d.lower() not in RESELLER_STORES and not RENEWED_RE.search(d):
            candidates.append(d)

    for cand in candidates:
        low = cand.lower()
        if low in BRAND_ALIASES:
            return BRAND_ALIASES[low]
        head = low.split()[0]
        if head in BRAND_ALIASES:
            return BRAND_ALIASES[head]

    for pattern, brand in TITLE_BRAND_PATTERNS:
        if pattern.search(title or ""):
            return brand

    # Nothing canonical matched: fall back to a title-cased version of the best candidate
    # so small/no-name brands survive instead of collapsing into 'Unknown'.
    for cand in candidates:
        if re.search(r"[A-Za-z]", cand):
            return cand if cand.isupper() and len(cand) <= 5 else cand.title()
    return "Unknown"


def detect_renewed(store: str | None, title: str, details: dict | None = None) -> bool:
    """True when the listing is a renewed / refurbished / pre-owned unit."""
    details = details or {}
    blob = " ".join(
        str(x)
        for x in (store or "", title or "", details.get("Special Feature", ""), details.get("Color", ""))
    )
    return bool(RENEWED_RE.search(blob))


# --------------------------------------------------------------------------------------
# 3. Spec parsing
# --------------------------------------------------------------------------------------

_CPU_BRAND_MAP = {
    "intel": "Intel",
    "amd": "AMD",
    "apple": "Apple",
    "qualcomm": "Qualcomm",
    "snapdragon": "Qualcomm",
    "mediatek": "MediaTek",
    "arm": "ARM",
    "nvidia": "NVIDIA",
    "via": "VIA",
    "samsung": "Samsung",
    "rockchip": "Rockchip",
    "powerpc": "PowerPC",
    "transmeta": "Transmeta",
    "ibm": "PowerPC",
}

# family regexes are applied in order against a normalised text blob
_CPU_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str, str, float | None]] = [
    # (pattern, cpu_family, cpu_brand, cpu_tier)
    (re.compile(r"core\s*ultra\s*([579])"), "Core Ultra {0}", "Intel", None),
    (re.compile(r"\b(?:intel\s*)?core\s*[ _-]?i([3579])\b"), "Core i{0}", "Intel", None),
    (re.compile(r"\bi([3579])[ -]?\d{3,5}[a-z]{0,2}\b"), "Core i{0}", "Intel", None),
    (re.compile(r"\bcore\s*2\s*(?:duo|quad|solo)\b"), "Core 2", "Intel", None),
    (re.compile(r"\bcore\s*duo\b"), "Core Duo", "Intel", None),
    (re.compile(r"\bcore\s*m\d?\b|\bcore[ _-]?m\b|\bm[357][ -]?\d{4}\b"), "Core M", "Intel", None),
    (re.compile(r"\bceleron\b"), "Celeron", "Intel", None),
    (re.compile(r"\bpentium\b"), "Pentium", "Intel", None),
    (re.compile(r"\batom\b"), "Atom", "Intel", None),
    (re.compile(r"\bxeon\b"), "Xeon", "Intel", None),
    (re.compile(r"\bryzen\s*[_ -]?([3579])\b"), "Ryzen {0}", "AMD", None),
    (re.compile(r"\bryzen\b"), "Ryzen", "AMD", None),
    (re.compile(r"\bathlon\b"), "Athlon", "AMD", None),
    (re.compile(r"\bamd\s*a(\d{1,2})[- ]?\d{4}\b|\ba([46810])[- ]?\d{4}\b"), "A Series", "AMD", None),
    (re.compile(r"\bamd\s*a[- ]?series\b|\ba[- ]?series\b"), "A Series", "AMD", None),
    (re.compile(r"\bamd\s*r[- ]?series\b|\bradeon r series\b"), "R Series", "AMD", None),
    (re.compile(r"\bamd\s*e[- ]?series\b|\be[12]-\d{4}\b"), "E Series", "AMD", None),
    (re.compile(r"\bfx[- ]?\d{4}\b|\bamd fx\b"), "FX", "AMD", None),
    (re.compile(r"\bturion\b"), "Turion", "AMD", None),
    (re.compile(r"\bsempron\b"), "Sempron", "AMD", None),
    (re.compile(r"\bphenom\b"), "Phenom", "AMD", None),
    (re.compile(r"\bapple\s*m([1234])\b|\bm([1234])\s*(?:pro|max|ultra)?\s*chip\b"), "M{0}", "Apple", None),
    (re.compile(r"\bsnapdragon\b"), "Snapdragon", "Qualcomm", None),
    (re.compile(r"\bmediatek\b|\bmt\d{4}\b"), "MediaTek", "MediaTek", None),
    (re.compile(r"\bcore\s*solo\b"), "Core Solo", "Intel", None),
]

_TIER_FAMILIES = re.compile(r"(?:Core i|Ryzen |Core Ultra )([3579])$")


def _clean_cpu_text(*parts: Any) -> str:
    """Lowercase + de-underscore CPU text fragments ('core_i7' -> 'core i7')."""
    txt = " ".join(str(p) for p in parts if p not in (None, "", float("nan")))
    txt = txt.replace("_", " ").replace("®", " ").replace("™", " ")
    return re.sub(r"\s+", " ", txt).strip().lower()


def _extract_ghz(text: str, prefer_base: bool = True) -> float:
    """Pull a plausible clock speed in GHz out of free text (NaN when absent)."""
    if not text:
        return math.nan
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(ghz|mhz)", text, re.I))
    if not matches:
        return math.nan
    base, boost = [], []
    for m in matches:
        val = float(m.group(1))
        if m.group(2).lower() == "mhz":
            val /= 1000.0
        if not (0.5 <= val <= 6.5):
            continue
        preceding = text[max(0, m.start() - 12) : m.start()].lower()
        (boost if "up to" in preceding else base).append(val)
    pool = base or boost if prefer_base else boost or base
    if not pool:
        return math.nan
    return float(pool[0])


def parse_cpu(details: dict | None, title: str = "") -> dict[str, Any]:
    """Parse CPU brand / family / tier / clock speed.

    Reads ``details`` fields ``Processor`` (e.g. ``'2.6 GHz core_i7'``), ``CPU Model``
    (``'Core i7'``, ``'AMD R Series'``), ``Processor Brand``, ``CPU Speed`` and falls back
    to the product title (``'Intel i7-1260P 12-Core 2.10GHz'``).

    Returns
    -------
    dict with keys ``cpu_brand``, ``cpu_family``, ``cpu_tier``, ``cpu_ghz``.
    """
    details = details or {}
    det_blob = _clean_cpu_text(
        details.get("CPU Model"),
        details.get("Processor"),
        details.get("Processor Description"),
        details.get("CPU Manufacturer"),
        details.get("Processor Brand"),
    )
    title_blob = _clean_cpu_text(title)

    family = "Unknown"
    brand = "Unknown"
    for blob in (det_blob, title_blob):
        if not blob:
            continue
        for pattern, fam_tpl, fam_brand, _ in _CPU_FAMILY_PATTERNS:
            m = pattern.search(blob)
            if m:
                groups = [g for g in m.groups() if g]
                family = fam_tpl.format(*groups) if groups and "{0}" in fam_tpl else fam_tpl.replace("{0}", "")
                family = family.strip()
                brand = fam_brand
                break
        if family != "Unknown":
            break

    # explicit brand fields win when they disagree only on the brand
    raw_brand = _clean_cpu_text(details.get("Processor Brand"), details.get("CPU Manufacturer"))
    if brand == "Unknown":
        for key, val in _CPU_BRAND_MAP.items():
            if re.search(rf"\b{re.escape(key)}\b", raw_brand):
                brand = val
                break
    if brand == "Unknown":
        for key, val in _CPU_BRAND_MAP.items():
            if re.search(rf"\b{re.escape(key)}\b", title_blob):
                brand = val
                break

    tier_match = _TIER_FAMILIES.search(family)
    tier = float(tier_match.group(1)) if tier_match else math.nan

    ghz = math.nan
    for source in (
        _clean_cpu_text(details.get("Processor")),
        _clean_cpu_text(details.get("CPU Speed")),
        _clean_cpu_text(details.get("Processor Speed")),
        title_blob,
    ):
        ghz = _extract_ghz(source)
        if not math.isnan(ghz):
            break

    return {"cpu_brand": brand, "cpu_family": family, "cpu_tier": tier, "cpu_ghz": ghz}


_RAM_TYPE_RE = re.compile(
    r"\b(lpddr5x|lpddr5|lpddr4x|lpddr4|lpddr3|ddr5|ddr4l|ddr4|ddr3l|ddr3|ddr2|dddr4|sdram|so-?dimm)\b",
    re.I,
)
_RAM_TYPE_CANON = {
    "dddr4": "DDR4",
    "ddr4l": "DDR4",
    "so-dimm": "Unknown",
    "sodimm": "Unknown",
    "sdram": "SDRAM",
}


def _to_gb(value: float, unit: str) -> float:
    """Convert a (value, unit) pair to GB (1 TB -> 1024 GB, 512 MB -> 0.5 GB)."""
    unit = unit.lower()
    if unit.startswith("tb") or unit.startswith("t"):
        return value * 1024.0
    if unit.startswith("mb") or unit == "m":
        return value / 1024.0
    return value


# Capacities a laptop actually ships with. Anything else in a RAM field is a unit slip
# or a storage capacity that leaked into the wrong column.
_PLAUSIBLE_RAM_GB = frozenset(
    {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0,
     20.0, 24.0, 32.0, 36.0, 40.0, 48.0, 64.0, 96.0, 128.0}
)


def _detail_storage_gb(details: dict | None) -> float | None:
    """Best-effort drive capacity in GB, used only to spot RAM/storage column swaps."""
    for key in ("Hard Drive", "Hard Disk Size", "Flash Memory Size", "Total Storage Capacity"):
        val = (details or {}).get(key)
        if val is None:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", str(val), re.I)
        if m:
            return _to_gb(float(m.group(1)), m.group(2))
    return None


def parse_ram(details: dict | None, title: str = "") -> dict[str, Any]:
    """Parse installed RAM size (GB) and memory type.

    Handles ``'16 GB DDR4'``, bare ``'8 GB'``, bare ``'DDR4'``, ``'1024 MB'`` and title
    forms such as ``'(12GB DDR4, 512GB SSD)'`` or ``'RAM: 8 GB'``.

    Returns
    -------
    dict with keys ``ram_gb`` and ``ram_type``.
    """
    details = details or {}
    ram_gb = math.nan
    ram_type = "Unknown"

    det_fields = [
        details.get("Ram Memory Installed Size"),
        details.get("RAM"),
        details.get("Memory Storage Capacity"),
        details.get("Installed RAM"),
    ]
    # Field order is empirical: 'Ram Memory Installed Size' agrees with the title more
    # often than 'RAM' does when the two disagree (66% vs 30%), so it stays first.
    # It does, however, sometimes echo the DRIVE size, so candidates are screened below.
    drive_gb = _detail_storage_gb(details)
    candidates: list[float] = []
    for field in det_fields:
        if field is None:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", str(field), re.I)
        if m:
            cand = _to_gb(float(m.group(1)), m.group(2))
            if 0.12 <= cand <= 256:
                candidates.append(cand)
    for cand in candidates:
        # Reject sizes no laptop ships (catches storage capacities and unit slips),
        # and reject a value that merely echoes this product's own drive size.
        if cand not in _PLAUSIBLE_RAM_GB:
            continue
        if drive_gb is not None and cand >= 16 and abs(cand - drive_gb) < 0.01:
            continue
        ram_gb = cand
        break
    if math.isnan(ram_gb) and candidates:
        # Nothing passed the screen; fall through to the title parse below, and only
        # use the raw detail value if the title yields nothing either.
        pass

    type_blob = " ".join(
        str(x) for x in (details.get("RAM"), details.get("Computer Memory Type"), details.get("Memory Technology")) if x
    )
    m = _RAM_TYPE_RE.search(type_blob)
    if m:
        raw = m.group(1).lower()
        ram_type = _RAM_TYPE_CANON.get(raw, raw.upper())

    if math.isnan(ram_gb) or ram_type == "Unknown":
        t = title or ""
        if math.isnan(ram_gb):
            pats = [
                r"(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s*(?:of\s*)?(?:lp)?(?:ddr\d\w*\s*)?(?:sdram|ram|memory)\b",
                r"\bram\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(gb|tb|mb)\b",
                r"(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s*(?:lp)?ddr\d\w*\b",
            ]
            for pat in pats:
                m = re.search(pat, t, re.I)
                if m:
                    cand = _to_gb(float(m.group(1)), m.group(2))
                    if 0.12 <= cand <= 256:
                        ram_gb = cand
                        break
        if ram_type == "Unknown":
            m = _RAM_TYPE_RE.search(t)
            if m:
                raw = m.group(1).lower()
                ram_type = _RAM_TYPE_CANON.get(raw, raw.upper())

    if math.isnan(ram_gb) and candidates:
        ram_gb = candidates[0]

    return {"ram_gb": ram_gb, "ram_type": ram_type}


_STORAGE_TYPE_PATTERNS = [
    (re.compile(r"\be-?mmc\b", re.I), "eMMC"),
    (re.compile(r"\b(ssd|solid[ -]?state|nvme|pcie|m\.?2|flash memory solid state)\b", re.I), "SSD"),
    (re.compile(r"\b(hdd|hard disk drive|mechanical hard|sata hard|5400 ?rpm|7200 ?rpm)\b", re.I), "HDD"),
    (re.compile(r"\b(hybrid|sshd|fusion drive)\b", re.I), "Hybrid"),
]


def parse_storage(details: dict | None, title: str = "") -> dict[str, Any]:
    """Parse primary storage capacity (normalized to GB) and storage type.

    ``'1 TB SSD'`` -> ``(1024.0, 'SSD')``; bare ``'SSD'`` -> ``(NaN, 'SSD')``;
    ``'64 GB Emmc'`` -> ``(64.0, 'eMMC')``. Falls back to the title
    (``'512GB PCIe SSD'``, ``'Storage: 256 GB'``).

    Returns
    -------
    dict with keys ``storage_gb`` and ``storage_type``.
    """
    details = details or {}
    storage_gb = math.nan
    storage_type = "Unknown"

    size_fields = [
        details.get("Hard Disk Size"),
        details.get("Hard Drive"),
        details.get("Hard Drive Size"),
        details.get("Flash Memory Size"),
        details.get("Memory Storage Capacity"),
    ]
    for field in size_fields:
        if field is None:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", str(field), re.I)
        if m:
            cand = _to_gb(float(m.group(1)), m.group(2))
            if 4 <= cand <= 16384:
                storage_gb = cand
                break

    type_blob = " ".join(
        str(x)
        for x in (
            details.get("Hard Drive"),
            details.get("Hard Disk Description"),
            details.get("Hard Drive Interface"),
            details.get("Flash Memory Size"),
            details.get("Hard Drive Rotational Speed"),
        )
        if x
    )
    for pattern, label in _STORAGE_TYPE_PATTERNS:
        if pattern.search(type_blob):
            storage_type = label
            break
    if storage_type == "Unknown" and details.get("Flash Memory Size"):
        storage_type = "SSD"

    t = title or ""
    if math.isnan(storage_gb):
        pats = [
            r"(\d+(?:\.\d+)?)\s*(tb|gb)\s*(?:pcie\s*|nvme\s*|m\.?2\s*|sata\s*|gen\s*\d\s*)*"
            r"(?:ssd|hdd|e-?mmc|hard drive|solid state|storage)",
            r"storage\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(tb|gb)\b",
        ]
        for pat in pats:
            m = re.search(pat, t, re.I)
            if m:
                cand = _to_gb(float(m.group(1)), m.group(2))
                if 4 <= cand <= 16384:
                    storage_gb = cand
                    break
    if storage_type == "Unknown":
        for pattern, label in _STORAGE_TYPE_PATTERNS:
            if pattern.search(t):
                storage_type = label
                break

    return {"storage_gb": storage_gb, "storage_type": storage_type}


_DISCRETE_RE = re.compile(
    r"(geforce|\brtx\b|\bgtx\b|\bgtx\d|(?:geforce|nvidia)\s*mx\s?\d{3}|quadro|\bnvs\b|titan"
    r"|firepro|\bfire ?gl\b|radeon\s*(?:pro\s*)?(?:rx|hd|r[5-9])\s*\d{3,4}|radeon\s*pro\s*\d"
    r"|\brx\s*[5-7]\d{3}\b|radeon\s*r[5-9]\s*m\d{2,3}|geforce\s*gt\s?\d{3})",
    re.I,
)
_INTEGRATED_RE = re.compile(
    r"(intel|uhd graphics|hd graphics|iris|integrated|apple|radeon graphics|radeon vega"
    r"|rx vega [1-9]\b|rx vega 1[01]\b|vega [1-9]\b|graphics media accelerator|mediatek|adreno"
    r"|radeon\s*r[2-9]\b|\bgma\b|shared)",
    re.I,
)
_GPU_MODEL_PATTERNS = [
    (re.compile(r"\brtx\s*(a?\d{3,4}\s*(?:ti|super|max-q|ada)?)\b", re.I), "RTX {0}"),
    (re.compile(r"\bgtx\s*(\d{3,4}\s*(?:ti|super|max-q)?)\b", re.I), "GTX {0}"),
    (re.compile(r"\b(?:geforce\s*)?mx\s*(\d{3})\b", re.I), "GeForce MX{0}"),
    (re.compile(r"\bquadro\s*([a-z]?\d{3,4}\s*m?)\b", re.I), "Quadro {0}"),
    (re.compile(r"\bgeforce\s*(gt[x]?\s*\d{3,4}m?)\b", re.I), "GeForce {0}"),
    (re.compile(r"\bradeon\s*(?:pro\s*)?rx\s*(\d{3,4}\s*[a-z]{0,2})\b", re.I), "Radeon RX {0}"),
    (re.compile(r"\bradeon\s*(r[2-9]\s*m?\d{0,3})\b", re.I), "Radeon {0}"),
    (re.compile(r"\b(iris xe(?:\s*max)?)\b", re.I), "Iris Xe"),
    (re.compile(r"\b(iris plus|iris pro|iris)\b", re.I), "Iris"),
    (re.compile(r"\buhd graphics\s*(\d{3,4})?\b", re.I), "UHD Graphics {0}"),
    (re.compile(r"\bhd graphics\s*(\d{3,4})?\b", re.I), "HD Graphics {0}"),
    (re.compile(r"\bradeon\s*(vega\s*\d{1,2})\b", re.I), "Radeon {0}"),
    (re.compile(r"\b(radeon graphics)\b", re.I), "Radeon Graphics"),
    (re.compile(r"\bapple\s*(m[1234](?:\s*(?:pro|max|ultra))?)\b", re.I), "Apple {0} GPU"),
]


def parse_gpu(details: dict | None, title: str = "") -> dict[str, Any]:
    """Parse graphics brand / model and whether the GPU is discrete.

    Distinguishes integrated silicon (Intel UHD/Iris, AMD "Radeon Graphics"/Vega APU,
    Apple) from discrete parts (GeForce/RTX/GTX/MX/Quadro, Radeon RX 5500M, FirePro).
    ``details['Card Description']`` ('Integrated'/'Dedicated') is used only as a
    tie-breaker when the model string is uninformative.

    Returns
    -------
    dict with keys ``gpu_brand``, ``gpu_model``, ``is_discrete_gpu``.
    """
    details = details or {}
    gpu_text = " ".join(
        str(x)
        for x in (
            details.get("Graphics Coprocessor"),
            details.get("Graphics Card Description"),
            details.get("Graphics Processor Manufacturer"),
        )
        if x
    ).strip()
    card_desc = str(details.get("Card Description") or "")
    chipset = str(details.get("Chipset Brand") or "")

    search_blobs = [b for b in (gpu_text, title or "") if b]

    model = "Unknown"
    for blob in search_blobs:
        for pattern, tpl in _GPU_MODEL_PATTERNS:
            m = pattern.search(blob)
            if m:
                groups = [g.strip() for g in m.groups() if g]
                model = tpl.format(*groups) if groups else tpl.replace(" {0}", "")
                model = re.sub(r"\s+", " ", model).strip()
                break
        if model != "Unknown":
            break

    blob_all = " ".join(search_blobs)
    brand = "Unknown"
    if re.search(r"nvidia|geforce|\brtx\b|\bgtx\b|quadro|\bmx\s?\d{3}\b", blob_all, re.I):
        brand = "NVIDIA"
    elif re.search(r"\bamd\b|radeon|firepro|\bati\b", blob_all, re.I):
        brand = "AMD"
    elif re.search(r"intel|uhd graphics|hd graphics|iris|graphics media accelerator", blob_all, re.I):
        brand = "Intel"
    elif re.search(r"apple\s*m[1234]|\bapple\b", blob_all, re.I):
        brand = "Apple"
    elif re.search(r"mediatek|adreno|qualcomm|mali", blob_all, re.I):
        brand = "Other"
    elif chipset:
        low = chipset.lower()
        for key, val in (
            ("nvidia", "NVIDIA"),
            ("amd", "AMD"),
            ("ati", "AMD"),
            ("intel", "Intel"),
            ("apple", "Apple"),
        ):
            if key in low:
                brand = val
                break

    # discrete / integrated decision
    is_discrete: bool | None = None
    model_blob = f"{model} {gpu_text}"
    if _DISCRETE_RE.search(model_blob) and not re.search(r"rx vega \d{1,2}\b", model_blob, re.I):
        is_discrete = True
    elif _INTEGRATED_RE.search(model_blob):
        is_discrete = False
    if is_discrete is None and title:
        if _DISCRETE_RE.search(title) and not re.search(r"rx vega \d{1,2}\b", title, re.I):
            is_discrete = True
        elif _INTEGRATED_RE.search(title):
            is_discrete = False
    if is_discrete is None:
        low = card_desc.lower()
        if "dedicated" in low:
            is_discrete = True
        elif "integrated" in low or "shared" in low:
            is_discrete = False
    if is_discrete is None:
        is_discrete = False

    if brand == "Unknown" and is_discrete is False and model != "Unknown":
        brand = "Intel"

    return {"gpu_brand": brand, "gpu_model": model, "is_discrete_gpu": bool(is_discrete)}


_RES_KEYWORDS = [
    (re.compile(r"\b(?:4k|uhd\+?|3840\s*x\s*2160)\b", re.I), (3840.0, 2160.0)),
    (re.compile(r"\b(?:qhd\+|wqxga|2560\s*x\s*1600)\b", re.I), (2560.0, 1600.0)),
    (re.compile(r"\b(?:qhd|wqhd|2k|1440p|2560\s*x\s*1440)\b", re.I), (2560.0, 1440.0)),
    (re.compile(r"\b(?:wuxga|1920\s*x\s*1200|fhd\+)\b", re.I), (1920.0, 1200.0)),
    (re.compile(r"\b(?:fhd|full hd|1080p|1920\s*x\s*1080)\b", re.I), (1920.0, 1080.0)),
    (re.compile(r"\bhd\+\b", re.I), (1600.0, 900.0)),
    (re.compile(r"\b(?:hd ready|hd)\b", re.I), (1366.0, 768.0)),
]


def parse_screen(details: dict | None, title: str = "") -> dict[str, Any]:
    """Parse screen diagonal (inches) and native resolution.

    Reads ``'Standing screen display size'`` / ``'Screen Size'`` (``'15.6 Inches'``) and
    ``'Max Screen Resolution'`` / ``'Screen Resolution'`` (``'1920 x 1080 Pixels'``,
    ``'1920x1080'``, ``'1024-by-768'``), falling back to the title (``'15.6"'``, ``'FHD'``).

    Returns
    -------
    dict with keys ``screen_in``, ``screen_w``, ``screen_h``.
    """
    details = details or {}
    screen_in = math.nan
    for key in ("Standing screen display size", "Screen Size", "Display Size", "Screen size"):
        val = details.get(key)
        if val is None:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)", str(val))
        if m:
            cand = float(m.group(1))
            if 6.0 <= cand <= 22.0:
                screen_in = cand
                break

    w = h = math.nan
    for key in ("Max Screen Resolution", "Screen Resolution", "Resolution", "Display Resolution Maximum"):
        val = details.get(key)
        if val is None:
            continue
        m = re.search(r"(\d{3,4})\s*(?:x|by|-by-|\*)\s*(\d{3,4})", str(val), re.I)
        if m:
            cw, ch = float(m.group(1)), float(m.group(2))
            if 640 <= cw <= 7680 and 400 <= ch <= 4800:
                w, h = cw, ch
                break

    t = title or ""
    if math.isnan(screen_in) and t:
        m = re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:\"|''|”|-?\s*inch(?:es)?\b)", t, re.I)
        if m:
            cand = float(m.group(1))
            if 6.0 <= cand <= 22.0:
                screen_in = cand
    if math.isnan(w) and t:
        m = re.search(r"\(?(\d{3,4})\s*x\s*(\d{3,4})\)?", t, re.I)
        if m:
            cw, ch = float(m.group(1)), float(m.group(2))
            if 640 <= cw <= 7680 and 400 <= ch <= 4800:
                w, h = cw, ch
        else:
            for pattern, (cw, ch) in _RES_KEYWORDS:
                if pattern.search(t):
                    w, h = cw, ch
                    break

    return {"screen_in": screen_in, "screen_w": w, "screen_h": h}


_OS_PATTERNS = [
    (re.compile(r"chrome ?os|chromebook", re.I), "ChromeOS"),
    (re.compile(r"\bmac ?os|\bos ?x\b|macbook|\bmacos\b", re.I), "macOS"),
    (re.compile(r"windows|\bwin ?(?:7|8|10|11|xp|vista)\b|\bwin ?\d+\b", re.I), "Windows"),
    (re.compile(r"\bubuntu\b|\blinux\b|\bmint\b|\bfedora\b|\bfreedos\b|\bdos\b", re.I), "Linux"),
    (re.compile(r"\bandroid\b", re.I), "Android"),
]


def parse_os(details: dict | None, title: str = "") -> str:
    """Return the OS family: Windows / macOS / ChromeOS / Linux / Android / Unknown."""
    details = details or {}
    for source in (str(details.get("Operating System") or ""), title or ""):
        if not source:
            continue
        for pattern, label in _OS_PATTERNS:
            if pattern.search(source):
                return label
    return "Unknown"


def min_plausible_weight_lb(screen_in: float) -> float:
    """Lightest a laptop of this screen size can credibly be, in pounds.

    Source rows sometimes carry an ounce-scale or shipping weight, which survives the
    unit conversion as a physically impossible value (a 15.6" machine at 0.72 lb). The
    floor scales with screen size because that is what actually bounds chassis mass.
    """
    if screen_in is None or (isinstance(screen_in, float) and math.isnan(screen_in)):
        return 0.5
    if screen_in < 12:
        return 1.5
    if screen_in < 14:
        return 1.8
    if screen_in < 16:
        return 2.2
    return 3.0


def parse_weight(details: dict | None, title: str = "") -> float:
    """Return the item weight in pounds (NaN when unknown or implausible)."""
    details = details or {}
    for key in ("Item Weight", "Weight", "Product Weight"):
        val = details.get(key)
        if val is None:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(pounds|lbs?|ounces|oz|kilograms|kg|grams|g)\b", str(val), re.I)
        if not m:
            continue
        v, unit = float(m.group(1)), m.group(2).lower()
        if unit in ("ounces", "oz"):
            v /= 16.0
        elif unit in ("kilograms", "kg"):
            v *= 2.20462
        elif unit in ("grams", "g"):
            v *= 0.00220462
        if 0.5 <= v <= 18:
            return v
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds)\b", title or "", re.I)
    if m:
        v = float(m.group(1))
        if 0.5 <= v <= 18:
            return v
    return math.nan


# --------------------------------------------------------------------------------------
# 4. Segment assignment
# --------------------------------------------------------------------------------------

_GAMING_TEXT_RE = re.compile(
    r"\b(gaming|gamer|" + "|".join(re.escape(h) for h in GAMING_BRAND_HINTS) + r")\b", re.I
)
_BUSINESS_TEXT_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in BUSINESS_SERIES_HINTS) + r")\b", re.I)
_ULTRA_TEXT_RE = re.compile(
    r"(" + "|".join(re.escape(h) for h in ULTRABOOK_SERIES_HINTS) + r"|\bultrabook\b|\bultra ?slim\b)", re.I
)
_BUDGET_CPUS = {"Celeron", "Pentium", "Atom", "A Series", "E Series", "Sempron", "MediaTek"}


def assign_segment(row: dict[str, Any] | pd.Series) -> str:
    """Assign one of :data:`config.SEGMENTS` to a parsed product row.

    Priority: chromebook -> gaming -> business -> ultrabook -> budget -> mainstream.
    Uses the hint lists from ``config.py`` on the title/series/brand plus parsed specs
    (discrete GPU, weight, RAM, CPU family, price when available).

    Parameters
    ----------
    row:
        Mapping with at least ``title``, ``brand``, ``os_family``, ``is_discrete_gpu``,
        ``ram_gb``, ``cpu_family``, ``weight_lb``, ``screen_in``, ``price``.

    Returns
    -------
    str
        A member of :data:`config.SEGMENTS`.
    """
    get = row.get if isinstance(row, dict) else row.get
    title = str(get("title", "") or "")
    brand = str(get("brand", "") or "")
    series = str(get("series_raw", "") or "")
    text = f"{brand} {series} {title}"

    os_family = str(get("os_family", "Unknown") or "Unknown")
    if os_family == "ChromeOS" or re.search(r"chromebook", text, re.I):
        return "chromebook"

    def _f(key: str) -> float:
        v = get(key, math.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return math.nan
        return v

    is_discrete = bool(get("is_discrete_gpu", False))
    gpu_model = str(get("gpu_model", "") or "")
    ram_gb = _f("ram_gb")
    weight = _f("weight_lb")
    price = _f("price")
    screen_in = _f("screen_in")
    cpu_family = str(get("cpu_family", "Unknown") or "Unknown")

    gaming_hint = bool(_GAMING_TEXT_RE.search(text))
    if is_discrete and (gaming_hint or re.search(r"\b(rtx|gtx)\b", gpu_model, re.I)):
        return "gaming"
    if gaming_hint and re.search(r"\bgaming\b", title, re.I) and is_discrete:
        return "gaming"

    if _BUSINESS_TEXT_RE.search(text):
        return "business"

    if _ULTRA_TEXT_RE.search(text):
        return "ultrabook"
    light = (not math.isnan(weight)) and weight <= 3.3
    premium = (not math.isnan(price) and price >= 900) or (
        cpu_family in {"Core i7", "Core i9", "Ryzen 7", "Ryzen 9", "Core Ultra 7", "Core Ultra 9", "M1", "M2", "M3"}
        and not math.isnan(ram_gb)
        and ram_gb >= 8
    )
    if light and premium and (math.isnan(screen_in) or screen_in <= 15.0) and not is_discrete:
        return "ultrabook"

    if (not math.isnan(price) and price < 400) or cpu_family in _BUDGET_CPUS or (
        not math.isnan(ram_gb) and ram_gb <= 4
    ):
        return "budget"

    return "mainstream"


# --------------------------------------------------------------------------------------
# 5. Variant de-duplication
# --------------------------------------------------------------------------------------

_MODEL_STOPWORDS = {
    "laptop", "notebook", "computer", "pc", "new", "newest", "latest", "premium", "flagship",
    "business", "home", "student", "professional", "pro", "touchscreen", "touch", "screen",
    "display", "led", "lcd", "ips", "fhd", "hd", "uhd", "qhd", "wuxga", "ssd", "hdd", "emmc",
    "ram", "memory", "storage", "intel", "amd", "core", "ryzen", "celeron", "pentium", "athlon",
    "nvidia", "geforce", "rtx", "gtx", "radeon", "graphics", "backlit", "keyboard", "wifi",
    "wi", "fi", "bluetooth", "hdmi", "webcam", "renewed", "refurbished", "certified", "black",
    "silver", "gray", "grey", "blue", "white", "red", "gold", "rose", "windows", "win", "with",
    "and", "the", "for", "inch", "inches", "thin", "light", "slim", "performance", "high",
    "gaming", "edition", "series", "model", "quad", "dual", "octa", "hexa", "cores", "core2",
    "bundle", "upgraded", "custom", "usb", "type", "dvd", "rw", "drive", "battery", "fast",
    "charging", "portable", "cheap", "office", "microsoft", "365",
}
_GENERIC_MODEL_TOKENS = {"laptop", "notebook", "book", "pc", "computer", "series", "model", "na", "n/a"}


def _model_signature(title: str, brand: str) -> str:
    """Build a coarse model signature from the title by stripping spec noise.

    ``'HP Envy 17t 17.3" Touchscreen FHD Laptop (i7-1260P, 64GB RAM, 2TB SSD)'``
    -> ``'hp envy 17t'``. Configuration-specific text (RAM, storage, clock speeds,
    colours, marketing words) is removed so that 8 GB and 16 GB builds of the same model
    produce the same signature.
    """
    t = (title or "").lower()
    t = t.split("(")[0]
    t = re.split(r"\||,|\s-\s|:", t)[0]
    t = re.sub(r"\d+(?:\.\d+)?\s*(?:gb|tb|mb|ghz|mhz|hz|wh|nits?)\b", " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)
    t = re.sub(r"\d+(?:\.\d+)?\s*(?:\"|''|”|inch(?:es)?)", " ", t)
    t = re.sub(r"\bi[3579][- ]?\d{3,5}[a-z]{0,2}\b", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    tokens = [tok for tok in t.split() if tok not in _MODEL_STOPWORDS and len(tok) > 1]
    brand_tokens = {b for b in re.sub(r"[^a-z0-9 ]", " ", brand.lower()).split()}
    tokens = [tok for tok in tokens if tok not in brand_tokens]
    tokens = tokens[:4]
    if not tokens:
        return ""
    if len(tokens) == 1 and (tokens[0] in _GENERIC_MODEL_TOKENS or len(tokens[0]) < 3):
        return ""
    return f"{brand.lower()}|" + " ".join(tokens)


def build_dedup_key(row: pd.Series) -> str:
    """Return the grouping key that collapses configuration variants of one model.

    The key is brand + model signature + the specs that define the *model* rather than
    the *configuration* (screen size, CPU family/tier, discrete-GPU flag, OS). RAM and
    storage are deliberately excluded so an 8 GB and a 16 GB build collapse together.
    Records without a usable model signature keep their own ``parent_asin`` as key so
    they are never merged blindly.
    """
    sig = row["model_sig"]
    if not sig:
        return f"__uniq__{row[PRODUCT_KEY]}"
    screen = row["screen_in"]
    screen_s = f"{screen:.1f}" if isinstance(screen, float) and not math.isnan(screen) else "na"
    tier = row["cpu_tier"]
    tier_s = f"{tier:.0f}" if isinstance(tier, float) and not math.isnan(tier) else "na"
    return "|".join(
        [
            sig,
            screen_s,
            str(row["cpu_family"]),
            tier_s,
            str(row["gpu_model"]),
            "d" if row["is_discrete_gpu"] else "i",
            str(row["os_family"]),
            "R" if row["is_renewed"] else "N",
        ]
    )


# --------------------------------------------------------------------------------------
# 6. Review text cleaning
# --------------------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]{1,80}>")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: Any) -> str:
    """Strip HTML tags/entities and control characters, collapse whitespace."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text)
    if "&" in s:
        s = html.unescape(s)
        if "&" in s:
            s = html.unescape(s)
    s = s.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    if "<" in s:
        s = _TAG_RE.sub(" ", s)
    s = _CTRL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


# --------------------------------------------------------------------------------------
# 7. I/O helpers
# --------------------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a (possibly very large) JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _coerce_price(value: Any) -> float:
    """Coerce the raw price field to a float; missing / nonsense values become NaN."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        return v if 20.0 <= v <= 20000.0 else math.nan
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", str(value))
    if not m:
        return math.nan
    v = float(m.group(1).replace(",", ""))
    return v if 20.0 <= v <= 20000.0 else math.nan


# --------------------------------------------------------------------------------------
# 8. Pipeline stages
# --------------------------------------------------------------------------------------


def load_products(path: Path = RAW_META) -> pd.DataFrame:
    """Load the raw meta JSONL, drop accessories/non-laptops and return a DataFrame.

    Returns a frame with the raw fields plus ``details`` kept as a dict column for the
    downstream parsing stage.
    """
    rows: list[dict] = []
    reasons: Counter[str] = Counter()
    n_raw = 0
    for rec in iter_jsonl(path):
        n_raw += 1
        title = rec.get("title") or ""
        drop, reason = is_accessory(title, rec.get("categories"))
        if drop:
            reasons[reason] += 1
            continue
        rows.append(
            {
                PRODUCT_KEY: rec.get("parent_asin"),
                "title": clean_text(title),
                "store": rec.get("store"),
                "details": rec.get("details") or {},
                "categories": rec.get("categories") or [],
                "price_raw": rec.get("price"),
                "average_rating": rec.get("average_rating"),
                "rating_number": rec.get("rating_number"),
            }
        )
    df = pd.DataFrame(rows)

    print(f"[load] raw meta records            : {n_raw:,}")
    for reason, cnt in reasons.most_common():
        print(f"[load]   dropped ({reason:<18}): {cnt:,}")
    print(f"[load] after accessory filter      : {len(df):,}")

    n_before = len(df)
    df = df[df[PRODUCT_KEY].notna()]
    df = df.drop_duplicates(subset=[PRODUCT_KEY], keep="first").reset_index(drop=True)
    if len(df) != n_before:
        print(f"[load]   dropped (bad/dupe asin  ): {n_before - len(df):,}")
    return df


def parse_specs(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the raw ``details`` dict + title into the typed spec columns."""
    parsed: list[dict[str, Any]] = []
    for details, title in zip(df["details"].tolist(), df["title"].tolist()):
        rec: dict[str, Any] = {}
        rec.update(parse_cpu(details, title))
        rec.update(parse_ram(details, title))
        rec.update(parse_storage(details, title))
        rec.update(parse_screen(details, title))
        rec.update(parse_gpu(details, title))
        rec["os_family"] = parse_os(details, title)
        weight = parse_weight(details, title)
        # Drop weights the chassis size makes impossible (ounce-scale / shipping weights).
        if not math.isnan(weight) and weight < min_plausible_weight_lb(rec.get("screen_in")):
            weight = math.nan
        rec["weight_lb"] = weight
        rec["series_raw"] = str(details.get("Series") or details.get("Model Name") or "")
        parsed.append(rec)
    specs = pd.DataFrame(parsed, index=df.index)
    return pd.concat([df, specs], axis=1)


def dedupe_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse configuration variants of the same model into one representative row.

    The representative is the variant with the most ratings (ties broken by review
    volume then by richer specs). ``n_variants`` records how many listings collapsed and
    ``variant_asins`` keeps the merged ``parent_asin`` values so the review table can be
    re-pointed at the survivor.
    """
    df = df.copy()
    df["model_sig"] = [
        _model_signature(t, b) for t, b in zip(df["title"].tolist(), df["brand"].tolist())
    ]
    df["dedup_key"] = df.apply(build_dedup_key, axis=1)

    df["_spec_richness"] = (
        df[["cpu_ghz", "ram_gb", "storage_gb", "screen_in", "weight_lb", "price"]].notna().sum(axis=1)
    )
    df["_rn"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0)

    order = df.sort_values(
        ["dedup_key", "_rn", "_spec_richness", PRODUCT_KEY], ascending=[True, False, False, True]
    )
    grouped = order.groupby("dedup_key", sort=False)
    keep = grouped.head(1).copy()

    counts = grouped.size()
    keep["n_variants"] = keep["dedup_key"].map(counts).astype("int64")

    variant_map: dict[str, str] = {}
    for key, asins in order.groupby("dedup_key", sort=False)[PRODUCT_KEY]:
        asin_list = asins.tolist()
        rep = asin_list[0]
        for a in asin_list[1:]:
            variant_map[a] = rep
    keep.attrs["variant_map"] = variant_map

    n_collapsed = len(df) - len(keep)
    multi = int((keep["n_variants"] > 1).sum())
    print(f"[dedup] products before            : {len(df):,}")
    print(f"[dedup] products after             : {len(keep):,}")
    print(f"[dedup] variant rows collapsed     : {n_collapsed:,} into {multi:,} model groups")
    if multi:
        print(f"[dedup] max variants in one group  : {int(keep['n_variants'].max())}")

    return keep.drop(columns=["_spec_richness", "_rn"]).reset_index(drop=True)


def load_reviews(keep_asins: set[str], variant_map: dict[str, str], path: Path = RAW_REVIEWS) -> pd.DataFrame:
    """Load, filter and clean the review dump.

    Reviews whose ``parent_asin`` was collapsed into a surviving variant are re-pointed
    at the survivor (the original id is kept in ``orig_parent_asin``) so that no genuine
    review text is thrown away, while every ``parent_asin`` in the output is guaranteed
    to exist in ``products.parquet``.
    """
    records: list[tuple] = []
    n_raw = 0
    n_no_product = 0
    n_empty = 0
    for rec in iter_jsonl(path):
        n_raw += 1
        pa = rec.get("parent_asin")
        orig = pa
        if pa not in keep_asins:
            pa = variant_map.get(pa)
            if pa is None or pa not in keep_asins:
                n_no_product += 1
                continue
        text = clean_text(rec.get("text"))
        if len(text) < 2:
            n_empty += 1
            continue
        title = clean_text(rec.get("title"))
        rating = rec.get("rating")
        records.append(
            (
                pa,
                orig,
                rec.get("asin"),
                rec.get("user_id"),
                float(rating) if rating is not None else math.nan,
                title,
                text,
                int(rec.get("helpful_vote") or 0),
                bool(rec.get("verified_purchase")),
                rec.get("timestamp"),
            )
        )

    rv = pd.DataFrame(
        records,
        columns=[
            PRODUCT_KEY,
            "orig_parent_asin",
            "asin",
            "user_id",
            "rating",
            "review_title",
            "text",
            "helpful_vote",
            "verified_purchase",
            "timestamp",
        ],
    )
    n_before = len(rv)
    rv = rv.drop_duplicates(subset=["user_id", "asin", "text"], keep="first")
    n_dupe = n_before - len(rv)

    rv["timestamp"] = pd.to_datetime(rv["timestamp"], unit="ms", errors="coerce")
    rv = rv[rv["timestamp"].notna()].reset_index(drop=True)
    rv["review_year"] = rv["timestamp"].dt.year.astype("int16")
    rv["text_len"] = rv["text"].str.len().astype("int32")

    print(f"[reviews] raw reviews              : {n_raw:,}")
    print(f"[reviews]   dropped (no product)   : {n_no_product:,}")
    print(f"[reviews]   dropped (empty text)   : {n_empty:,}")
    print(f"[reviews]   dropped (duplicates)   : {n_dupe:,}")
    print(f"[reviews] retained                 : {len(rv):,}")
    return rv


# --------------------------------------------------------------------------------------
# 9. Summary
# --------------------------------------------------------------------------------------

_FINAL_COLUMNS = [
    PRODUCT_KEY,
    "title",
    "brand",
    "store",
    "is_renewed",
    "cpu_brand",
    "cpu_family",
    "cpu_tier",
    "cpu_ghz",
    "ram_gb",
    "ram_type",
    "storage_gb",
    "storage_type",
    "screen_in",
    "screen_w",
    "screen_h",
    "gpu_brand",
    "gpu_model",
    "is_discrete_gpu",
    "os_family",
    "weight_lb",
    "price",
    "price_is_missing",
    "average_rating",
    "rating_number",
    "segment",
    "n_reviews",
    "n_variants",
]


def print_summary(products: pd.DataFrame, reviews: pd.DataFrame) -> None:
    """Print row counts, per-column coverage, segment/brand distributions, price coverage."""
    line = "=" * 78
    print(line)
    print("PIPELINE SUMMARY")
    print(line)
    print(f"products.parquet rows : {len(products):,}")
    print(f"reviews.parquet  rows : {len(reviews):,}")
    print(f"products with >=1 review: {(products['n_reviews'] > 0).sum():,}")
    print()

    print("-- per-column non-null coverage (products) --")
    n = len(products)
    for col in products.columns:
        s = products[col]
        if s.dtype == bool:
            nn = n
        elif pd.api.types.is_numeric_dtype(s):
            nn = int(s.notna().sum())
        else:
            nn = int((s.notna() & (s.astype("str").str.strip() != "") & (s.astype("str") != "Unknown")).sum())
        print(f"   {col:<18} {nn:>7,}  {nn / n * 100:6.2f}%")
    print()

    print("-- per-column non-null coverage (reviews) --")
    m = len(reviews)
    for col in reviews.columns:
        s = reviews[col]
        nn = m if s.dtype == bool else int(s.notna().sum())
        print(f"   {col:<18} {nn:>7,}  {nn / m * 100:6.2f}%")
    print()

    print("-- segment distribution --")
    seg = products["segment"].value_counts()
    for s in SEGMENTS:
        c = int(seg.get(s, 0))
        print(f"   {s:<12} {c:>7,}  {c / n * 100:6.2f}%")
    print()

    print("-- brand top-15 --")
    for b, c in products["brand"].value_counts().head(15).items():
        print(f"   {str(b):<22} {c:>7,}  {c / n * 100:6.2f}%")
    print()

    print("-- price coverage --")
    have = int(products["price"].notna().sum())
    print(f"   price present        {have:>7,}  {have / n * 100:6.2f}%")
    print(f"   price_is_missing     {n - have:>7,}  {(n - have) / n * 100:6.2f}%")
    if have:
        p = products["price"].dropna()
        print(
            f"   price median ${p.median():,.2f}  p10 ${p.quantile(0.10):,.2f}  "
            f"p90 ${p.quantile(0.90):,.2f}"
        )
    print()

    print("-- other sanity numbers --")
    print(f"   is_renewed True      {int(products['is_renewed'].sum()):>7,}")
    print(f"   is_discrete_gpu True {int(products['is_discrete_gpu'].sum()):>7,}")
    print(f"   n_variants > 1       {int((products['n_variants'] > 1).sum()):>7,}")
    print(f"   median ram_gb        {products['ram_gb'].median()}")
    print(f"   median storage_gb    {products['storage_gb'].median()}")
    print(f"   median screen_in     {products['screen_in'].median()}")
    print(line)


def run(meta_path: Path = RAW_META, reviews_path: Path = RAW_REVIEWS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline end-to-end and write both parquet files."""
    t0 = time.time()

    df = load_products(meta_path)

    # --- brand + renewed ---
    df["brand"] = [
        normalize_brand(s, d, t)
        for s, d, t in zip(df["store"].tolist(), df["details"].tolist(), df["title"].tolist())
    ]
    df["is_renewed"] = [
        detect_renewed(s, t, d)
        for s, t, d in zip(df["store"].tolist(), df["title"].tolist(), df["details"].tolist())
    ]
    n_reseller = int(
        pd.Series([str(s or "").lower() in RESELLER_STORES for s in df["store"].tolist()]).sum()
    )
    print(f"[brand] reseller stores repaired   : {n_reseller:,}")
    print(f"[brand] distinct brands            : {df['brand'].nunique():,}")
    print(f"[brand] renewed / refurbished      : {int(df['is_renewed'].sum()):,}")

    # --- price (never fabricated) ---
    df["price"] = [_coerce_price(v) for v in df["price_raw"].tolist()]
    df["price_is_missing"] = df["price"].isna()

    # --- specs ---
    df = parse_specs(df)

    # A >=19" panel is a desktop / all-in-one that slipped past the title filter.
    oversize = df["screen_in"] >= 19.0
    if int(oversize.sum()):
        print(f"[load]   dropped (oversize screen ): {int(oversize.sum()):,}")
        df = df[~oversize].reset_index(drop=True)
        print(f"[load] real laptops retained       : {len(df):,}")

    # --- dedup ---
    products = dedupe_variants(df)
    variant_map: dict[str, str] = products.attrs.get("variant_map", {})

    # --- segments ---
    products["segment"] = [assign_segment(r) for _, r in products.iterrows()]
    print("[segment] distribution:")
    for seg, cnt in products["segment"].value_counts().items():
        print(f"[segment]   {seg:<12} {cnt:>7,}")

    # --- reviews ---
    keep_asins = set(products[PRODUCT_KEY].tolist())
    reviews = load_reviews(keep_asins, variant_map, reviews_path)

    counts = reviews[PRODUCT_KEY].value_counts()
    products["n_reviews"] = products[PRODUCT_KEY].map(counts).fillna(0).astype("int64")

    # --- final typing / column order ---
    products["average_rating"] = pd.to_numeric(products["average_rating"], errors="coerce")
    products["rating_number"] = pd.to_numeric(products["rating_number"], errors="coerce").fillna(0).astype("int64")
    products["cpu_tier"] = pd.to_numeric(products["cpu_tier"], errors="coerce")
    for col in ("is_renewed", "is_discrete_gpu", "price_is_missing"):
        products[col] = products[col].astype(bool)
    for col in ("brand", "cpu_brand", "cpu_family", "ram_type", "storage_type", "gpu_brand", "gpu_model", "os_family"):
        products[col] = products[col].fillna("Unknown").astype("str")
    products["store"] = products["store"].astype("str")

    products = products[_FINAL_COLUMNS].copy()

    PRODUCTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    products.to_parquet(PRODUCTS_PARQUET, index=False)
    reviews.to_parquet(REVIEWS_PARQUET, index=False)
    print(f"[write] {PRODUCTS_PARQUET}  ({PRODUCTS_PARQUET.stat().st_size / 1e6:.1f} MB)")
    print(f"[write] {REVIEWS_PARQUET}  ({REVIEWS_PARQUET.stat().st_size / 1e6:.1f} MB)")
    print(f"[time]  {time.time() - t0:.1f}s")

    print_summary(products, reviews)
    return products, reviews


if __name__ == "__main__":
    run()
