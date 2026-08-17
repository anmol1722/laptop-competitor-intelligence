"""End-to-end tests for the web API.

The load-bearing test here is ``test_no_nan_reaches_the_browser``: every endpoint's
payload is round-tripped through ``json.dumps(..., allow_nan=False)`` and then scanned
recursively for non-finite floats.  ``json.dumps`` emits the bare tokens ``NaN``,
``Infinity`` and ``-Infinity`` by default, none of which are valid JSON, and
``JSON.parse`` in the browser throws on all three - so a single unsanitised pandas cell
would break the UI at runtime rather than at build time.  This test makes that a
CI failure instead.

Run::

    .venv/bin/python src/web/test_api.py            # everything except the LLM
    .venv/bin/python src/web/test_api.py --chat     # also exercises POST /api/chat
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import app, sanitize  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSED.append(name)
    else:
        FAILED.append((name, detail))
        print(f"  FAIL {name}: {detail}")
    return bool(condition)


# --------------------------------------------------------------------------------------
# recursive NaN / Infinity scan
# --------------------------------------------------------------------------------------


def find_non_finite(obj, path: str = "$") -> list[str]:
    """Return the paths of every non-finite float in a decoded JSON payload."""
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += find_non_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += find_non_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(f"{path} = {obj!r}")
    return bad


def assert_json_safe(name: str, response) -> dict | list | None:
    """Status, strict re-serialisation, raw-token and recursive-value checks."""
    ok = check(f"{name}: status {response.status_code}",
               response.status_code < 400,
               f"body={response.text[:300]}")
    raw = response.text
    check(f"{name}: no bare NaN/Infinity token in the wire bytes",
          not any(tok in raw for tok in (":NaN", ": NaN", ":Infinity", ":-Infinity",
                                         "[NaN", " NaN,")),
          raw[:300])
    try:
        payload = response.json()                      # json.loads rejects bare NaN
    except Exception as exc:
        check(f"{name}: decodes as JSON", False, str(exc))
        return None
    check(f"{name}: decodes as JSON", True)
    try:
        json.dumps(payload, allow_nan=False)           # the actual round-trip contract
        check(f"{name}: json.dumps(allow_nan=False) round-trip", True)
    except (ValueError, TypeError) as exc:
        check(f"{name}: json.dumps(allow_nan=False) round-trip", False, str(exc))
    bad = find_non_finite(payload)
    check(f"{name}: no non-finite floats anywhere in the payload", not bad, "; ".join(bad[:5]))
    return payload if ok else payload


# --------------------------------------------------------------------------------------
# sanitizer unit tests
# --------------------------------------------------------------------------------------


def test_sanitizer() -> None:
    print("\n== sanitizer ==")
    payload = {
        "np_int": np.int64(7),
        "np_float": np.float32(1.5),
        "np_bool": np.bool_(True),
        "nan": float("nan"),
        "np_nan": np.float64("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "nat": pd.NaT,
        "pd_na": pd.NA,
        "ts": pd.Timestamp("2024-03-01T12:30:00"),
        "arr": np.array([1.0, np.nan, 3.0]),
        "series": pd.Series({"a": np.nan, "b": 2}),
        "frame": pd.DataFrame({"x": [1, np.nan], "y": ["a", None]}),
        "nested": {"deep": [{"v": np.float64("nan")}]},
        1: "int key",
        "bytes": b"hello",
        "td": pd.Timedelta(seconds=90),
    }
    clean = sanitize(payload)
    text = json.dumps(clean, allow_nan=False)          # must not raise
    check("sanitizer: dumps with allow_nan=False", True)
    check("sanitizer: NaN -> None", clean["nan"] is None and clean["np_nan"] is None)
    check("sanitizer: Inf -> None", clean["inf"] is None and clean["neg_inf"] is None)
    check("sanitizer: NaT/NA -> None", clean["nat"] is None and clean["pd_na"] is None)
    check("sanitizer: numpy scalars -> python",
          isinstance(clean["np_int"], int) and isinstance(clean["np_float"], float)
          and clean["np_bool"] is True)
    check("sanitizer: Timestamp -> ISO string", clean["ts"] == "2024-03-01T12:30:00")
    check("sanitizer: ndarray NaN -> None", clean["arr"] == [1.0, None, 3.0])
    check("sanitizer: Series -> dict", clean["series"] == {"a": None, "b": 2.0})
    check("sanitizer: DataFrame -> records",
          clean["frame"] == [{"x": 1.0, "y": "a"}, {"x": None, "y": None}],
          str(clean["frame"]))
    check("sanitizer: nested NaN", clean["nested"]["deep"][0]["v"] is None)
    check("sanitizer: int key -> str", "1" in clean)
    check("sanitizer: bytes -> str", clean["bytes"] == "hello")
    check("sanitizer: timedelta -> seconds", clean["td"] == 90.0)
    check("sanitizer: no NaN token in output", "NaN" not in text)


# --------------------------------------------------------------------------------------
# endpoint tests
# --------------------------------------------------------------------------------------


def pick_asins(client: TestClient) -> dict[str, str]:
    """Real asins from the live index: one popular+priced, one with sentiment, one without."""
    out: dict[str, str] = {}
    r = client.get("/api/products/search", params={"sort": "reviews_desc", "limit": 50})
    items = r.json()["items"]
    out["popular"] = items[0]["parent_asin"]
    out["priced"] = next(i["parent_asin"] for i in items if i["price_available"])
    out["with_sentiment"] = next(i["parent_asin"] for i in items if i["has_sentiment"])
    r2 = client.get("/api/products/search",
                    params={"has_sentiment": False, "sort": "reviews_desc", "limit": 5})
    out["no_sentiment"] = r2.json()["items"][0]["parent_asin"]
    # a genuinely unpriced product (~69% of the catalogue) so "price not listed" is tested
    for params in ({"sort": "reviews_desc", "limit": 200},
                   {"sort": "title_asc", "limit": 200},
                   {"q": "laptop", "sort": "rating_desc", "limit": 200}):
        page = client.get("/api/products/search", params=params).json()["items"]
        unpriced = [i["parent_asin"] for i in page if not i["price_available"]]
        if unpriced:
            out["unpriced"] = unpriced[0]
            break
    return out


def test_endpoints(client: TestClient, asins: dict[str, str], with_chat: bool) -> None:
    print("\n== health ==")
    h = assert_json_safe("GET /api/health", client.get("/api/health"))
    check("health: products row count > 0", (h or {}).get("rows", {}).get("products", 0) > 0)
    check("health: artifacts listed", len((h or {}).get("artifacts", [])) >= 4)
    check("health: price coverage reported",
          "coverage" in (h or {}) and "price" in h["coverage"])
    check("health: module status for all backend modules",
          set(["config", "pricing", "matching", "sentiment", "rag"])
          <= set((h or {}).get("modules", {}).keys()))

    print("\n== search ==")
    cases = [
        ("plain", {}),
        ("text", {"q": "thinkpad"}),
        ("text tokens", {"q": "16gb rtx gaming"}),
        ("nonsense text", {"q": "zzzzqqqxxx"}),
        ("brand", {"brand": "Dell"}),
        ("segment", {"segment": "gaming"}),
        ("price range", {"min_price": 500, "max_price": 1500}),
        ("min_ram", {"min_ram": 16}),
        ("discrete gpu", {"has_discrete_gpu": True}),
        ("combo", {"q": "laptop", "segment": "ultrabook,business", "min_ram": 8,
                   "has_discrete_gpu": False, "sort": "price_desc", "limit": 5}),
        ("paging", {"limit": 5, "offset": 20}),
        ("sentiment only", {"has_sentiment": True, "limit": 3}),
    ]
    for label, params in cases:
        r = client.get("/api/products/search", params=params)
        p = assert_json_safe(f"GET /api/products/search [{label}]", r)
        if p:
            check(f"search[{label}]: has total/items", "total" in p and "items" in p)
            for it in p["items"]:
                if not it["price_available"] and it["price_display"] != "price not listed":
                    check(f"search[{label}]: missing price shown honestly", False,
                          it["price_display"])
                    break
            else:
                check(f"search[{label}]: missing price shown honestly", True)

    for sort in ("relevance", "price_asc", "price_desc", "rating_desc",
                 "reviews_desc", "ram_desc", "title_asc"):
        p = assert_json_safe(f"GET /api/products/search [sort={sort}]",
                             client.get("/api/products/search",
                                        params={"q": "laptop", "sort": sort, "limit": 5}))
        if p and sort == "price_asc":
            prices = [i["price"] for i in p["items"] if i["price"] is not None]
            check("search: price_asc is non-decreasing", prices == sorted(prices), str(prices))

    print("\n== search: bad params -> 422 ==")
    for label, params in [
        ("limit=0", {"limit": 0}),
        ("limit=9999", {"limit": 9999}),
        ("offset=-1", {"offset": -1}),
        ("bad sort", {"sort": "banana"}),
        ("bad segment", {"segment": "spaceship"}),
        ("min>max price", {"min_price": 900, "max_price": 100}),
        ("negative price", {"min_price": -5}),
    ]:
        r = client.get("/api/products/search", params=params)
        check(f"search 422 [{label}]", r.status_code == 422,
              f"got {r.status_code}: {r.text[:160]}")
        assert_json_safe(f"search 422 body [{label}]",
                         type("R", (), {"status_code": 200, "text": r.text,
                                        "json": r.json})())

    print("\n== product detail ==")
    for label, asin in asins.items():
        p = assert_json_safe(f"GET /api/products/{{{label}}}", client.get(f"/api/products/{asin}"))
        if not p:
            continue
        check(f"detail[{label}]: price_position present", "price_position" in p)
        check(f"detail[{label}]: no_sentiment_data flag present", "no_sentiment_data" in p)
        check(f"detail[{label}]: sentiment consistency",
              (p["sentiment"] is None) == p["no_sentiment_data"])
        check(f"detail[{label}]: review counts present",
              {"n_reviews_retained", "rating_number", "n_reviews_scored"} <= set(p["reviews"]))
        if not p["price_available"]:
            check(f"detail[{label}]: unpriced shows 'price not listed'",
                  p["price_display"] == "price not listed" and p["price"] is None)
        vs = p["price_position"].get("vs_segment", {})
        check(f"detail[{label}]: segment stat carries n + coverage",
              "n" in vs and "coverage" in vs, str(list(vs)[:8]))

    r = client.get("/api/products/NOSUCHASIN1")
    check("detail: unknown asin -> 404", r.status_code == 404, f"got {r.status_code}")
    check("detail: 404 body is structured JSON", "error" in r.json(), r.text[:200])

    print("\n== competitors ==")
    asin = asins["popular"]
    for label, params in [("guard on", {"k": 8, "guard": True}),
                          ("guard off", {"k": 8, "guard": False}),
                          ("k=1", {"k": 1}),
                          ("no brand cap", {"k": 10, "max_per_brand": 50}),
                          ("cross-brand", {"k": 5, "exclude_same_brand": True}),
                          ("specs only", {"k": 5, "text_weight": 0.0})]:
        p = assert_json_safe(f"GET /competitors [{label}]",
                             client.get(f"/api/products/{asin}/competitors", params=params))
        if not p:
            continue
        check(f"competitors[{label}]: count matches k",
              p["count"] == min(params.get("k", 10), p["count"]) and p["count"] > 0)
        first = p["competitors"][0]
        check(f"competitors[{label}]: carries price + specs + rating + score",
              {"price", "specs", "average_rating", "score", "similarity"} <= set(first))
        check(f"competitors[{label}]: similarity components present",
              {"text_sim", "spec_sim", "score"} <= set(first["similarity"]))

    on = client.get(f"/api/products/{asin}/competitors", params={"k": 10, "guard": True}).json()
    off = client.get(f"/api/products/{asin}/competitors", params={"k": 10, "guard": False}).json()
    ids_on = [c["parent_asin"] for c in on["competitors"]]
    ids_off = [c["parent_asin"] for c in off["competitors"]]
    check("competitors: guard toggle changes the ranking (ablation works)",
          ids_on != ids_off or on["coverage"] != off["coverage"],
          f"on={ids_on[:3]} off={ids_off[:3]}")
    check("competitors: unknown asin -> 404",
          client.get("/api/products/NOSUCHASIN1/competitors").status_code == 404)
    check("competitors: k=0 -> 422",
          client.get(f"/api/products/{asin}/competitors", params={"k": 0}).status_code == 422)
    check("competitors: text_weight=2 -> 422",
          client.get(f"/api/products/{asin}/competitors",
                     params={"text_weight": 2}).status_code == 422)

    print("\n== reviews ==")
    p = assert_json_safe("GET /reviews [with sentiment]",
                         client.get(f"/api/products/{asins['with_sentiment']}/reviews",
                                    params={"k": 3}))
    if p:
        check("reviews: no_sentiment_data is False", p["no_sentiment_data"] is False)
        check("reviews: aspects returned", len(p["aspects"]) > 0)
        has_snip = any(a.get("praises") or a.get("complaints") for a in p["aspects"])
        check("reviews: at least one verbatim snippet", has_snip)
        for a in p["aspects"]:
            for s in (a.get("praises", []) + a.get("complaints", [])):
                check("reviews: snippet carries text + polarity + aspect",
                      {"snippet", "polarity", "aspect"} <= set(s))
                break
            break

    p = assert_json_safe("GET /reviews [no sentiment]",
                         client.get(f"/api/products/{asins['no_sentiment']}/reviews"))
    if p:
        check("reviews: uncovered product flags no_sentiment_data",
              p["no_sentiment_data"] is True and p["sentiment_note"])
        check("reviews: uncovered product returns no fake aspects", p["aspects"] == [])

    assert_json_safe("GET /reviews [single aspect]",
                     client.get(f"/api/products/{asins['with_sentiment']}/reviews",
                                params={"aspect": "battery", "k": 2}))
    check("reviews: bad aspect -> 422",
          client.get(f"/api/products/{asins['with_sentiment']}/reviews",
                     params={"aspect": "smell"}).status_code == 422)
    check("reviews: unknown asin -> 404",
          client.get("/api/products/NOSUCHASIN1/reviews").status_code == 404)

    print("\n== segments / brands / market ==")
    p = assert_json_safe("GET /api/segments", client.get("/api/segments"))
    if p:
        check("segments: one row per segment", len(p["rows"]) >= 5)
        for r in p["rows"]:
            if not ({"n_products", "n_priced", "coverage", "reliable"} <= set(r)):
                check("segments: every row carries n + coverage + reliable", False, str(list(r)))
                break
        else:
            check("segments: every row carries n + coverage + reliable", True)

    p = assert_json_safe("GET /api/brands", client.get("/api/brands", params={"limit": 25}))
    if p:
        check("brands: unreliable flag preserved",
              all("unreliable" in r and "reliable" in r for r in p["rows"]))
        check("brands: n_unreliable reported", "n_unreliable" in p)
    assert_json_safe("GET /api/brands [min_products=50]",
                     client.get("/api/brands", params={"min_products": 50, "sort": "median",
                                                       "desc": False}))
    assert_json_safe("GET /api/brands [q filter]",
                     client.get("/api/brands", params={"q": "len"}))
    check("brands: limit=0 -> 422", client.get("/api/brands",
                                               params={"limit": 0}).status_code == 422)

    p = assert_json_safe("GET /api/market/overview", client.get("/api/market/overview"))
    if p:
        check("market: catalogue block", p["catalogue"]["n_products"] > 0)
        check("market: price block carries n + coverage",
              "coverage" in p["price"] and "n" in p["price"])
        check("market: sentiment coverage reported", "coverage" in p["sentiment"])
        check("market: caveats present", len(p["caveats"]) >= 2)

    print("\n== chat status ==")
    p = assert_json_safe("GET /api/chat/status", client.get("/api/chat/status"))
    if p:
        check("chat status: state is one of the four",
              p["state"] in ("unloaded", "loading", "ready", "unavailable"), p["state"])
        check("chat status: exposes accepting_input for the UI", "accepting_input" in p)
        check("chat status: reports GPU memory", "gpu" in p)

    print("\n== chat body validation ==")
    for label, body in [("blank", {"question": ""}),
                        ("missing", {}),
                        ("too many tokens", {"question": "hello there", "max_new_tokens": 99999})]:
        r = client.post("/api/chat", json=body)
        check(f"chat 422 [{label}]", r.status_code == 422, f"got {r.status_code}")

    if with_chat:
        print("\n== chat (LLM) ==")
        t0 = time.time()
        r = client.post("/api/chat", json={
            "question": "Which gaming laptops under $1500 do reviewers rate best on "
                        "thermals and fan noise?",
            "max_new_tokens": 300,
        })
        dt = time.time() - t0
        p = assert_json_safe("POST /api/chat", r)
        if r.status_code == 200 and p:
            check("chat: answer text returned", bool(p["answer"]))
            check("chat: evidence list returned", isinstance(p["evidence"], list))
            check("chat: audit block returned",
                  {"grounded", "unsupported_markers", "unverified_numbers",
                   "misattributed_reviews"} <= set(p["audit"]))
            check("chat: latency_s reported", isinstance(p["latency_s"], float))
            print(f"  chat wall time {dt:.1f}s, latency_s={p['latency_s']}, "
                  f"evidence={p['n_evidence']}, grounded={p['audit']['grounded']}")
        else:
            check("chat: structured error (not a stack trace)",
                  isinstance(p, dict) and "error" in p, str(p)[:300])
            print(f"  chat unavailable -> {r.status_code}: {str(p)[:200]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true", help="also exercise the LLM chat endpoint")
    args = ap.parse_args(argv)

    test_sanitizer()
    with TestClient(app) as client:              # triggers the lifespan startup
        asins = pick_asins(client)
        print(f"\nusing asins: {asins}")
        test_endpoints(client, asins, with_chat=args.chat)

    print(f"\n{'=' * 70}\npassed {len(PASSED)}, failed {len(FAILED)}")
    for name, detail in FAILED:
        print(f"  FAILED {name}: {detail[:200]}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
