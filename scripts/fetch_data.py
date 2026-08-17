"""Stream-filter the Amazon Reviews 2023 Electronics dumps down to laptops only.

The upstream files are large (meta 5.25 GB, reviews 22.6 GB) but we only need the
laptop slice, so we never store them. Each pass streams over HTTP and writes only
matching lines to data/raw/.

Pass 1: meta_Electronics.jsonl  -> laptops_meta.jsonl   (+ the parent_asin set)
Pass 2: Electronics.jsonl       -> laptops_reviews.jsonl (reviews for those asins)

Stdlib only, so it can run before the venv exists. Resumes on connection drops
via HTTP Range requests, which matters for the 22 GB pass.
"""

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
META_URL = f"{BASE}/meta_categories/meta_Electronics.jsonl"
REVIEW_URL = f"{BASE}/review_categories/Electronics.jsonl"

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
META_OUT = os.path.join(RAW, "laptops_meta.jsonl")
REVIEW_OUT = os.path.join(RAW, "laptops_reviews.jsonl")
ASIN_OUT = os.path.join(RAW, "laptop_asins.txt")

MAX_RETRIES = 8


def stream_lines(url, start_byte=0):
    """Yield (line_bytes, absolute_offset_after_line) from url, resuming on drops."""
    pos = start_byte
    retries = 0
    buf = b""
    while True:
        req = urllib.request.Request(url)
        if pos:
            req.add_header("Range", f"bytes={pos}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # A resumed request that comes back 200 means the server ignored
                # Range; we'd otherwise silently re-read from zero and duplicate.
                if pos and resp.status != 206:
                    raise RuntimeError(f"expected 206 on resume, got {resp.status}")
                retries = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        if buf:
                            yield buf, pos + len(buf)
                        return
                    buf += chunk
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        line = buf[:nl]
                        buf = buf[nl + 1:]
                        pos += nl + 1
                        yield line, pos
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError, OSError) as e:
            retries += 1
            if retries > MAX_RETRIES:
                raise
            # Drop the partial line; we resume from the last complete newline.
            buf = b""
            wait = min(30, 2 ** retries)
            print(f"  [retry {retries}/{MAX_RETRIES}] {type(e).__name__}: {e} -- resuming at byte {pos} in {wait}s",
                  flush=True)
            time.sleep(wait)


def is_laptop(rec):
    """Real laptops sit under Computers & Tablets > Laptops.

    Accessories (skins, bags, decals) sit under 'Laptop Accessories', so matching
    'Laptops' as an exact list element separates them cleanly.
    """
    cats = rec.get("categories") or []
    return "Laptops" in cats and "Laptop Accessories" not in cats


def pass1():
    print(f"PASS 1: streaming meta -> {META_OUT}", flush=True)
    seen = 0
    kept = 0
    asins = set()
    t0 = time.time()
    with open(META_OUT, "w", encoding="utf-8") as out:
        for line, pos in stream_lines(META_URL):
            if not line.strip():
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_laptop(rec):
                kept += 1
                pa = rec.get("parent_asin")
                if pa:
                    asins.add(pa)
                out.write(json.dumps(rec) + "\n")
            if seen % 200000 == 0:
                mb = pos / 1e6
                print(f"  meta: {seen:,} rows | {kept:,} laptops | {mb:,.0f} MB | {time.time()-t0:,.0f}s",
                      flush=True)
    with open(ASIN_OUT, "w", encoding="utf-8") as f:
        for a in sorted(asins):
            f.write(a + "\n")
    print(f"PASS 1 done: {seen:,} rows scanned, {kept:,} laptops, {len(asins):,} asins, {time.time()-t0:,.0f}s",
          flush=True)
    return asins


def pass2(asins):
    print(f"PASS 2: streaming reviews -> {REVIEW_OUT} ({len(asins):,} target asins)", flush=True)
    seen = 0
    kept = 0
    t0 = time.time()
    with open(REVIEW_OUT, "w", encoding="utf-8") as out:
        for line, pos in stream_lines(REVIEW_URL):
            if not line.strip():
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("parent_asin") in asins:
                kept += 1
                out.write(json.dumps(rec) + "\n")
            if seen % 1000000 == 0:
                mb = pos / 1e6
                print(f"  reviews: {seen:,} rows | {kept:,} kept | {mb:,.0f} MB | {time.time()-t0:,.0f}s",
                      flush=True)
    print(f"PASS 2 done: {seen:,} reviews scanned, {kept:,} kept, {time.time()-t0:,.0f}s", flush=True)


def main():
    os.makedirs(RAW, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    asins = None
    if which in ("all", "meta"):
        asins = pass1()
    if which in ("all", "reviews"):
        if asins is None:
            with open(ASIN_OUT, encoding="utf-8") as f:
                asins = {l.strip() for l in f if l.strip()}
        pass2(asins)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
