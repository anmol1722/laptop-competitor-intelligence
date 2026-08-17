"""Shared paths, schema contract and segment definitions.

Every module imports from here so the column names stay consistent across the
pipeline, the matching index, the sentiment pass, the RAG agent and the app.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVAL = ROOT / "eval"
REPORT = ROOT / "report"

for _d in (PROCESSED, EVAL, REPORT):
    _d.mkdir(parents=True, exist_ok=True)

# --- raw inputs (produced by scripts/fetch_data.py) ---
RAW_META = RAW / "laptops_meta.jsonl"
RAW_REVIEWS = RAW / "laptops_reviews.jsonl"

# --- pipeline outputs ---
PRODUCTS_PARQUET = PROCESSED / "products.parquet"
REVIEWS_PARQUET = PROCESSED / "reviews.parquet"

# --- module artifacts ---
EMBEDDINGS_NPY = PROCESSED / "product_embeddings.npy"
EMBED_IDS_JSON = PROCESSED / "product_embedding_ids.json"
REVIEW_SENTIMENT_PARQUET = PROCESSED / "review_sentiment.parquet"
PRODUCT_SENTIMENT_PARQUET = PROCESSED / "product_sentiment.parquet"

# --- models ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
LLM_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# --- products.parquet schema contract ---
# Identity
#   parent_asin (str, unique key)  title (str)  brand (str, normalized)
#   store (str, raw)  is_renewed (bool)
# Parsed specs
#   cpu_brand (str)      e.g. Intel / AMD / Apple / Qualcomm / Unknown
#   cpu_family (str)     e.g. Core i7 / Ryzen 7 / Celeron / M1
#   cpu_tier (int)       3/5/7/9 for i3/i5/i7/i9 & Ryzen, else NaN
#   cpu_ghz (float)
#   ram_gb (float)       ram_type (str)  e.g. DDR4
#   storage_gb (float)   storage_type (str) SSD/HDD/eMMC/Unknown
#   screen_in (float)    screen_w (float)  screen_h (float)
#   gpu_brand (str)      NVIDIA / AMD / Intel / Apple / Unknown
#   gpu_model (str)      e.g. RTX 4060
#   is_discrete_gpu (bool)
#   os_family (str)      Windows / macOS / ChromeOS / Linux / Unknown
#   weight_lb (float)
# Market
#   price (float, may be NaN)  price_is_missing (bool)
#   average_rating (float)  rating_number (int)
#   segment (str)  one of SEGMENTS
#   n_reviews (int)  actual reviews retained for this product
PRODUCT_KEY = "parent_asin"

SEGMENTS = ["gaming", "ultrabook", "business", "budget", "mainstream", "chromebook"]

# --- review aspects for aspect-level mining ---
ASPECTS = {
    "performance": ["fast", "speed", "slow", "lag", "performance", "processor", "cpu",
                    "ram", "boot", "responsive", "sluggish", "freeze", "powerful"],
    "battery": ["battery", "charge", "charging", "unplugged", "battery life", "power adapter"],
    "display": ["screen", "display", "monitor", "resolution", "brightness", "glare",
                "colors", "viewing angle", "dim", "bright"],
    "keyboard_trackpad": ["keyboard", "keys", "trackpad", "touchpad", "typing", "mouse pad",
                          "backlit", "key travel"],
    "build_quality": ["build", "flex", "flimsy", "sturdy", "plastic", "hinge", "cheap feel",
                      "solid", "durable", "creak"],
    "thermals_noise": ["hot", "heat", "overheat", "fan", "noisy", "loud", "thermal", "cooling",
                       "burning", "warm"],
    "value_price": ["price", "value", "worth", "cheap", "expensive", "bargain", "money",
                    "affordable", "overpriced"],
    "support_service": ["support", "customer service", "warranty", "return", "rma",
                        "tech support", "refund"],
}

# --- segment classification hints (applied in pipeline.py) ---
GAMING_BRAND_HINTS = ["alienware", "razer", "rog", "predator", "nitro", "omen", "legion",
                      "tuf", "msi", "sager", "gigabyte", "aorus"]
BUSINESS_SERIES_HINTS = ["thinkpad", "latitude", "elitebook", "probook", "precision",
                         "zbook", "travelmate", "vostro", "toughbook"]
ULTRABOOK_SERIES_HINTS = ["xps", "spectre", "yoga", "swift", "zenbook", "macbook air",
                          "surface laptop", "envy", "gram"]
