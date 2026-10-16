FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARROW_DEFAULT_MEMORY_POOL=system \
    LLM_PROVIDER=groq \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app
COPY requirements-web.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-web.txt \
    && useradd --create-home --uid 1000 app

COPY src/ ./src/
COPY data/processed/products.parquet data/processed/product_sentiment.parquet data/processed/review_sentiment.parquet data/processed/product_embeddings.npy data/processed/product_embedding_ids.json ./data/processed/
RUN mkdir -p eval report /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache

USER app
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn src.web.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
