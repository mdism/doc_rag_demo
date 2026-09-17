FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-install-project

COPY . .
RUN uv sync

RUN uv run python -c "from llama_index.readers.docling import DoclingReader; DoclingReader()" || true
RUN uv run python -c "from llama_index.core.postprocessor import SentenceTransformerRerank; SentenceTransformerRerank(model='cross-encoder/ms-marco-MiniLM-L-6-v2', top_n=3)" || true

ENTRYPOINT ["uv", "run", "crewai", "run"]