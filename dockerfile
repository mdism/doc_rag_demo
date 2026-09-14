FROM python:3.12-slim

# System dependencies: git (some Python packages install from source refs),
# build-essential (compiles native extensions used by torch/sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv -- fast Python package manager, matches local dev setup
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first so Docker can cache this layer -- rebuilding
# only reinstalls packages when pyproject.toml actually changes, not on
# every code edit.
COPY pyproject.toml ./
RUN uv sync --no-install-project

# Now copy the rest of the application code
COPY . .
RUN uv sync

# Pre-download the two models that would otherwise download on first run
# (Docling's layout model, the reranker's cross-encoder) -- bakes them into
# the image so container startup doesn't hang on a Hugging Face download.
RUN uv run python -c "from llama_index.readers.docling import DoclingReader; DoclingReader()" || true
RUN uv run python -c "from llama_index.core.postprocessor import SentenceTransformerRerank; SentenceTransformerRerank(model='cross-encoder/ms-marco-MiniLM-L-6-v2', top_n=3)" || true

ENTRYPOINT ["uv", "run", "crewai", "run"]