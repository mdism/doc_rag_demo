# Document Search Platform — Agentic RAG Backend

A document search platform with an **Agentic RAG** backend, exposing REST APIs that integrate
with **OpenWebUI** as the chat interface. Built with Docling, LlamaIndex, CrewAI, PostgreSQL +
PGVector, Ollama, Arize Phoenix, and RAGAs.

---

## 1. Architecture Overview

```
                         ┌─────────────┐
                         │  OpenWebUI  │  (chat interface)
                         └──────┬──────┘
                                │  OpenAI-compatible API
                                ▼
                         ┌─────────────┐
                         │   FastAPI   │  app.py
                         │  (this repo)│
                         └──────┬──────┘
                                │
                 ┌──────────────┼───────────────┐
                 ▼              ▼               ▼
          ┌────────────┐ ┌────────────┐  ┌────────────┐
          │  Ingestion │ │  CrewAI    │  │  Phoenix   │
          │  Pipeline  │ │  Crew      │  │  Tracing   │
          │ (ingest.py)│ │ (crew.py)  │  │(instrument-│
          └─────┬──────┘ └─────┬──────┘  │  ation.py) │
                │              │         └────────────┘
                ▼              ▼
       ┌─────────────────────────────┐
       │        PostgreSQL           │
       │      + PGVector             │
       └─────────────────────────────┘
                │              │
                ▼              ▼
       ┌─────────────────────────────┐
       │   Ollama (LLM + Embeddings) │
       │  local / Kaggle+ngrok /     │
       │  vast.ai (remote GPU)       │
       └─────────────────────────────┘
```

**Ingestion pipeline** (`ingest.py`): PDF → Docling (structure-aware parsing) → markdown-heading
split → table-aware resize pass (merges undersized chunks, splits oversized ones while never
cutting through a markdown table) → batched embedding via Ollama → stored in PGVector.

**Retrieval pipeline** (`tools/retriever_tool.py`): query → embed → PGVector similarity search
(top 10, fast/approximate) → cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`, top 3,
slow/precise) → returned to the agent as scored, source-attributed passages.

**Agentic layer** (`crew.py`): two CrewAI agents running sequentially —
1. **Document Retrieval Specialist** — decides what to search for, calls the retriever tool,
   can re-search if results look insufficient.
2. **Answer Synthesizer** — writes the final answer strictly from retrieved context, citing
   exact figures and source sections, and explicitly says so when context is insufficient
   rather than guessing.

This retrieve-then-verify loop (rather than a single fixed retrieval + generation pass) is
what makes the pipeline **agentic**, not just RAG.

---

## 2. Tech Stack (mandatory tools)

| Tool | Role in this project |
|---|---|
| **Docling** | PDF parsing — layout-aware extraction (headings, tables) into markdown |
| **PostgreSQL + PGVector** | Vector store for chunk embeddings |
| **LlamaIndex** | Node parsing, embedding orchestration, vector store integration, reranking |
| **CrewAI** | Multi-agent orchestration (Retriever + Synthesizer) |
| **Ollama** | LLM + embedding inference, local or remote (Kaggle/vast.ai GPU) |
| **Arize Phoenix** | OpenTelemetry-based tracing of every agent/tool/LLM call |
| **RAGAs** | Faithfulness, Answer Relevancy, Context Precision, Context Recall scoring |
| **OpenWebUI** | Chat frontend, connected via an OpenAI-compatible API |
| **FastAPI** | REST API layer, Swagger/OpenAPI docs at `/docs` |

---

## 3. Project Structure

```
doc_rag_demo/
├── app.py                    # FastAPI app: /ask, /ingest, /v1/models, /v1/chat/completions
├── crew.py                   # CrewAI crew definition, dynamic per-request model selection
├── ingest.py                 # Ingestion pipeline (also runnable standalone via CLI)
├── evaluate.py                # RAGAs evaluation script
├── instrumentation.py        # Phoenix/OpenTelemetry tracing setup
├── tools/
│   └── retriever_tool.py     # CrewAI tool: vector search + reranking
├── config/
│   ├── agents.yaml           # Agent role/goal/backstory (externalized prompts)
│   └── tasks.yaml            # Task descriptions (externalized prompts)
├── sample_docs/               # Ingested source PDFs
├── rag_test_questions.xlsx   # Manual test set: question, expected answer, actual answer
├── ragas_results_*.csv       # RAGAs evaluation output, per model configuration
├── docker-compose.yml        # Postgres + PGVector, Phoenix
├── Dockerfile                 # Experimental: containerizes the FastAPI app itself (see §9)
├── pyproject.toml
└── .env                        # Local configuration (not committed — see §4)
```

---

## 4. Configuration (`.env`)

All configuration is environment-variable driven, with sensible localhost defaults so the
project runs identically whether every service is local, or Ollama is remote.

```dotenv
# Postgres / PGVector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Ollama -- point this at local, Kaggle+ngrok, or a rented GPU (see §7)
OLLAMA_BASE_URL=http://localhost:11434

# Which model the CrewAI agents use by default
# (OpenWebUI's model picker can override this per-request -- see §6)
AGENT_LLM_MODEL=ollama/llama3.1:8b

# Phoenix tracing endpoint
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
```

---

## 5. Setup & Installation

### 5.1 Prerequisites
- Python 3.10–3.13, [`uv`](https://docs.astral.sh/uv/)
- Docker Desktop
- Ollama (installed locally, or accessible remotely — see §7)

> **Windows + OneDrive note:** if your project folder is inside a OneDrive-synced path, `uv`
> may fail to install packages with a hardlink error. Fix once, permanently, by setting the
> environment variable `UV_LINK_MODE=copy` (Windows Environment Variables settings).

### 5.2 Start backing services

```bash
docker compose up -d
```

This starts Postgres (with PGVector pre-installed) and Phoenix. On first run, enable the
PGVector extension:

```bash
docker exec -it rag-postgres psql -U postgres -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5.3 Install Python dependencies

```bash
uv sync
```

### 5.4 Pull required Ollama models

```bash
ollama pull llama3.1:8b
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

(Or on a remote host — see §7 — run these there instead.)

### 5.5 Ingest the sample documents

```bash
uv run python ingest.py sample_docs/employee_handbook.pdf
uv run python ingest.py sample_docs/MasterServicesAgreement.pdf
```

Or, once the API is running (§5.6), ingest via HTTP:

```bash
curl -X POST http://localhost:8000/ingest -F "file=@sample_docs/your_document.pdf"
```

### 5.6 Run the API

```bash
uv run uvicorn app:app --reload
```

- Swagger / OpenAPI docs: **http://localhost:8000/docs**
- Health check: `GET /health`

### 5.7 Connect OpenWebUI

```bash
docker run -d -p 3000:8080 --name openwebui --add-host=host.docker.internal:host-gateway ghcr.io/open-webui/open-webui:main
```

Open `http://localhost:3000` → **Settings → Connections → OpenAI API**:
- Base URL: `http://host.docker.internal:8000/v1`
- API Key: any placeholder value (not validated)

Two models will appear in the chat model picker — see §6.

> **Recommended:** In **Settings → Interface**, disable "Follow-Up Generation" (and similar
> auto-generation toggles). OpenWebUI sends these as internal background requests through the
> same chat endpoint; the API detects and short-circuits them automatically (see §8), but
> disabling at the source avoids the wasted round-trip entirely.

---

## 6. Model Selection

The agent's LLM is configurable two ways:

1. **Default, via `.env`**: `AGENT_LLM_MODEL=ollama/llama3.1:8b`
2. **Per-request, via OpenWebUI's model picker**: the API exposes two named variants —
   `doc-rag-assistant-llama3.1` and `doc-rag-assistant-gemma3` — selectable directly in chat,
   no restart needed. Add more variants by editing `AVAILABLE_MODELS` in `app.py`.

**Why both models are supported, and a real compatibility issue this surfaced:** Gemma 3 has
no native tool-calling support on any tag, unlike Llama 3.1. CrewAI's auto-detection of this
is unreliable (a known upstream issue), so `crew.py` explicitly registers Gemma 3 as
non-tool-calling via `litellm.register_model(...)`, forcing the correct text-based (ReAct)
tool-calling path rather than a native function-call attempt that would otherwise fail with a
400 error.

---

## 7. Running Ollama Remotely (GPU acceleration)

Local CPU inference works but is slow (agentic runs can take 10–20+ minutes in the worst
case). Two working remote-GPU setups were used during development and testing:

### 7.1 Kaggle (free, T4 GPU)
Kaggle provides free GPU notebooks but no direct public networking, so a tunnel is required:

```python
import os
os.environ["OLLAMA_ORIGINS"] = "*"      # Ollama rejects non-localhost Host headers by default
os.environ["OLLAMA_HOST"] = "0.0.0.0"
os.environ["OLLAMA_KEEP_ALIVE"] = "30m"  # avoid repeated cold-start model reloads
os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2"

import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
```
```bash
!ollama pull llama3.1:8b && ollama pull gemma3:latest && ollama pull nomic-embed-text
```
```python
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
print(ngrok.connect(11434, "http").public_url)
```
Set `OLLAMA_BASE_URL` to the printed ngrok URL. Note: ngrok's free tier shows an interstitial
warning page to non-browser clients — every Ollama client in this codebase already sends
`ngrok-skip-browser-warning: true` to bypass it.

### 7.2 vast.ai (rented GPU, paid)
Simpler — no tunnel needed, vast.ai exposes a direct public port mapping. After renting an
instance (an `ollama` template pre-installs Ollama):

```bash
export OLLAMA_HOST=0.0.0.0
ollama serve &
ollama pull llama3.1:8b && ollama pull gemma3:latest && ollama pull nomic-embed-text
```

Find the external port mapped to internal `11434` (vast.ai dashboard → instance → IP & Port
Info), then set:
```
OLLAMA_BASE_URL=http://<public-ip>:<mapped-port>
```

**Important:** remote GPU rentals are billed continuously while running (Kaggle has weekly
quotas; vast.ai bills per second). Stop/delete instances when not actively in use.

---

## 8. Known Issues Found & Fixed During Development

Documented here deliberately — these were real bugs caught via Phoenix tracing and manual
testing, not hypothetical edge cases:

| Issue | Root cause | Fix |
|---|---|---|
| Retrieved chunks with no table header (meaningless fragments) | Naive character-count splitting cut through markdown tables | `_contains_table()` check keeps table-bearing sections atomic instead of size-splitting them |
| A smaller model (Gemma 3) occasionally **fabricated** its own tool "Observation" text instead of using real retrieved data | Text-based (ReAct) tool calling relies on the model correctly stopping at the right point; smaller models are less reliable at this | Documented as a known limitation; mitigated by an explicit anti-fabrication instruction in the Retriever's backstory, and by preferring Llama 3.1 (native tool calling) where possible |
| OpenWebUI's internal "follow-up suggestions" requests were routed through the full RAG pipeline, wasting a crew run per chat turn | `/v1/chat/completions` had no way to distinguish OpenWebUI's own utility prompts from real user questions | `_is_openwebui_utility_prompt()` detects the `### Task:` prefix OpenWebUI uses for all internal meta-prompts and short-circuits them |
| Agent LLM calls silently used `localhost` even when `OLLAMA_BASE_URL` pointed at a remote host | `crew.py`'s `LLM` object was missing an explicit `base_url` | Added `base_url=OLLAMA_BASE_URL` |
| Ingestion took 1+ minute even for a ~40-chunk document | One HTTP round-trip per chunk, sequential | Switched to `get_text_embedding_batch()` — all chunks embedded in one (or few) batched API calls |
| RAGAs evaluation returned `NaN` for every metric with no visible error | `evaluate()` silently swallows exceptions by default | Added `raise_exceptions=True`; separately fixed a `RunConfig` timeout that was too short (180s default) for local/remote CPU-bound judge inference |

---

## 9. Evaluation (RAGAs)

`evaluate.py` reads `rag_test_questions.xlsx` (6 hand-curated test cases spanning: a baseline
factual lookup, a terminology-mismatch case, a multi-hop/multi-table question, a precise
numeric lookup, a deliberate out-of-scope/hallucination probe, and an ambiguous question
requiring category disambiguation), fetches live retrieved context for each via the retriever
tool, and scores the already-generated answers against four RAGAs metrics using a local
Ollama model as the judge.

**Latest results** (agent: Llama 3.1 8B, judge: Llama 3.1 8B):

| Metric | Score | Interpretation |
|---|---|---|
| Faithfulness | 0.80 | Most claims are grounded in retrieved context; some overreach on ambiguous multi-category questions |
| Answer Relevancy | 0.655 | Answers sometimes over-broad (e.g. returning a full table instead of the specific row asked about) |
| Context Precision | 0.564 | Retrieval finds relevant content but doesn't always rank it clearly above noise |
| Context Recall | 1.00 | Retrieval never missed information that was actually needed |

**Run it:**
```bash
uv run python evaluate.py
```

Note: judge model choice (Llama 3.1 8B, same family as the agent under test in this run) was
constrained by local/rented hardware (16GB RAM, no dedicated GPU on the primary dev machine).
Reusing the agent's model family as judge is a documented simplification — an independent,
larger judge model would reduce potential self-evaluation bias in a production setting.

---

## 10. Tracing

Every agent step, tool call, and LLM call is traced via Arize Phoenix (self-hosted,
OpenTelemetry-based). View traces at **http://localhost:6006** after any request. This was
instrumental in diagnosing real issues during development (see §8) — e.g. directly observing
a fabricated tool "Observation" versus a genuine one, and confirming exact prompts sent to
each model.

---

## 11. API Reference

Full interactive documentation (Swagger/OpenAPI): **http://localhost:8000/docs**

| Endpoint | Method | Purpose |
|---|---|---|
| `/ask` | POST | Simple question → answer, `{"question": "...", "model": "..."}` |
| `/ingest` | POST | Upload a PDF (multipart), ingest into the knowledge base |
| `/health` | GET | Health check |
| `/v1/models` | GET | OpenAI-compatible model list (consumed by OpenWebUI) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint (streaming + non-streaming) |

---

## 12. Future Improvements

- Full containerization of the FastAPI app + Ollama (a `Dockerfile` and an extended
  `docker-compose.yml` covering all four services were prototyped but deferred in favor of
  finishing core functionality first — see commit history)
- An independent, larger judge model for RAGAs evaluation, once GPU budget allows
- Multi-turn conversation memory (currently each turn is treated independently — only the
  latest user message is passed to the crew, a deliberate simplification documented in
  `app.py`)
- Hierarchical CrewAI process (currently sequential — sufficient for the two-agent design
  used here, but a manager-delegated process could support a larger agent team)
