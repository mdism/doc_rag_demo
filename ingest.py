"""
Full ingestion pipeline: PDF -> Docling -> split -> merge/resize -> embed -> PGVector

Usable two ways:
  - CLI:  python ingest.py path/to/file.pdf
  - API:  imported and called by app.py's /ingest endpoint
"""

import os
import re
import sys

from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# ---- Config ----------------------------------------------------------
MIN_CHARS = 200
MAX_CHARS = 1200
SPLIT_CHUNK_SIZE = 800
SPLIT_OVERLAP = 100
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
TABLE_NAME = "employee_handbook"

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
    OLLAMA_BASE_URL = "http://" + OLLAMA_BASE_URL


def _split_big(text, metadata, max_chars=SPLIT_CHUNK_SIZE, overlap=SPLIT_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        chunks.append(TextNode(text=text[start:end], metadata=metadata))
        start = end - overlap
    return chunks


def _contains_table(text: str) -> bool:
    """Detect a markdown table by looking for its separator row, e.g. |---|---|"""
    return bool(re.search(r"\|[\s:-]*-{2,}[\s:-]*\|", text))


def ingest_document(pdf_path: str, source_name: str) -> int:
    """
    Runs the full ingestion pipeline for one PDF and stores it in PGVector.
    Returns the number of chunks (nodes) ingested.
    """
    reader = DoclingReader()
    documents = reader.load_data(file_path=pdf_path)

    parser = MarkdownNodeParser()
    section_nodes = parser.get_nodes_from_documents(documents)

    final_nodes = []
    buffer_text, buffer_meta = "", None

    def flush_buffer():
        nonlocal buffer_text, buffer_meta
        if buffer_text:
            final_nodes.append(TextNode(text=buffer_text, metadata=buffer_meta))
            buffer_text, buffer_meta = "", None

    for node in section_nodes:
        text, meta = node.text, node.metadata
        if len(text) > MAX_CHARS and not _contains_table(text):
            flush_buffer()
            final_nodes.extend(_split_big(text, meta))
        elif len(text) > MAX_CHARS and _contains_table(text):
            flush_buffer()
            final_nodes.append(TextNode(text=text, metadata=meta))
        elif len(text) < MIN_CHARS:
            buffer_text = (buffer_text + "\n\n" + text) if buffer_text else text
            buffer_meta = buffer_meta or meta
            if len(buffer_text) >= MIN_CHARS:
                flush_buffer()
        else:
            if buffer_text:
                buffer_text += "\n\n" + text
                flush_buffer()
            else:
                final_nodes.append(node)

    if buffer_text:
        if final_nodes:
            final_nodes[-1].text += "\n\n" + buffer_text
        else:
            final_nodes.append(TextNode(text=buffer_text, metadata=buffer_meta))

    for node in final_nodes:
        node.metadata["source_file"] = source_name

    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        embed_batch_size=100,
        client_kwargs={
            "headers": {"ngrok-skip-browser-warning": "true"},
            "timeout": 300,
        },
    )
    texts = [node.text for node in final_nodes]
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
    for node, embedding in zip(final_nodes, embeddings):
        node.embedding = embedding

    # ---- Step 5: Store in PGVector -------------------------------------------
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
    )
    vector_store.add(final_nodes)

    return len(final_nodes)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/file.pdf> [source_name]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    source_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(pdf_path)

    print(f"Ingesting {pdf_path} as '{source_name}'...")
    count = ingest_document(pdf_path, source_name)
    print(f"Done. Ingested {count} chunks into PGVector table '{TABLE_NAME}'.")