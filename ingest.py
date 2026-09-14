"""
Full ingestion pipeline: PDF -> Docling -> split -> merge/resize -> embed -> PGVector
Run this once per document (or whenever your source documents change).
"""

import os
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# ---- Config ----------------------------------------------------------
PDF_PATH = "sample_docs/employee_handbook.pdf"
SOURCE_NAME = "employee_handbook.pdf"  # stamped onto every chunk's metadata
MIN_CHARS = 200
MAX_CHARS = 1200
SPLIT_CHUNK_SIZE = 800
SPLIT_OVERLAP = 100
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---- Step 1: Load PDF via Docling ------------------------------------
print("Loading PDF via Docling...")
reader = DoclingReader()
documents = reader.load_data(file_path=PDF_PATH)
print(f"  -> {len(documents)} document(s) loaded")

# ---- Step 2: Split by markdown headings -------------------------------
print("Splitting by section headings...")
parser = MarkdownNodeParser()
section_nodes = parser.get_nodes_from_documents(documents)
print(f"  -> {len(section_nodes)} section-level nodes")

# ---- Step 3: Merge undersized nodes, split oversized ones -------------
print("Resizing nodes (merge small, split big)...")

def split_big(text, metadata, max_chars=SPLIT_CHUNK_SIZE, overlap=SPLIT_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        chunks.append(TextNode(text=text[start:end], metadata=metadata))
        start = end - overlap
    return chunks

final_nodes = []
buffer_text, buffer_meta = "", None

def flush_buffer():
    global buffer_text, buffer_meta
    if buffer_text:
        final_nodes.append(TextNode(text=buffer_text, metadata=buffer_meta))
        buffer_text, buffer_meta = "", None

def contains_table(text: str) -> bool:
    """Detect a markdown table by looking for its separator row, e.g. |---|---|"""
    import re
    return bool(re.search(r"\|[\s:-]*-{2,}[\s:-]*\|", text))

for node in section_nodes:
    text, meta = node.text, node.metadata
    if len(text) > MAX_CHARS and not contains_table(text):
        flush_buffer()
        final_nodes.extend(split_big(text, meta))
    elif len(text) > MAX_CHARS and contains_table(text):
        # Contains a markdown table -- splitting by raw character count would
        # cut through the table and separate the header row from its data
        # rows, producing meaningless fragments (e.g. "Casual | 7 days | ..."
        # with no column labels). Keep the whole section as one chunk instead,
        # even though it exceeds MAX_CHARS -- correctness over size here.
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

# trailing small leftover merges backward instead of standing alone
if buffer_text:
    if final_nodes:
        final_nodes[-1].text += "\n\n" + buffer_text
    else:
        final_nodes.append(TextNode(text=buffer_text, metadata=buffer_meta))

print(f"  -> {len(final_nodes)} final nodes after resizing")

# Stamp every node with which source file it came from -- this is what
# ref_doc_id was supposed to give us automatically, but our manual
# merge/split step (Step 3) creates new TextNode objects that lose that
# link. Setting it explicitly here means every chunk can always be traced
# back to its source document, which matters once you ingest multiple PDFs.
for node in final_nodes:
    node.metadata["source_file"] = SOURCE_NAME

# ---- Step 4: Embed each node -------------------------------------------
print(f"Embedding {len(final_nodes)} nodes via Ollama ({EMBED_MODEL})...")
embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

for i, node in enumerate(final_nodes):
    node.embedding = embed_model.get_text_embedding(node.text)
    print(f"  embedded node {i+1}/{len(final_nodes)}", end="\r")
print()

# ---- Step 5: Store in PGVector -------------------------------------------
print("Storing nodes in PGVector...")
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name="employee_handbook",
    embed_dim=EMBED_DIM,
)
vector_store.add(final_nodes)

print(f"\nDone. Ingested {len(final_nodes)} nodes into PGVector table 'data_employee_handbook'.")