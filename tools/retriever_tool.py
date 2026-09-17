import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
FETCH_TOP_K = 10
FINAL_TOP_K = 3

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
    OLLAMA_BASE_URL = "http://" + OLLAMA_BASE_URL
 

_reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=FINAL_TOP_K,
)


class RetrieverInput(BaseModel):
    query: str = Field(..., description="The search query to look up in the knowledge base")


class DocumentRetrieverTool(BaseTool):
    name: str = "Document Retriever"
    description: str = (
        "Search the employee handbook knowledge base and return the most "
        "relevant passages for a given query. Use this whenever you need to "
        "find specific facts, numbers, or policy details to answer a question."
    )
    args_schema: type[BaseModel] = RetrieverInput

    def _run(self, query: str) -> str:
        embed_model = OllamaEmbedding(model_name=EMBED_MODEL, 
                                      base_url=OLLAMA_BASE_URL,
                                      client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"},
                                                    "timeout": 300,})
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            password=DB_PASSWORD,
            port=DB_PORT,
            user=DB_USER,
            table_name="employee_handbook",
            embed_dim=EMBED_DIM,
        )

        # Stage 1: vector search -- fast, approximate, wide net
        query_vector = embed_model.get_text_embedding(query)
        result = vector_store.query(
            VectorStoreQuery(query_embedding=query_vector, similarity_top_k=FETCH_TOP_K)
        )

        if not result.nodes:
            return "No relevant passages found in the knowledge base."

        candidates = [
            NodeWithScore(node=node, score=score)
            for node, score in zip(result.nodes, result.similarities)
        ]

        reranked = _reranker.postprocess_nodes(candidates, query_bundle=QueryBundle(query_str=query))

        formatted = []
        for nws in reranked:
            source = nws.node.metadata.get("source_file", "unknown")
            formatted.append(f"[source: {source}, relevance: {nws.score:.2f}]\n{nws.node.text}")

        return "\n\n---\n\n".join(formatted)