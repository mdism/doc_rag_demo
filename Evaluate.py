"""
RAGAs evaluation of the RAG pipeline.

Reads Question / Expected Answer / Actual Answer from rag_test_questions.xlsx,
fetches the real retrieved contexts for each question via the retriever tool,
and scores everything against RAGAs' four core metrics using a local Ollama
model as the judge (via Ollama's OpenAI-compatible endpoint).
"""

import os
import sys
import openpyxl
from openai import OpenAI

from ragas import evaluate, EvaluationDataset
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

sys.path.insert(0, ".")
from tools.retriever_tool import DocumentRetrieverTool

# ---- Config -----------------------------------------------------------
SPREADSHEET_PATH = "rag_test_questions.xlsx"
JUDGE_MODEL = "llama3.1:8b"       # ~4.7GB -- fits 16GB RAM, no GPU needed
JUDGE_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"

# ---- Step 1: Load your filled-in test cases from the spreadsheet -------
print("Loading test cases from spreadsheet...")
wb = openpyxl.load_workbook(SPREADSHEET_PATH)
ws = wb["Test Questions"]

rows = []
for row in ws.iter_rows(min_row=2, values_only=True):
    number, category, question, expected, actual = row[0], row[1], row[2], row[3], row[4]
    if not question or not actual:
        continue  # skip rows you haven't filled in yet
    rows.append({"question": question, "reference": expected, "answer": actual})

print(f"  -> {len(rows)} filled-in test cases found")
if not rows:
    raise SystemExit("No filled-in rows found -- fill in 'Actual Answer' in the spreadsheet first.")

# ---- Step 2: Fetch real retrieved contexts for each question -----------
print("Fetching retrieved contexts for each question...")
retriever = DocumentRetrieverTool()

eval_rows = []
for r in rows:
    raw = retriever._run(r["question"])
    contexts = [c.strip() for c in raw.split("---") if c.strip()]
    eval_rows.append({
        "user_input": r["question"],
        "retrieved_contexts": contexts,
        "response": r["answer"],
        "reference": r["reference"],
    })

dataset = EvaluationDataset.from_list(eval_rows)

# ---- Step 3: Set up the local judge (Ollama via its OpenAI-compatible API) --
# Ollama serves an OpenAI-compatible endpoint at /v1, so we can use the
# standard OpenAI client pointed at localhost instead of a real OpenAI key.
ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")  # api_key is ignored by Ollama

judge_llm = llm_factory(JUDGE_MODEL, client=ollama_client)
judge_embeddings = OpenAIEmbeddings(client=ollama_client, model=JUDGE_EMBED_MODEL)

# ---- Step 4: Run evaluation ----------------------------------------------
print("Running RAGAs evaluation (this calls the judge LLM once per metric per row)...")
results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ],
)

print("\n=== Results (averaged across all test cases) ===")
print(results)

df = results.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("\nPer-question results saved to ragas_results.csv")