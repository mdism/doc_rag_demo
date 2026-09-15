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

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

sys.path.insert(0, ".")
from tools.retriever_tool import DocumentRetrieverTool

# ---- Config -----------------------------------------------------------
SPREADSHEET_PATH = "rag_test_questions.xlsx"
JUDGE_MODEL = "gemma3:latest"     # NOTE: no "ollama/" prefix here -- ChatOllama
                                    # (langchain_ollama) wants the raw Ollama
                                    # model tag, unlike CrewAI/LiteLLM in crew.py
                                    # which needs the "ollama/" prefix. Same
                                    # model, different naming convention per library.
JUDGE_EMBED_MODEL = "nomic-embed-text"

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

# ---- Step 3: Set up the local judge (Ollama via LangChain wrapper) --------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
    OLLAMA_BASE_URL = "http://" + OLLAMA_BASE_URL

judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, 
                                           base_url=OLLAMA_BASE_URL, temperature=0,
                                           json_mode=True,
                                           client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"}},)
                                           )
judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=JUDGE_EMBED_MODEL, 
                                                               base_url=OLLAMA_BASE_URL,
                                                               client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"}},)
                                                               )

# ---- Step 4: Run evaluation ----------------------------------------------
print("Running RAGAs evaluation (this calls the judge LLM once per metric per row)...")
from ragas.run_config import RunConfig


judge_run_config = RunConfig(timeout=1800, max_workers=1)

results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ],
    run_config=judge_run_config,
    raise_exceptions=True,
)

print("\n=== Results (averaged across all test cases) ===")
print(results)

df = results.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("\nPer-question results saved to ragas_results.csv")