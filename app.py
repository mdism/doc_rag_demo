from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import os
import shutil
import time
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from instrumentation import setup_tracing
from crew import DocRagDemoCrew
from ingest import ingest_document

setup_tracing()

app = FastAPI(title="Document Search Platform API")

_executor = ThreadPoolExecutor(max_workers=2)

UPLOAD_DIR = "sample_docs"

AVAILABLE_MODELS = {
    "doc-rag-assistant-llama3.1": "ollama/llama3.1:8b",
    "doc-rag-assistant-gemma3": "ollama/gemma3:latest",
}
DEFAULT_MODEL_ID = "doc-rag-assistant-llama3.1"


class AskRequest(BaseModel):
    question: str
    model: str = DEFAULT_MODEL_ID


class AskResponse(BaseModel):
    answer: str


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a new PDF and add it to the knowledge base."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(
        _executor,
        lambda: ingest_document(dest_path, file.filename),
    )
    return IngestResponse(filename=file.filename, chunks_ingested=count)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    llm_model = AVAILABLE_MODELS.get(req.model, AVAILABLE_MODELS[DEFAULT_MODEL_ID])
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: DocRagDemoCrew(llm_model=llm_model).crew().kickoff(inputs={"question": req.question}),
    )
    return AskResponse(answer=result.raw)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---- OpenAI-compatible endpoints, for OpenWebUI ---------------------------

@app.get("/v1/models")
async def list_models():
    """OpenWebUI calls this on startup to populate its model picker --
    every key in AVAILABLE_MODELS shows up as its own selectable entry."""
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": created, "owned_by": "doc-rag-demo"}
            for model_id in AVAILABLE_MODELS
        ],
    }


def _extract_question(messages: list) -> str:
    """OpenWebUI sends the full conversation history; we only pass the
    latest user message through to the crew as {question}. Multi-turn
    memory is a known simplification -- see README for the tradeoff."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _is_openwebui_utility_prompt(text: str) -> bool:
    """OpenWebUI sends its own internal meta-prompts (follow-up suggestions,
    chat title generation, auto-tagging, etc.) through the same chat
    completions endpoint as real user messages. These are always prefixed
    with '### Task:' -- detecting this lets us skip the expensive RAG
    pipeline entirely for requests that were never a real user question."""
    return text.strip().startswith("### Task:")


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request.get("messages", [])
    stream = request.get("stream", False)
    question = _extract_question(messages)

    requested_model_id = request.get("model", DEFAULT_MODEL_ID)
    llm_model = AVAILABLE_MODELS.get(requested_model_id, AVAILABLE_MODELS[DEFAULT_MODEL_ID])

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if _is_openwebui_utility_prompt(question):
        empty_answer = '{"follow_ups": []}'
        if not stream:
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": requested_model_id,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": empty_answer}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        async def empty_stream():
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": requested_model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": empty_answer}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    if not stream:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: DocRagDemoCrew(llm_model=llm_model).crew().kickoff(inputs={"question": question}),
        )
        answer = result.raw
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": requested_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def event_stream():
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: DocRagDemoCrew(llm_model=llm_model).crew().kickoff(inputs={"question": question}),
        )
        answer = result.raw

        content_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model_id,
            "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")