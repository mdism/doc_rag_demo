from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from instrumentation import setup_tracing
from crew import DocRagDemoCrew

setup_tracing()

app = FastAPI(title="Document Search Platform API")

# CrewAI's kickoff() is synchronous/blocking -- run it in a thread pool so it
# doesn't block FastAPI's event loop, which would otherwise stall every other
# request while one crew run is in progress.
_executor = ThreadPoolExecutor(max_workers=2)

MODEL_ID = "doc-rag-assistant"


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: DocRagDemoCrew().crew().kickoff(inputs={"question": req.question}),
    )
    return AskResponse(answer=result.raw)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---- OpenAI-compatible endpoints, for OpenWebUI ---------------------------

@app.get("/v1/models")
async def list_models():
    """OpenWebUI calls this on startup to populate its model picker."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "doc-rag-demo",
            }
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


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request.get("messages", [])
    stream = request.get("stream", False)
    question = _extract_question(messages)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not stream:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: DocRagDemoCrew().crew().kickoff(inputs={"question": question}),
        )
        answer = result.raw
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Streaming: yield an initial chunk immediately so OpenWebUI's response
    # actually starts flowing right away (triggering its loading/typing
    # state) instead of the connection sitting silent while the blocking
    # crew.kickoff() call runs. The real answer still only arrives as one
    # chunk once kickoff() finishes -- CrewAI doesn't yield partial tokens --
    # but at least the UI shows activity instead of looking frozen.
    async def event_stream():
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: DocRagDemoCrew().crew().kickoff(inputs={"question": question}),
        )
        answer = result.raw

        content_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")