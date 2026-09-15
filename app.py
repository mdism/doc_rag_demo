from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel

from instrumentation import setup_tracing
from crew import DocRagDemoCrew

setup_tracing()

app = FastAPI(title="Document Search Platform API")

# CrewAI's kickoff() is synchronous/blocking -- run it in a thread pool so it
# doesn't block FastAPI's event loop, which would otherwise stall every other
# request while one crew run is in progress.
_executor = ThreadPoolExecutor(max_workers=2)


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