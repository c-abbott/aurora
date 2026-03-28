import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from data import load_all
from models import AskRequest, AskResponse, ResponseMetadata

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    app.state.data = await load_all()
    logger.info("Data loaded in %.2fs", time.monotonic() - t0)
    yield


app = FastAPI(title="Aurora Q&A", version="0.1.0", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return AskResponse(
        answer=f"Stub response for: {request.question}",
        confidence=0.0,
        sources=[],
        metadata=ResponseMetadata(reasoning="Stub — not yet implemented."),
    )
