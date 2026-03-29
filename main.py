import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google import genai
from starlette.requests import Request

from data import build_index, load_all
from models import AskRequest, AskResponse
from prompt import PROJECT, LOCATION, ask as answer_question

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    app.state.data = await load_all()
    logger.info("Data loaded in %.2fs", time.monotonic() - t0)
    app.state.genai_client = genai.Client(
        vertexai=True, project=PROJECT, location=LOCATION
    )
    t1 = time.monotonic()
    await build_index(app.state.data, client=app.state.genai_client)
    logger.info("Index built in %.2fs", time.monotonic() - t1)
    yield


app = FastAPI(title="Aurora Q&A", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    data = getattr(request.app.state, "data", None)
    if data and data.members:
        return {"status": "healthy", "members": len(data.members)}
    return JSONResponse(status_code=503, content={"status": "not ready"})


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    return await answer_question(
        body.question, request.app.state.data, request.app.state.genai_client
    )
