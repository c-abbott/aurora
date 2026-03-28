import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from starlette.requests import Request

from data import load_all
from models import AskRequest, AskResponse
from prompt import ask as answer_question

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    app.state.data = await load_all()
    logger.info("Data loaded in %.2fs", time.monotonic() - t0)
    yield


app = FastAPI(title="Aurora Q&A", version="0.1.0", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    return await answer_question(body.question, request.app.state.data)
