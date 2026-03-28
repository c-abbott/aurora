from fastapi import FastAPI

from models import AskRequest, AskResponse, ResponseMetadata

app = FastAPI(title="Aurora Q&A", version="0.1.0")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return AskResponse(
        answer=f"Stub response for: {request.question}",
        confidence=0.0,
        sources=[],
        metadata=ResponseMetadata(reasoning="Stub — not yet implemented."),
    )
