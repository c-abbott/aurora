from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str


class ResponseMetadata(BaseModel):
    reasoning: str


class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]
    metadata: ResponseMetadata
