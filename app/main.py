"""HTTP API. Run locally:  uvicorn app.main:app --reload
Cloud Run starts it the same way (see Dockerfile)."""
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import ask, make_graph
from .ingest import get_collection, ingest

app = FastAPI(title="Agentic RAG on Google Cloud",
              description="Multi-agent RAG with self-checking, tracing and cost metrics")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@lru_cache
def graph():
    if get_collection().count() == 0:  # fresh container: build the vector DB once
        ingest("data")
    return make_graph()  # built once per container, reused across requests


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(q: Question):
    return ask(q.question, graph())
