"""The multi-agent workflow, built with LangGraph.

    retrieve  ->  answer  ->  check --PASS--> END
                    ^            |
                    +--FAIL------+   (self-reflection: retry with the critic's feedback,
                                      at most MAX_RETRIES times)

- Retriever agent: finds the top-k chunks in the vector DB.
- Answer agent: writes an answer using ONLY those chunks, citing them as [1], [2]...
- Checker agent: reviews the answer against the sources and returns PASS or FAIL + reason.
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import settings
from .ingest import get_collection
from .llm import Usage, get_llm
from .tracing import Trace


class RAGState(TypedDict, total=False):
    question: str
    sources: list[dict]
    answer: str
    feedback: str
    verdict: str
    attempts: int
    usage: Usage
    trace: Trace


ANSWER_PROMPT = """You are an enterprise assistant. Answer the question using ONLY the sources.
Cite every fact with its source number like [1]. If the sources do not contain the answer,
say "I don't know based on the provided documents."
{feedback}
QUESTION: {question}

SOURCES:
{sources}"""

CHECK_PROMPT = """You are a strict reviewer. Check the ANSWER against the SOURCES.
Reply with exactly "PASS" if every claim is supported and cited.
Otherwise reply "FAIL: <one-sentence reason>".

QUESTION: {question}
ANSWER: {answer}

SOURCES:
{sources}"""


def _fmt(sources: list[dict]) -> str:
    return "\n".join(f"[{i + 1}] ({s['source']}) {s['text']}" for i, s in enumerate(sources))


def make_graph(llm=None):
    llm = llm or get_llm()
    collection = get_collection()

    def retrieve(state: RAGState) -> RAGState:
        with state["trace"].span("retrieve", top_k=settings.top_k) as rec:
            qvec = llm.embed([state["question"]])[0]
            res = collection.query(query_embeddings=[qvec], n_results=settings.top_k)
            sources = [{"id": i, "text": d, "source": m["source"], "kind": m["kind"],
                        "distance": round(dist, 4)}
                       for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]
            rec["hits"] = [s["id"] for s in sources]
        return {"sources": sources}

    def answer(state: RAGState) -> RAGState:
        attempt = state.get("attempts", 0) + 1
        with state["trace"].span("answer", attempt=attempt):
            fb = state.get("feedback", "")
            prompt = ANSWER_PROMPT.format(
                question=state["question"], sources=_fmt(state["sources"]),
                feedback=f"A reviewer rejected your last answer: {fb}\nFix it." if fb else "")
            text = llm.generate(prompt, state["usage"], step=f"answer_{attempt}")
        return {"answer": text.strip(), "attempts": attempt}

    def check(state: RAGState) -> RAGState:
        with state["trace"].span("check") as rec:
            text = llm.generate(CHECK_PROMPT.format(question=state["question"],
                                                    answer=state["answer"],
                                                    sources=_fmt(state["sources"])),
                                state["usage"], step="check").strip()
            verdict = "PASS" if text.upper().startswith("PASS") else "FAIL"
            rec["verdict"] = verdict
        return {"verdict": verdict, "feedback": "" if verdict == "PASS" else text}

    def route(state: RAGState) -> str:
        if state["verdict"] == "PASS" or state["attempts"] > settings.max_retries:
            return "done"
        return "retry"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("check", check)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "check")
    g.add_conditional_edges("check", route, {"retry": "answer", "done": END})
    return g.compile()


def ask(question: str, graph=None) -> dict:
    graph = graph or make_graph()
    usage, trace = Usage(), Trace(question)
    with trace.span("request"):
        final = graph.invoke({"question": question, "usage": usage, "trace": trace,
                              "attempts": 0})
    return {
        "trace_id": trace.trace_id,
        "question": question,
        "answer": final["answer"],
        "verified": final["verdict"] == "PASS",
        "attempts": final["attempts"],
        "sources": [{k: s[k] for k in ("id", "source", "kind", "distance")}
                    for s in final["sources"]],
        "metrics": {**usage.summary(),
                    "total_latency_ms": next(s["latency_ms"] for s in trace.spans
                                             if s["span"] == "request")},
        "trace": trace.spans,
    }
