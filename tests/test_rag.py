from pathlib import Path

from fastapi.testclient import TestClient

from app.graph import ask, make_graph
from app.ingest import build_records, chunk_text
from app.llm import FakeLLM
from app.main import app


def test_chunking_keeps_all_text_and_respects_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) > 1
    assert all(f"Paragraph {i} " in "".join(chunks) for i in range(20))
    assert max(len(c) for c in chunks) < 1000


def test_pipeline_reads_structured_and_unstructured_data():
    kinds = {r["meta"]["kind"] for r in build_records(Path("data"))}
    assert kinds == {"structured", "unstructured"}


def test_csv_rows_become_self_describing_records():
    rows = [r for r in build_records(Path("data")) if r["meta"]["kind"] == "structured"]
    assert any("plan: Enterprise" in r["text"] and "99.99" in r["text"] for r in rows)


def test_ask_returns_grounded_answer_with_metrics_and_trace():
    r = ask("What is the hotel limit per night in London?")
    assert r["sources"][0]["source"] == "travel_expenses_policy.md"
    assert r["verified"] is True
    assert r["metrics"]["llm_calls"] == 2  # answer + check
    assert {"tokens_in", "tokens_out", "cost_usd", "tokens_per_sec"} <= r["metrics"].keys()
    assert [s["span"] for s in r["trace"]] == ["retrieve", "answer", "check", "request"]


def test_structured_question_retrieves_csv_row():
    r = ask("What is the uptime SLA of the Enterprise support plan?")
    assert r["sources"][0]["source"] == "support_plans.csv"


class AlwaysFailChecker(FakeLLM):
    def generate(self, prompt, usage, step):
        if step == "check":
            usage.add(step, 1, 1, 0.0)
            return "FAIL: missing citation"
        return super().generate(prompt, usage, step)


def test_self_reflection_retries_then_stops():
    r = ask("How long must passwords be?", make_graph(AlwaysFailChecker()))
    assert r["verified"] is False
    assert r["attempts"] == 2  # first try + 1 retry (MAX_RETRIES=1), then stop
    assert [s["span"] for s in r["trace"]].count("answer") == 2


def test_api_health_and_ask():
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    res = c.post("/ask", json={"question": "What is the daily meal allowance?"})
    assert res.status_code == 200 and "45" in res.json()["answer"]
    assert c.post("/ask", json={"question": "x"}).status_code == 422  # validation
