"""Evaluation pipeline: runs every question in questions.jsonl and reports
retrieval hit rate, answer accuracy, verification rate, latency (p50/p95) and cost.

Run:  python -m eval.run_eval
"""
import json
import statistics
from pathlib import Path

from app.graph import ask, make_graph


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main(path: str = "eval/questions.jsonl") -> dict:
    graph = make_graph()
    cases = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = []
    for c in cases:
        r = ask(c["question"], graph)
        hit = any(s["source"] == c["expected_source"] for s in r["sources"])
        correct = all(m.lower() in r["answer"].lower() for m in c["must_contain"])
        rows.append({"question": c["question"], "retrieval_hit": hit, "correct": correct,
                     "verified": r["verified"], "latency_ms": r["metrics"]["total_latency_ms"],
                     "cost_usd": r["metrics"]["cost_usd"],
                     "tokens_per_sec": r["metrics"]["tokens_per_sec"]})

    n = len(rows)
    lat = [r["latency_ms"] for r in rows]
    report = {
        "cases": n,
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
        "answer_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "verified_rate": round(sum(r["verified"] for r in rows) / n, 3),
        "latency_p50_ms": round(statistics.median(lat), 1),
        "latency_p95_ms": round(pct(lat, 95), 1),
        "avg_cost_per_request_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "avg_tokens_per_sec": round(sum(r["tokens_per_sec"] for r in rows) / n, 1),
    }
    print(json.dumps(report, indent=2))
    for r in rows:
        mark = "OK " if r["correct"] else "BAD"
        print(f"  {mark} hit={r['retrieval_hit']!s:5} {r['question']}")
    Path("eval/report.json").write_text(json.dumps({"summary": report, "rows": rows}, indent=2))
    return report


if __name__ == "__main__":
    main()
