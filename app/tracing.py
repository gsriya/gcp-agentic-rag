"""Granular tracing: every agent step logs one JSON line with its timing.
On Cloud Run these lines show up as structured logs in Cloud Logging."""
import json
import sys
import time
import uuid
from contextlib import contextmanager


class Trace:
    def __init__(self, question: str):
        self.trace_id = uuid.uuid4().hex[:12]
        self.question = question
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        t0 = time.perf_counter()
        record = {"trace_id": self.trace_id, "span": name, **attrs}
        try:
            yield record
            record["status"] = "ok"
        except Exception as e:  # keep the error in the trace, then re-raise
            record["status"] = "error"
            record["error"] = str(e)
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.spans.append(record)
            print(json.dumps({"severity": "INFO", **record}), file=sys.stderr)
