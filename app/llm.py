"""Thin wrapper around Gemini that also records LLM-native metrics:
tokens in/out, latency, tokens/sec and cost per call.

FakeLLM gives deterministic answers so tests run offline with no API key.
"""
import hashlib
import math
import re
import time
from dataclasses import dataclass, field

from .config import settings


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    per_call: list = field(default_factory=list)

    def add(self, name: str, tin: int, tout: int, secs: float):
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout
        self.seconds += secs
        self.per_call.append({"step": name, "tokens_in": tin, "tokens_out": tout,
                              "latency_s": round(secs, 3)})

    @property
    def cost_usd(self) -> float:
        return (self.tokens_in * settings.price_in_per_m
                + self.tokens_out * settings.price_out_per_m) / 1_000_000

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens_out / self.seconds if self.seconds else 0.0

    def summary(self) -> dict:
        return {"llm_calls": self.calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "llm_seconds": round(self.seconds, 3),
                "tokens_per_sec": round(self.tokens_per_sec, 1),
                "cost_usd": round(self.cost_usd, 6), "per_call": self.per_call}


class GeminiLLM:
    def __init__(self):
        from google import genai
        if settings.use_vertex:
            self.client = genai.Client(vertexai=True, project=settings.gcp_project,
                                       location=settings.gcp_location)
        else:
            self.client = genai.Client(api_key=settings.api_key)

    def generate(self, prompt: str, usage: Usage, step: str) -> str:
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(model=settings.chat_model, contents=prompt)
        secs = time.perf_counter() - t0
        meta = resp.usage_metadata
        usage.add(step, meta.prompt_token_count or 0, meta.candidates_token_count or 0, secs)
        return resp.text or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 50):  # batch to stay under request limits
            res = self.client.models.embed_content(model=settings.embed_model,
                                                   contents=texts[i:i + 50])
            out.extend(e.values for e in res.embeddings)
        return out


STOP = {"the", "a", "an", "of", "is", "what", "how", "to", "in", "and", "or", "for",
        "must", "be", "do", "i", "per", "my", "can", "which", "on", "it", "are", "with"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


class FakeLLM:
    """Offline stand-in for tests. Embeddings are hashed bags of words (similar text stays
    close); answers return the source sentence that best overlaps the question."""
    DIM = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in words(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs

    def generate(self, prompt: str, usage: Usage, step: str) -> str:
        t0 = time.perf_counter()
        if step == "check":
            text = "PASS"
        else:
            question = words(prompt.split("QUESTION:")[1].split("SOURCES:")[0])
            sources = re.split(r"\n(?=\[\d+\])", prompt.split("SOURCES:")[-1].strip())
            best = (0, "I don't know based on the provided documents.")
            for idx, src in enumerate(sources, 1):
                for sent in re.split(r"(?<=[.!?])\s+|\n+|; ", src):
                    score = len(question & words(sent))
                    if score > best[0]:
                        best = (score, f"{sent.strip()} [{idx}]")
            text = best[1]
        usage.add(step, len(prompt.split()), len(text.split()), time.perf_counter() - t0)
        return text


def get_llm():
    return FakeLLM() if settings.llm_mode == "fake" else GeminiLLM()
