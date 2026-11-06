"""Settings, read from environment variables so the same code runs locally and on Cloud Run."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # "gemini" = real Google model, "fake" = offline mode for tests (no API key, no cost)
    llm_mode: str = os.getenv("LLM_MODE", "gemini")
    # Use Vertex AI (GCP project) if set, otherwise the Gemini API key
    use_vertex: bool = os.getenv("USE_VERTEX", "false").lower() == "true"
    gcp_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    gcp_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west2")  # London
    api_key: str = os.getenv("GEMINI_API_KEY", "")

    chat_model: str = os.getenv("CHAT_MODEL", "gemini-flash-latest")
    embed_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")

    # Price per 1M tokens in USD. Check Google's current pricing page and update.
    price_in_per_m: float = float(os.getenv("PRICE_INPUT_PER_M", "0.30"))
    price_out_per_m: float = float(os.getenv("PRICE_OUTPUT_PER_M", "2.50"))

    db_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    collection: str = os.getenv("COLLECTION", "docs")
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "1"))  # self-check retry loops


settings = Settings()
