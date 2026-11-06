import os
import tempfile

# Must be set before app modules are imported: offline LLM + throwaway vector DB
os.environ["LLM_MODE"] = "fake"
os.environ["CHROMA_PATH"] = tempfile.mkdtemp()
os.environ["COLLECTION"] = "test_docs"

import pytest  # noqa: E402

from app.ingest import ingest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def loaded_db():
    return ingest("data")
