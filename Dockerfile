FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# No API keys baked into the image: on Cloud Run the app calls Vertex AI with the
# service account's own identity. The vector DB is built on first start (see app/main.py).
ENV USE_VERTEX=true

# Cloud Run sends traffic to $PORT (8080 by default)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
