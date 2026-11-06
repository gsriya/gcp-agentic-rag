#!/usr/bin/env bash
# Deploy to Google Cloud Run in London (europe-west2), calling Gemini through Vertex AI.
# No API keys: the Cloud Run service account is given permission to use Vertex AI.
#
# Before running:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION=europe-west2
SERVICE=agentic-rag
SA="$SERVICE-sa@$PROJECT_ID.iam.gserviceaccount.com"

gcloud services enable run.googleapis.com aiplatform.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com

# A dedicated service account with only the permission it needs (least privilege)
gcloud iam service-accounts create "$SERVICE-sa" --display-name="Agentic RAG" 2>/dev/null || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user" --condition=None >/dev/null

# Build from source with Cloud Build and deploy
gcloud run deploy "$SERVICE" --source . --region="$REGION" \
  --service-account="$SA" --allow-unauthenticated \
  --memory=1Gi --max-instances=2 \
  --set-env-vars="USE_VERTEX=true,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "Deployed: $URL/docs"
echo "curl -X POST $URL/ask -H 'Content-Type: application/json' \\"
echo "  -d '{\"question\": \"What is the hotel limit per night in London?\"}'"
