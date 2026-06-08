# be-panganlink-ai-service

FastAPI backend service template.

## Requirements

- Python 3.11+
- Docker optional

## Setup

1. Copy environment variables: `cp .env.example .env`
2. Fill in values in `.env`
3. Install dependencies: `pip install uv && uv pip install --system -e '.[dev]'`
4. Run: `uvicorn app.main:app --reload`

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/placeholder` | Placeholder endpoint |
