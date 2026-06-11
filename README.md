# PanganLink AI Service

FastAPI backend service untuk melayani prediksi harga dan rekomendasi menggunakan model Machine Learning.

## Documentation
- [API Documentation](./API_DOCUMENTATION.md)
- [AI Documentation](./AI_DOCUMENTATION.md)

## Requirements
- Python 3.11+
- Docker (optional)

## Setup
1. Copy environment variables: `cp .env.example .env`
2. Fill in values in `.env`
3. Install dependencies: `pip install uv && uv pip install --system -e '.[dev]'`
4. Run server: `uvicorn app.main:app --reload`
