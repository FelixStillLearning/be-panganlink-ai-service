# PanganLink AI Service API Documentation

Layanan AI menyediakan endpoint untuk mendapatkan prediksi harga dan rekomendasi yang diproses oleh model machine learning.

## Base URL
`/api/v1`

## Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Mengecek status ketersediaan layanan AI.

### 2. Prediksi Harga
- **Endpoint**: `POST /predict/price`
- **Description**: Mendapatkan prediksi harga komoditas berdasarkan tren pasar historis.
- **Payload**:
  ```json
  {
    "commodity_id": "string",
    "date": "YYYY-MM-DD"
  }
  ```

### 3. Rekomendasi Petani
- **Endpoint**: `GET /recommend/farmer/:farmer_id`
- **Description**: Memberikan rekomendasi strategis untuk petani berdasarkan data pasar terbaru.
