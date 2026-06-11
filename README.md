<div align="center">
  <h1>🧠 PanganLink - AI Service</h1>
  <p><em>Layanan cerdas untuk prediksi harga dan rekomendasi komoditas menggunakan Machine Learning.</em></p>
  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
</div>

---

## 📖 Tentang Layanan
Repositori ini adalah spesifik *AI Service* PanganLink yang dibangun menggunakan Python dan FastAPI. Layanan ini memanfaatkan model **Huber Regression** untuk memberikan analitik tren prediksi harga pasar di masa depan serta *insight* rekomendasi langsung kepada petani.

## 👥 Tim Pengembang
Proyek ini dikembangkan untuk memenuhi tugas mata kuliah Komputasi Awan oleh:

| NRP | Nama |
| :--- | :--- |
| `152023018` | Ghinova Klarisa Irawadi |
| `152023148` | Felix Angga Resky |
| `152023141` | Parisan Apro |
| `152023167` | Raelqiansyah Putranta Dibrata |
| `152023186` | Difie Anggely |

## 📚 Dokumentasi
- [API Documentation](./API_DOCUMENTATION.md) - Daftar endpoint untuk mengakses prediksi AI.
- [AI Documentation](./AI_DOCUMENTATION.md) - Detail arsitektur model dan implementasi pipeline AI.

## 🚀 Setup & Instalasi

### Persyaratan
- **Python** (Versi 3.11+)
- **uv** (Package manager Python)

### Langkah-langkah
1. Salin file environment:
   ```bash
   cp .env.example .env
   ```
2. Sesuaikan konfigurasi di dalam `.env`.
3. Instal dependensi proyek:
   ```bash
   pip install uv
   uv pip install --system -e '.[dev]'
   ```
4. Jalankan server FastAPI:
   ```bash
   uvicorn app.main:app --reload
   ```
