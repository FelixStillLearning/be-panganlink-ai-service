# Dokumentasi AI PanganLink

Layanan AI pada PanganLink dirancang untuk memberikan *insight* berbasis data, seperti prediksi tren harga pasar dan rekomendasi komoditas bagi petani, guna membantu pengambilan keputusan strategis.

## 1. Arsitektur Model
Model utama yang digunakan untuk prediksi harga adalah **Bayesian Ridge Regression**. Berdasarkan uji coba, model ini terbukti lebih stabil dan efektif dibandingkan algoritma kompleks lainnya (seperti LSTM atau Extra Trees) dalam menangani data harga komoditas yang memiliki volatilitas tinggi namun dengan ukuran dataset yang relatif terbatas.

## 2. Alur Kerja (Pipeline)
1. **Data Ingestion**: Mengambil data historis harga pasar secara periodik dari Data Handler Service PanganLink.
2. **Preprocessing**: Melakukan pembersihan data, penanganan *missing values*, dan normalisasi.
3. **Training & Update**: Model dilatih ulang secara berkala ketika ada data pasar terbaru untuk menjaga tingkat akurasi (MAPE) tetap optimal.
4. **Inference**: Menggunakan model yang telah dilatih untuk memprediksi harga komoditas pada beberapa hari ke depan.
5. **Post-processing**: Hasil prediksi diterjemahkan menjadi rekomendasi bisnis (*actionable insights*) untuk pengguna.

## 3. Fitur Utama
- **Price Forecasting**: Memprediksi fluktuasi harga komoditas di masa mendatang (contoh: prediksi 7 hari ke depan) lengkap dengan batas atas dan batas bawah (confidence intervals).
- **Smart Recommendations**: Memberikan saran waktu terbaik untuk menjual komoditas atau menanam komoditas tertentu berdasarkan tren pasar agar petani dapat memaksimalkan keuntungan mereka.
