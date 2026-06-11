# Dokumentasi AI PanganLink

Layanan AI pada PanganLink dirancang untuk memberikan *insight* berbasis data, seperti prediksi tren harga pasar dan rekomendasi komoditas bagi petani, guna membantu pengambilan keputusan strategis.

## 1. Arsitektur Model
Model utama yang digunakan untuk prediksi harga adalah **Huber Regression**. Algoritma ini dipilih karena sifatnya yang sangat tangguh (*robust*) dalam menangani data harga komoditas yang rentan terhadap volatilitas ekstrem dan seringkali memiliki banyak *outliers* (pencilan). Huber Regression secara cerdas memadukan *Mean Squared Error* (MSE) untuk error yang kecil dan *Mean Absolute Error* (MAE) untuk error yang besar, menjadikannya sangat efektif untuk memprediksi fluktuasi harga pasar pangan yang dinamis.

## 2. Alur Kerja (Pipeline)
1. **Data Ingestion**: Mengambil data historis harga pasar secara periodik dari Data Handler Service PanganLink.
2. **Preprocessing**: Melakukan pembersihan data, penanganan *missing values*, dan normalisasi.
3. **Training & Update**: Model **Huber Regression** dilatih secara berkala ketika ada data pasar terbaru, menyesuaikan *loss function* secara otomatis untuk menjaga tingkat kesalahan (MAPE) tetap serendah mungkin.
4. **Inference**: Menggunakan model yang telah dilatih untuk memprediksi tren harga komoditas pada beberapa waktu ke depan (misalnya 7 hari atau 30 hari ke depan).
5. **Post-processing**: Hasil prediksi diterjemahkan menjadi rekomendasi bisnis (*actionable insights*) yang mudah dipahami oleh pengguna (petani).

## 3. Fitur Utama
- **Price Forecasting**: Memprediksi fluktuasi harga komoditas pangan secara handal, meminimalisir gangguan dari lonjakan atau penurunan harga seketika yang ekstrem berkat *robustness* Huber Regression.
- **Smart Recommendations**: Memberikan saran waktu terbaik untuk menjual komoditas atau menanam jenis komoditas tertentu berdasarkan analisis tren pasar agar petani dapat memaksimalkan potensi keuntungan mereka.
