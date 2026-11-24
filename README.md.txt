# Prediksi Nilai Median Rumah California (Streamlit)

Aplikasi web berbasis **Streamlit** dan **Scikit-Learn** untuk memperkirakan nilai median rumah pada tingkat kelompok blok sensus di California berdasarkan data sensus historis.

## Struktur Proyek
* `train.py` — Skrip pelatihan model regresi dan penyimpanan artefak ke `model_california.joblib`.
* `app_streamlit.py` — Antarmuka web interaktif berbasis Streamlit beserta validasi input dan penanganan hasil ekstrem.
* `requirements.txt` — Daftar pustaka Python yang dibutuhkan untuk menjalankan proyek.
* `model_california.joblib` — Artefak model terlatih beserta metadata versi dan metrik evaluasi.

## Menjalankan Aplikasi Secara Lokal
1. Masuk ke folder proyek:
   ```bash
   cd /d D:\california-housing-deploy
   ```
2. Instal dependensi:
   ```bash
   py -m pip install -r requirements.txt
   ```
3. Latih model (jika `model_california.joblib` belum tersedia):
   ```bash
   py train.py
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   py -m streamlit run app_streamlit.py
   ```

## Ringkasan Evaluasi Model (Versi 1.0.0)
* **Baseline MAE (Median):** `0.8740` (`USD 87,403.99`)
* **Model MAE:** `0.5332` (`USD 53,320.01`)
* **Model RMSE:** `0.7456` (`USD 74,558.14`)
* **Model R^2:** `0.5758`

## Pengujian Skenario
1. **Profil Valid A:** Pengujian menggunakan nilai input bawaan (*default*) melalui tombol **Prediksi Harga**.
2. **Profil Valid B:** Pengujian dengan mengubah nilai fitur valid (misalnya `MedInc` dan `HouseAge`) untuk melihat perubahan prediksi.
3. **Input Tidak Valid:** Pengujian aturan logika (misalnya `AveBedrms` lebih besar dari `AveRooms`) guna memastikan muncul pesan *error* yang jelas tanpa *crash*.
4. **Input Valid Ekstrem:** Pengujian kombinasi nilai ekstrem/di luar pola umum untuk memverifikasi respons peringatan model.

## Interpretasi & Keterbatasan
* Hasil prediksi berupa estimasi median harga rumah untuk kelompok blok sensus dengan karakteristik serupa pada data sensus historis California (tahun 1990).
* Model tidak mencerminkan harga pasar properti terkini maupun kondisi fisik rumah secara individual, sehingga hanya digunakan sebagai gambaran awal dan bukan penilaian properti resmi.