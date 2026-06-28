# ⚡ Energy Efficiency - Cooling Load Predictor & Analytics

Repositori ini berisi proyek lengkap **Mini Project 1 (MP_01)** dan **Mini Project 2 (MP_02)** untuk menganalisis karakteristik fisik bangunan dan memprediksi **Beban Pendinginan (Cooling Load)** ruangan. Proyek ini mencakup pengembangan model Machine Learning (Ridge Regression & Random Forest) serta Deep Learning (Artificial Neural Network - ANN), serta penyediaan aplikasi web interaktif berbasis **Streamlit** untuk visualisasi data dan kalkulator prediksi.

---

## 🧭 Struktur Direktori Proyek

*   `MP_01_(Rizki_Dwi_Permadi)_Fixed.ipynb` — Notebook pelatihan model Machine Learning (Ridge Regression & Random Forest Regressor) beserta evaluasinya.
*   `MP_02_(Rizki_Dwi_Permadi)_Fixed.ipynb` — Notebook pelatihan model Deep Learning (ANN menggunakan Keras & TensorFlow) beserta perbandingan kinerja dari ketiga model secara komparatif.
*   `app.py` — File kode utama aplikasi web Streamlit multi-page dengan visualisasi interaktif dan menu prediksi multi-model.
*   `energy_efficiency_data.csv` — Dataset karakteristik fisik bangunan dan beban energi (target).
*   `model_mp2.keras` — Berkas model Artificial Neural Network (ANN) Keras tersimpan secara langsung.
*   `model_pipeline_mp2.joblib` — Berkas serialisasi Joblib berisi `RobustScaler` untuk normalisasi data, model Linear Regression (Ridge), dan model Random Forest Regressor yang siap pakai.
*   `requirements.txt` — Daftar pustaka (dependencies) Python yang dibutuhkan untuk menjalankan repositori ini.
*   `.gitignore` — Pengaturan file git agar file cache (`__pycache__/`, `.ipynb_checkpoints/`) tidak ikut diunggah ke GitHub.

---

## 📊 Hasil Evaluasi & Perbandingan Model

Berdasarkan hasil pelatihan dan pengujian pada data uji (154 sampel), berikut perbandingan metrik kinerja model:

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | R² Score (Akurasi / R-Squared) |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 2.2748 | 3.2034 | 88.21% |
| **Random Forest Regressor** | **1.1428** | **1.7957** | **96.29%** |
| **Artificial Neural Network (ANN)** | 1.6189 | 2.3646 | 93.57% |

*Catatan: Random Forest Regressor memiliki akurasi tertinggi ($R^2$ sebesar 96.29%), diikuti oleh ANN (93.57%) dan Ridge Regression (88.21%).*

---

## 🚀 Panduan Menjalankan Aplikasi Web Streamlit

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi web interaktif di komputer lokal Anda:

### 1. Buat dan Aktifkan Virtual Environment (Opsional)
Sangat direkomendasikan untuk menggunakan lingkungan virtual (venv atau conda) agar tidak terjadi bentrok versi library.

**Menggunakan venv:**
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# atau
venv\Scripts\activate     # Untuk Windows
```

**Menggunakan Conda:**
```bash
conda create -n env_efficiency python=3.10
conda activate env_efficiency
```

### 2. Instalasi Library / Pustaka Pendukung
Jalankan perintah berikut untuk menginstal semua library yang terdaftar pada file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Jalankan Server Streamlit
Jalankan aplikasi web dengan perintah berikut:
```bash
streamlit run app.py
```
Setelah server aktif, aplikasi akan otomatis terbuka di browser Anda pada alamat default: `http://localhost:8501`.

---

## 🎨 Fitur Utama Aplikasi Streamlit

Aplikasi ini dirancang dengan gaya visual modern premium, responsif, dan terbagi menjadi dua halaman utama melalui navigasi sidebar:

1.  **📊 Halaman Analisis & Visualisasi Data**:
    *   **Ringkasan Statistik**: Menampilkan ukuran dataset serta statistika deskriptif (mean, std, min, max) seluruh variabel arsitektur bangunan secara dinamis.
    *   **Eksplorasi Fitur Interaktif**: Menu dropdown untuk memilih karakteristik fisik bangunan (seperti luas dinding, rasio luas kaca, orientasi, dll) dan merender grafik scatter plot Plotly secara interaktif untuk menganalisis hubungannya dengan Cooling Load.
    *   **Perbandingan Model**: Visualisasi grafik batang akurasi ($R^2$) dari ketiga model agar pengguna dapat membandingkan performa model secara komparatif.
2.  **🔮 Halaman Prediksi Cooling Load**:
    *   Kalkulator interaktif berbasis form.
    *   Pengguna dapat memilih model prediksi yang ingin digunakan (`ANN`, `Ridge`, atau `Random Forest`).
    *   Normalisasi input pengguna dilakukan secara otomatis menggunakan `RobustScaler` terintegrasi, dan menampilkan estimasi beban pendinginan (Cooling Load) beserta interpretasi praktisnya untuk desain bangunan hemat energi.
