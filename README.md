# 🥬🍎 Food & Vegetable Freshness Classification (EfficientNetB3)

Proyek Deep Learning untuk mengklasifikasikan tingkat kesegaran bahan makanan, buah, dan sayuran ke dalam dua kategori utama (**BAIK / Segar** dan **JELEK / Busuk**) menggunakan pendekatan *Transfer Learning* dan *Fine-Tuning* pada arsitektur **EfficientNetB3**.

---

## 📊 Ringkasan Dataset
Model dilatih menggunakan gabungan dua dataset publik dari Kaggle dengan total **33.072 gambar** yang dibagi menjadi 80% data latih (*training*) dan 20% data validasi (*validation*):
* **Sumber Dataset:**
  1. [Fresh and Stale Classification](https://kaggle.com)
  2. [Fresh and Spoiled Food Image Dataset](https://kaggle.com)
* **Distribusi Kelas:**
  * **BAIK (Fresh):** 15.737 gambar
  * **JELEK (Stale / Spoiled / Rotten):** 17.335 gambar
* **Pembagian Data:** 26.458 gambar *training* & 6.614 gambar *validation* (resolusi `224x224` piksel).

---

## 🧠 Arsitektur & Pelatihan Model
Pelatihan dilakukan menggunakan TensorFlow/Keras di Google Colab (GPU T4) melalui dua tahap:

1. **Tahap 1 — Feature Extraction (12 Epoch):**
   * Menggunakan *base model* **EfficientNetB3** (*pretrained weights* ImageNet) dalam kondisi *frozen*.
   * Dilengkapi lapisan augmentasi otomatis (`RandomFlip`, `RandomRotation`, `RandomZoom`), `GlobalAveragePooling2D`, dan `Dropout(0.2)`.
   * Optimizer: `Adam` (`learning_rate = 0.0001`).
   * **Hasil Tahap 1:** Akurasi validasi mencapai **93,47%** (*val_loss*: `0.1881`).

2. **Tahap 2 — Fine-Tuning (8 Epoch):**
   * Membuka seluruh lapisan (*unfreeze*) pada arsitektur EfficientNetB3 untuk penyesuaian bobot secara menyeluruh.
   * Optimizer: `Adam` dengan *learning rate* lebih kecil (`0.00001`).
   * **Hasil Akhir:** Akurasi *training* **97,47%** dan akurasi validasi mencapai **97,16%** (*val_loss*: `0.0711`).

---

## 🚀 Cara Menjalankan Proyek
1. **Clone repositori ini:**
   ```bash
   git clone https://github.com
   ```
2. **Buka file notebook (`.ipynb`)** menggunakan Google Colab atau Jupyter Notebook.
3. **Siapkan API Kaggle:** Unggah file `kaggle.json` milik Anda saat diminta pada sel pertama untuk mengunduh dataset secara otomatis.
4. **Jalankan seluruh sel** secara berurutan hingga menghasilkan file model akhir (`model_efficientnet_sakti.keras`).

---

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman:** Python 3
* **Framework Deep Learning:** TensorFlow & Keras
* **Arsitektur Model:** EfficientNetB3
* **Lingkungan Komputasi:** Google Colaboratory (GPU T4)
