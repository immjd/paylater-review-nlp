# Suara Pengguna Paylater

Analisis sentimen dan topik keluhan pada ulasan aplikasi kredit digital Indonesia di Google Play.

> Status: dalam pengerjaan. Bagian bertanda TODO diisi setelah hasilnya ada.

## Pertanyaan yang dijawab

1. Seberapa besar selisih performa baseline klasik (TF-IDF + Logistic Regression) dengan IndoBERT hasil fine-tuning?
2. Keluhan apa yang paling sering muncul, dan apakah berbeda antar aplikasi?

## Data

- Sumber: ulasan publik Google Play, diambil dengan `google-play-scraper` (`lang="id"`, `country="id"`).
- Aplikasi (verifikasi dulu id-nya di halaman Play Store):

  | Aplikasi       | Package id              |
  | -------------- | ----------------------- |
  | Kredivo        | `com.finaccel.android`  |
  | Akulaku        | `io.silvrr.installment` |
  | My Home Credit | `id.co.myhomecredit`    |

- Tanggal scraping: 17 November 2026
- Jendela waktu (sama untuk kelima bintang): Kredivo sejak 2026-03-01, Akulaku sejak 2026-04-01, Home Credit sejak 2025-01-01
- Ulasan mentah: 98.389 (Kredivo 49.351; Akulaku 32.417; Home Credit 16.621); setelah cleaning: TODO
- Pengambilan data: seluruh ulasan yang memuat teks dalam jendela itu, jadi distribusi bintangnya asli (sangat condong ke bintang 5). Sampel seimbang dibuat di tahap sampling.
- Label: bintang dipetakan ke sentimen (1-2 negatif, 3 netral, 4-5 positif) untuk training; test set dilabeli manual berdasarkan isi teks (TODO: jumlah).
- Privasi: nama dan foto pengguna tidak disimpan; PII di dalam teks sangat jarang (1 email pada pemeriksaan awal) dan tetap di-mask sebelum data dibagikan.

## Struktur folder

```
paylater-review-project/
├── data/
│   ├── raw/         hasil scraping (satu CSV per aplikasi)
│   ├── interim/     setelah cleaning dan normalisasi
│   ├── labeled/     sampel berlabel manual (test set)
│   └── processed/   train / val / test final
├── notebooks/       eksplorasi dan eksperimen
├── src/             skrip (paths.py, scrape_reviews.py, ...)
├── models/          checkpoint model (tidak di-commit)
├── reports/figures/ grafik untuk laporan
└── app/             demo Streamlit / Hugging Face Spaces
```

## Cara menjalankan

Windows (PowerShell), dari folder proyek:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install google-play-scraper==1.2.7 "pandas>=2.2" scikit-learn Sastrawi emoji matplotlib jupyter

# 1. kumpulkan ulasan (jendela waktu disamakan per aplikasi; total sekitar 45-60 menit)
python src\scrape_reviews.py id.co.myhomecredit --since 2025-01-01
python src\scrape_reviews.py io.silvrr.installment --since 2026-04-01
python src\scrape_reviews.py com.finaccel.android --since 2026-03-01

# 2. bersihkan teks, buang duplikat
python src\preprocess.py

# 3. sampel seimbang + split train/val/test
python src\make_dataset.py

# 4. baseline TF-IDF + Logistic Regression
python src\train_baseline.py
```

Folder `data/` dan `models/` tidak ikut repo. Ulasan baru terus muncul di Google Play, jadi scraping ulang
di waktu lain akan menghasilkan data yang sedikit berbeda dari yang dipakai di bagian Hasil.

## Hasil

TODO: tabel perbandingan TF-IDF + LR vs IndoBERT (macro-F1, per kelas), topik keluhan utama per aplikasi, link demo.
