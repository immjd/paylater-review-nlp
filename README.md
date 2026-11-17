# Suara Pengguna Paylater

Analisis sentimen dan topik keluhan pada ulasan aplikasi kredit digital Indonesia di Google Play.

> Status: dalam pengerjaan. Bagian bertanda TODO diisi setelah hasilnya ada.

## Pertanyaan yang dijawab
1. Seberapa besar selisih performa baseline klasik (TF-IDF + Logistic Regression) dengan IndoBERT hasil fine-tuning?
2. Keluhan apa yang paling sering muncul, dan apakah berbeda antar aplikasi?

## Data
- Sumber: ulasan publik Google Play, diambil dengan `google-play-scraper` (`lang="id"`, `country="id"`).
- Aplikasi (verifikasi dulu id-nya di halaman Play Store):

  | Aplikasi | Package id |
  |---|---|
  | Kredivo | `com.finaccel.android` |
  | Akulaku | `io.silvrr.installment` |
  | My Home Credit | `id.co.myhomecredit` |

- Tanggal scraping: TODO
- Jumlah ulasan mentah / setelah cleaning: TODO
- Pengambilan sampel: dibagi rata per bintang (1-5), jadi distribusi kelas bukan distribusi asli di Play Store. Karena itu metrik utama adalah macro-F1.
- Label: bintang dipetakan ke sentimen (1-2 negatif, 3 netral, 4-5 positif) untuk training; test set dilabeli manual (TODO: jumlah).
- Privasi: nama dan foto pengguna tidak disimpan; nomor telepon, email, dan nomor rekening di dalam teks di-mask sebelum data dibagikan.

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
pip install google-play-scraper==1.2.7 pandas
python src\scrape_reviews.py com.finaccel.android io.silvrr.installment id.co.myhomecredit --per-score 100
```
Uji dulu dengan `--per-score 100`; kalau lancar, naikkan ke 1500.

## Hasil
TODO: tabel perbandingan TF-IDF + LR vs IndoBERT (macro-F1, per kelas), topik keluhan utama per aplikasi, link demo.
