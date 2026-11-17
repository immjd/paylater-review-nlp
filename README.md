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
pip install google-play-scraper==1.2.7 pandas
python src\scrape_reviews.py com.finaccel.android io.silvrr.installment id.co.myhomecredit --per-score 100
```

Uji dulu dengan `--per-score 100`; kalau lancar, naikkan ke 1500.

## Hasil

TODO: tabel perbandingan TF-IDF + LR vs IndoBERT (macro-F1, per kelas), topik keluhan utama per aplikasi, link demo.
