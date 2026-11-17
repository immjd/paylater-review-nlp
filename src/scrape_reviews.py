"""Scrape ulasan Google Play (bahasa Indonesia) ke CSV, per aplikasi dan per bintang.

Jalankan dari mana saja (path data selalu relatif terhadap folder proyek):
    python src/scrape_reviews.py com.finaccel.android io.silvrr.installment id.co.myhomecredit --per-score 100

Hasil: data/raw/<app_id>.csv

app_id = nilai `id=` di URL halaman Play Store, mis.
    https://play.google.com/store/apps/details?id=com.finaccel.android

Catatan:
- Diambil per bintang (1..5) supaya ulasan 1-2 bintang (sumber keluhan) cukup banyak.
  Kalau tidak, sort NEWEST didominasi bintang 5.
- google-play-scraper menelan error jaringan/rate-limit dan mengembalikan list kosong,
  jadi halaman kosong ditangani sebagai "mungkin diblok": tunggu, lalu ulang dari token terakhir.
- userName & userImage sengaja TIDAK disimpan (privasi).
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

from paths import RAW

KEEP = ["reviewId", "content", "score", "thumbsUpCount", "reviewCreatedVersion", "at"]


def fetch(app_id, score, target, lang, country, batch=100):
    """Ambil sampai `target` ulasan untuk satu aplikasi dan satu bintang."""
    rows, good_token, empty_streak = [], None, 0
    while len(rows) < target:
        page, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=batch,
            filter_score_with=score,
            continuation_token=good_token,
        )
        if page:
            rows += page
            good_token, empty_streak = token, 0
            print(f"  {app_id} bintang {score}: {len(rows)} ulasan")
            if token.token is None:  # halaman terakhir
                break
            time.sleep(random.uniform(1.5, 3.5))  # pelan-pelan supaya tidak diblok
        else:
            empty_streak += 1
            if empty_streak > 3:
                print("  kosong terus: kemungkinan diblok sementara atau memang sudah habis")
                break
            wait = 20 * empty_streak
            print(f"  halaman kosong, tunggu {wait}s lalu ulang dari token terakhir")
            time.sleep(wait)
    return rows[:target]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_ids", nargs="+")
    ap.add_argument("--per-score", type=int, default=1500, help="target ulasan per bintang per aplikasi")
    ap.add_argument("--lang", default="id")
    ap.add_argument("--country", default="id")
    ap.add_argument("--out", default=str(RAW), help="folder keluaran (default: data/raw di folder proyek)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for app_id in args.app_ids:
        path = out / f"{app_id}.csv"
        df = pd.read_csv(path) if path.exists() else None  # lanjut dari hasil sebelumnya
        for score in range(1, 6):
            rows = fetch(app_id, score, args.per_score, args.lang, args.country)
            if not rows:
                continue
            new = pd.DataFrame(rows).reindex(columns=KEEP).assign(app_id=app_id)
            df = new if df is None else pd.concat([df, new])
            df = df.drop_duplicates("reviewId")
            df.to_csv(path, index=False)  # simpan tiap selesai satu bintang
        if df is not None:
            print(f"{app_id}: {len(df)} ulasan tersimpan di {path}")
            print(df["score"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
