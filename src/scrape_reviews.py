"""Scrape ulasan Google Play (bahasa Indonesia) ke CSV, per aplikasi dan per bintang.

Dua mode (path data selalu relatif terhadap folder proyek: data/raw/<app_id>.csv):

1) Jendela waktu yang sama untuk semua bintang (disarankan; dipakai di proyek ini):
    python src/scrape_reviews.py com.finaccel.android --since 2026-03-01
   Setiap bintang diambil dari yang terbaru sampai tanggal --since, jadi kelima bintang
   mencakup periode yang sama. Cukup satu aplikasi per perintah.

2) Jumlah tetap per bintang (lama):
    python src/scrape_reviews.py com.finaccel.android --per-score 1500

app_id = nilai `id=` di URL halaman Play Store, mis.
    https://play.google.com/store/apps/details?id=com.finaccel.android

Catatan:
- Dengan --since, bintang yang sudah selesai dicatat di data/raw/<app_id>.state.json.
  Kalau perintah terhenti (diblok, laptop tidur), jalankan PERINTAH YANG SAMA lagi:
  hanya bintang yang belum selesai yang diambil ulang.
- Dengan --since, baris di CSV yang lebih lama dari tanggal itu dibuang supaya isi file
  konsisten dengan jendelanya.
- google-play-scraper menelan error jaringan/rate-limit dan mengembalikan list kosong,
  jadi halaman kosong ditangani sebagai "mungkin diblok": tunggu, lalu ulang dari token terakhir.
- userName & userImage sengaja TIDAK disimpan (privasi).
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

from paths import RAW

KEEP = ["reviewId", "content", "score", "thumbsUpCount", "reviewCreatedVersion", "at"]
SAFETY_CAP = 100_000  # batas pengaman per bintang kalau memakai --since


def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"format tanggal harus YYYY-MM-DD, bukan {s!r}")


def fetch(app_id, score, cap, lang, country, since=None, batch=100):
    """Ambil ulasan terbaru untuk satu aplikasi dan satu bintang.

    Berhenti saat `since` terlewati, halaman terakhir tercapai, atau `cap` terpenuhi.
    Mengembalikan (rows, lengkap). lengkap=False berarti berhenti sebelum mencapai target
    (diblok, atau cap habis sebelum tanggal --since).
    """
    rows, good_token, empty_streak, pages = [], None, 0, 0
    while len(rows) < cap:
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
            pages += 1
            empty_streak, good_token = 0, token
            lewat = since is not None and min(r["at"] for r in page) < since
            if since is not None:
                page = [r for r in page if r["at"] >= since]
            rows += page
            if pages % 10 == 0 and rows:
                print(f"  {app_id} bintang {score}: {len(rows)} ulasan (sampai {rows[-1]['at']:%Y-%m-%d})")
            if lewat or token.token is None:  # tanggal since terlewati / halaman terakhir
                return rows[:cap], True
            if len(rows) >= cap:
                return rows[:cap], since is None
            time.sleep(random.uniform(1.5, 3.5))  # pelan-pelan supaya tidak diblok
        else:
            empty_streak += 1
            if empty_streak > 3:
                print(f"  bintang {score}: halaman kosong terus, kemungkinan diblok sementara")
                return rows[:cap], False
            wait = 20 * empty_streak
            print(f"  halaman kosong, tunggu {wait}s lalu ulang dari token terakhir")
            time.sleep(wait)
    return rows[:cap], since is None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_ids", nargs="+")
    ap.add_argument("--since", type=parse_date, help="ambil ulasan sampai tanggal ini (YYYY-MM-DD)")
    ap.add_argument("--per-score", type=int, help="batas ulasan per bintang (default 1500 tanpa --since)")
    ap.add_argument("--lang", default="id")
    ap.add_argument("--country", default="id")
    ap.add_argument("--out", default=str(RAW), help="folder keluaran (default: data/raw di folder proyek)")
    args = ap.parse_args()

    since = args.since
    since_str = since.strftime("%Y-%m-%d") if since else None
    cap = args.per_score or (SAFETY_CAP if since else 1500)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for app_id in args.app_ids:
        path = out / f"{app_id}.csv"
        state_path = out / f"{app_id}.state.json"

        df = pd.read_csv(path, parse_dates=["at"]) if path.exists() else None
        if df is not None and since is not None:
            sebelum = len(df)
            df = df[df["at"] >= since]
            if sebelum - len(df):
                print(f"{app_id}: membuang {sebelum - len(df)} ulasan lama di luar jendela (< {since_str})")

        selesai = set()
        if since is not None and state_path.exists():
            st = json.loads(state_path.read_text())
            if st.get("since") == since_str:
                selesai = set(st.get("selesai", []))

        belum_lengkap = []
        for score in range(1, 6):
            if score in selesai:
                print(f"  bintang {score}: sudah selesai sebelumnya, dilewati")
                continue
            rows, lengkap = fetch(app_id, score, cap, args.lang, args.country, since)
            if rows:
                new = pd.DataFrame(rows).reindex(columns=KEEP).assign(app_id=app_id)
                df = new if df is None else pd.concat([df, new])
                df = df.drop_duplicates("reviewId")
                df.to_csv(path, index=False)  # simpan tiap selesai satu bintang
            if since is not None and lengkap:
                selesai.add(score)
                state_path.write_text(json.dumps({"since": since_str, "selesai": sorted(selesai)}))
            elif not lengkap:
                belum_lengkap.append(score)

        if df is not None:
            ringkas = df.groupby("score")["at"].agg(jumlah="count", terlama="min", terbaru="max")
            print(f"\n{app_id}: {len(df)} ulasan tersimpan di {path}")
            print(ringkas.to_string())
        if belum_lengkap:
            print(
                f"PERINGATAN: bintang {belum_lengkap} belum mencapai target. "
                "Jalankan perintah yang sama lagi untuk melanjutkan bintang tersebut."
            )


if __name__ == "__main__":
    main()
