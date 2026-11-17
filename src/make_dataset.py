"""Membuat dataset sentimen yang seimbang lalu membaginya menjadi train / val / test.

Masuk : data/interim/reviews_clean.csv   (hanya baris berstatus "unik" yang dipakai)
Keluar: data/processed/dataset.csv       (kolom `split` = train/val/test, `label` = 0 neg, 1 neu, 2 pos)

Aturan sampling
  1. PER APLIKASI, jumlah ulasan di tiap bintang (1-5) disamakan, sebanyak ulasan unik pada bintang
     TERLANGKA di aplikasi itu (bisa dibatasi dengan --per-bintang).
     Kenapa per aplikasi, bukan digabung? Kalau bintang 5 didominasi satu aplikasi dan bintang 1
     didominasi aplikasi lain, model bisa belajar "nama aplikasi = sentimen" tanpa membaca keluhannya.
     Dengan sampel seimbang per aplikasi, nama aplikasi tidak membawa informasi sentimen.
  2. Split stratified per (aplikasi x bintang), default 70 / 15 / 15.
  3. Pemeriksaan otomatis: tidak ada reviewId ganda dan tidak ada teks yang sama di split berbeda.

Label sentimen di sini masih label LEMAH dari bintang (1-2 neg, 3 neu, 4-5 pos). Label benar dari
membaca teks dibuat terpisah untuk subset test (gold set).

Jalankan dari akar proyek:  python src\\make_dataset.py
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import INTERIM, PROCESSED
from preprocess import buat_kunci

LABEL = {"neg": 0, "neu": 1, "pos": 2}
SKOR = [1, 2, 3, 4, 5]
KOLOM = ["split", "app_id", "reviewId", "score", "sentiment", "label", "at",
         "content", "text", "text_klasik", "n_kata"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default=str(INTERIM / "reviews_clean.csv"), help="file hasil preprocess.py")
    ap.add_argument("--out", default=str(PROCESSED), help="folder keluaran (default: data/processed)")
    ap.add_argument("--per-bintang", type=int, default=None,
                    help="batas atas ulasan per aplikasi x bintang (default: otomatis = bintang terlangka per aplikasi)")
    ap.add_argument("--val", type=float, default=0.15, help="porsi validasi")
    ap.add_argument("--test", type=float, default=0.15, help="porsi test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype={"reviewId": str}).fillna({"text": "", "text_klasik": ""})
    pool = df[df["status"] == "unik"]
    print(f"bahan sampling: {len(pool)} ulasan unik dari {len(df)} ulasan")

    # 1. jumlah per aplikasi x bintang
    cacah = pd.crosstab(pool["app_id"], pool["score"]).reindex(columns=SKOR, fill_value=0)
    kosong = cacah[(cacah == 0).any(axis=1)]
    if len(kosong):
        raise SystemExit(f"Ada bintang tanpa ulasan unik sama sekali:\n{kosong}\nSampel seimbang tidak bisa dibuat.")
    n_app = cacah.min(axis=1)
    if args.per_bintang:
        n_app = n_app.clip(upper=args.per_bintang)
    print("\n== ulasan per bintang yang diambil, per aplikasi ==")
    print(n_app.rename("per_bintang").to_string())

    bagian = [g.sample(n=int(n_app[app]), random_state=args.seed)
              for (app, _), g in pool.groupby(["app_id", "score"])]
    sampel = pd.concat(bagian)
    sampel["label"] = sampel["sentiment"].map(LABEL)

    # 2. split stratified per aplikasi x bintang
    strata = sampel["app_id"] + "|" + sampel["score"].astype(str)
    sisa_porsi = args.val + args.test
    train, sisa = train_test_split(sampel, test_size=sisa_porsi, stratify=strata, random_state=args.seed)
    val, test = train_test_split(sisa, test_size=args.test / sisa_porsi,
                                 stratify=strata.loc[sisa.index], random_state=args.seed)
    hasil = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    hasil = hasil.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 3. pemeriksaan kebocoran
    kunci = hasil["text"].map(buat_kunci)
    assert hasil["reviewId"].is_unique, "reviewId ganda"
    assert kunci.is_unique, "ada teks yang sama di dataset (bisa bocor antar split)"
    assert (kunci != "").all(), "ada teks kosong"
    assert len(hasil) == len(sampel)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hasil[KOLOM].to_csv(out / "dataset.csv", index=False)

    urut = ["train", "val", "test"]
    print(f"\n== ukuran split (total {len(hasil)}) ==")
    print(hasil["split"].value_counts().reindex(urut).to_string())
    print("\n== split x sentimen ==")
    print(pd.crosstab(hasil["split"], hasil["sentiment"]).reindex(urut)[["neg", "neu", "pos"]].to_string())
    print("\n== aplikasi x sentimen (harus proporsinya sama di tiap aplikasi) ==")
    print(pd.crosstab(hasil["app_id"], hasil["sentiment"], normalize="index").round(3)[["neg", "neu", "pos"]].to_string())
    print("\n== split x aplikasi ==")
    print(pd.crosstab(hasil["split"], hasil["app_id"]).reindex(urut).to_string())
    print("\n== ulasan pendek & panjang per sentimen ==")
    ringkas = hasil.groupby("sentiment")["n_kata"].agg(median="median", share_3_kata_atau_kurang=lambda s: (s <= 3).mean())
    print(ringkas.loc[["neg", "neu", "pos"]].round(3).to_string())
    print(f"\nTersimpan: {out / 'dataset.csv'}")


if __name__ == "__main__":
    main()
