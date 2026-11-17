"""Bedah kesalahan model baseline: apakah yang keliru MODELNYA, atau LABEL bintangnya?

Masuk : data/processed/pred_baseline_test.csv   (dari train_baseline.py)
Keluar: tercetak di terminal DAN disimpan ke data/processed/analisis_error.txt (UTF-8, tidak ikut git)

Isi laporan
  1. Bintang asli x sentimen yang diprediksi (persen per baris)
  2. Ketepatan menurut panjang ulasan
  3. Kesalahan PALING YAKIN per sel (mis. bintang 1-2 tapi diprediksi positif): ini tempat label
     berisik paling mudah ketahuan
  4. Sampel kesalahan ACAK: untuk memperkirakan seberapa sering label bintang itu yang keliru
     (kesalahan paling yakin cenderung berlebihan menampilkan label berisik, jadi jangan dipakai sendirian)

Email dan nomor telepon di teks disamarkan sebelum dicetak. Baca contohnya dengan satu pertanyaan:
"kalau saya membaca teksnya saja tanpa melihat bintang, sentimen apa yang saya beri?"

Jalankan dari akar proyek:  python src\\analisis_error.py
"""
import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

from paths import PROCESSED
from preprocess import RE_ANGKA, RE_EMAIL, RE_TELP, buat_kunci

NAMA = ["neg", "neu", "pos"]
APP = {"com.finaccel.android": "Kredivo", "id.co.myhomecredit": "HomeCredit", "io.silvrr.installment": "Akulaku"}


def aman(teks, maks):
    s = RE_EMAIL.sub("<email>", str(teks))
    s = RE_TELP.sub("<telepon>", s)
    s = RE_ANGKA.sub("<nomor>", s)
    s = " ".join(s.split())
    return s if len(s) <= maks else s[: maks - 3] + "..."


def contoh(r, maks):
    kepala = (f"[{APP.get(r['app_id'], r['app_id'])} | bintang {r['score']} | {NAMA[r['label']]}->{NAMA[r['pred']]} "
              f"| yakin {r['yakin']:.2f} | {r['n_kata']} kata]")
    return f"{kepala}\n    {aman(r['content'], maks)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default=str(PROCESSED / "pred_baseline_test.csv"))
    ap.add_argument("--out", default=str(PROCESSED / "analisis_error.txt"))
    ap.add_argument("--per-sel", type=int, default=5, help="contoh paling yakin per sel kesalahan")
    ap.add_argument("--acak", type=int, default=20, help="jumlah kesalahan acak")
    ap.add_argument("--maks-char", type=int, default=220, help="panjang maksimum teks yang dicetak")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # terminal Windows kadang tidak bisa mencetak emoji: jangan sampai berhenti karena itu
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    df = pd.read_csv(args.inp, dtype={"reviewId": str}).fillna({"content": "", "text": ""})
    df["n_kata"] = df["text"].map(lambda t: len(buat_kunci(t).split()))
    df["yakin"] = df[["p_neg", "p_neu", "p_pos"]].max(axis=1)
    df["benar"] = df["label"] == df["pred"]
    salah = df[~df["benar"]]

    baris = []
    tulis = baris.append
    tulis(f"== ringkasan ==\nulasan test {len(df)} | benar {df['benar'].mean():.3f} | salah {len(salah)}")

    tulis("\n== bintang asli x sentimen yang diprediksi (persen per baris) ==")
    tab = pd.crosstab(df["score"], df["pred_sentiment"], normalize="index").reindex(columns=NAMA, fill_value=0).round(2)
    tulis(tab.to_string())

    tulis("\n== ketepatan menurut panjang ulasan ==")
    kelompok = pd.cut(df["n_kata"], [-1, 3, 7, 15, 10**6], labels=["<=3 kata", "4-7", "8-15", ">=16"])
    per_panjang = df.groupby(kelompok, observed=True).agg(jumlah=("benar", "size"), ketepatan=("benar", "mean"))
    per_panjang["porsi_test"] = per_panjang["jumlah"] / len(df)
    tulis(per_panjang.round(3).to_string())

    tulis(f"\n== KESALAHAN PALING YAKIN per sel ({args.per_sel} teratas; 'yakin' = peluang kelas yang diprediksi) ==")
    for t, p in product(range(3), range(3)):
        if t == p:
            continue
        sel = salah[(salah["label"] == t) & (salah["pred"] == p)]
        tulis(f"\n-- sebenarnya {NAMA[t].upper()} (dari bintang), diprediksi {NAMA[p].upper()}: {len(sel)} ulasan --")
        for _, r in sel.nlargest(args.per_sel, "yakin").iterrows():
            tulis(contoh(r, args.maks_char))

    n_acak = min(args.acak, len(salah))
    tulis(f"\n== {n_acak} KESALAHAN ACAK (seed {args.seed}) ==")
    acak = salah.sample(n_acak, random_state=args.seed).sort_values(["label", "pred"])
    for _, r in acak.iterrows():
        tulis(contoh(r, args.maks_char))

    teks = "\n".join(baris) + "\n"
    print(teks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(teks, encoding="utf-8")
    print(f"Tersimpan (UTF-8): {out}")


if __name__ == "__main__":
    main()
