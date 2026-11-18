"""Tampilkan baris sampel validasi yang bermasalah, untuk menyetel pola kategori_keluhan.py.

Dua bagian (teks ulasan hanya dicetak ke terminal, tidak disimpan ke mana pun):
  1. TERLEWAT       : ulasan 'tanpa kategori spesifik' yang kamu beri benar = 1 (ada keluhan spesifik yang tidak tertangkap)
  2. SALAH TANGKAP  : baris kategori yang kamu beri benar = 0, hanya untuk kategori dengan presisi di bawah --ambang

Jalankan dari akar proyek:
  python src\\lihat_validasi.py
  python src\\lihat_validasi.py --ambang 0.9 --panjang 200
"""
import argparse

import pandas as pd

from kategori_keluhan import LAINNYA, baca_nilai
from paths import LABELED


def main():
    ap = argparse.ArgumentParser(description="Lihat baris validasi yang bermasalah.")
    ap.add_argument("--xlsx", default=str(LABELED / "validasi_kategori.xlsx"))
    ap.add_argument("--ambang", type=float, default=0.8, help="tampilkan salah tangkap hanya untuk kategori dengan presisi di bawah ini")
    ap.add_argument("--panjang", type=int, default=140, help="potong teks ulasan sekian karakter")
    args = ap.parse_args()

    x = pd.read_excel(args.xlsx, sheet_name="sampel", dtype=str).fillna("")
    x["nilai"] = x["benar"].map(baca_nilai)

    def potong(t):
        t = " ".join(str(t).split())
        return t if len(t) <= args.panjang else t[: args.panjang - 3] + "..."

    lain = x[x["kategori_diuji"] == LAINNYA]
    ada = lain[lain["nilai"] == 1]
    print(f"== TERLEWAT: {len(ada)} dari {int(lain['nilai'].isin([0, 1]).sum())} ulasan tanpa kategori ==")
    for r in ada.itertuples():
        print(f"  ({r.app}) {potong(r.teks)}")

    kat = x[(x["kategori_diuji"] != LAINNYA) & x["nilai"].isin([0, 1])]
    pres = kat.groupby("kategori_diuji")["nilai"].mean()
    buruk = [k for k in pres.index if pres[k] < args.ambang]
    print(f"\n== SALAH TANGKAP: kategori dengan presisi < {args.ambang:.0%}: {', '.join(buruk) if buruk else '(tidak ada)'} ==")
    for k in buruk:
        print(f"\n[{k}] presisi {pres[k]:.0%}")
        for r in kat[(kat["kategori_diuji"] == k) & (kat["nilai"] == 0)].itertuples():
            print(f"  <<{r.pemicu}>> {potong(r.teks)}")


if __name__ == "__main__":
    main()
