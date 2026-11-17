"""Cleaning ulasan: data/raw/*.csv -> data/interim/reviews_clean.csv

Jalankan dari mana saja:
    python src/preprocess.py            # cepat
    python src/preprocess.py --stem     # tambah stemming Sastrawi untuk kolom text_klasik (lambat)

Kolom keluaran (semua baris disimpan; yang dibuang ditandai di kolom `status`, bukan dihapus):
    text         teks bersih untuk IndoBERT (huruf kecil, PII di-mask, emoji jadi kata, slang dinormalisasi)
    text_klasik  untuk TF-IDF: tanda baca dan stopword dibuang (negasi dilindungi), opsional stem
    n_kata       jumlah kata pada text
    status       unik      -> satu wakil dari tiap teks (dipakai untuk sampling)
                 duplikat  -> salinan lain dari teks yang sama dengan sentimen mayoritas
                 minoritas -> salinan dengan sentimen BERBEDA dari mayoritas teks itu (label berisik)
                 ambigu    -> teks yang sama tanpa sentimen mayoritas yang jelas (mis. "baik" 50% netral, 50% positif)
                 kosong    -> tidak ada teks setelah dibersihkan

Sebuah teks punya sentimen mayoritas kalau porsinya >= --konsensus (default 0.8). Contoh: "mantap"
muncul 3.000 kali dan 1 kali bernilai bintang 3: mayoritas positif (99,9%), jadi satu wakil jadi `unik`,
sisanya `duplikat`, dan yang bintang 3 jadi `minoritas`. Teks tanpa mayoritas jelas menjadi `ambigu`.
    sentiment    neg (bintang 1-2), neu (3), pos (4-5): label LEMAH dari bintang

Duplikat dicari lintas aplikasi, supaya teks yang sama tidak bisa muncul di train dan test.
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from paths import INTERIM, RAW
from slang import EXTRA_STOP, FALLBACK_STOP, PROTECTED, SLANG

try:
    import emoji
except ImportError:  # tetap jalan, emoji dibiarkan apa adanya
    emoji = None

SENTIMEN = {1: "neg", 2: "neg", 3: "neu", 4: "pos", 5: "pos"}

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_TELP = re.compile(r"(?<!\d)(?:\+?62|0)[\s.-]?8(?:[\s.-]?\d){8,12}(?!\d)")
RE_ANGKA = re.compile(r"(?<!\d)\d(?:[\s.-]?\d){9,}(?!\d)")  # >= 10 digit, boleh dengan spasi/strip
RE_ULANG_HURUF = re.compile(r"([a-z])\1{2,}")  # baguuus -> bagus
RE_ULANG_TANDA = re.compile(r"([!?.,])\1+")  # !!! -> !
RE_REDUP = re.compile(r"\b([a-z]{2,})2\b")  # kata2 -> kata kata
RE_KATA = re.compile(r"[a-z]+")
RE_SPASI = re.compile(r"\s+")


@lru_cache(maxsize=None)
def nama_emoji(e):
    nama = emoji.demojize(e, language="id").strip(":").replace("_", " ")
    return nama if nama != e else ""


def emoji_ke_kata(s):
    """Ubah emoji jadi kata. Emoji yang sama berturut-turut (👍👍👍) dihitung satu."""
    if emoji is None or s.isascii():
        return s
    bagian, akhir, sebelumnya = [], 0, None
    for m in emoji.emoji_list(s):
        bagian.append(s[akhir : m["match_start"]])
        e = m["emoji"]
        if not (e == sebelumnya and m["match_start"] == akhir):
            bagian.append(f" {nama_emoji(e)} ")
        sebelumnya, akhir = e, m["match_end"]
    bagian.append(s[akhir:])
    return "".join(bagian)


def bersihkan(teks):
    s = unicodedata.normalize("NFKC", str(teks)).lower()
    s = RE_URL.sub(" ", s)
    s = RE_EMAIL.sub(" <email> ", s)
    s = RE_TELP.sub(" <telepon> ", s)
    s = RE_ANGKA.sub(" <nomor> ", s)
    s = emoji_ke_kata(s)
    s = RE_ULANG_HURUF.sub(r"\1", s)
    s = RE_REDUP.sub(r"\1 \1", s)
    s = RE_KATA.sub(lambda m: SLANG.get(m.group(0), m.group(0)), s)
    s = RE_ULANG_TANDA.sub(r"\1", s)
    return RE_SPASI.sub(" ", s).strip()


def buat_kunci(teks):
    return RE_SPASI.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", teks)).strip()


def muat_sastrawi():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except ImportError:
        return None, None
    return StemmerFactory().create_stemmer(), set(StopWordRemoverFactory().get_stop_words())


def kandidat_slang(teks_list, stemmer, stop, n):
    """Kata yang sering muncul tapi tidak ada di kamus akar kata Sastrawi, bukan stopword, dan belum di SLANG."""
    kamus = stemmer.delegatedStemmer.dictionary
    df_kata = Counter(k for t in teks_list for k in set(RE_KATA.findall(t)))
    hasil = []
    for kata, jumlah in df_kata.most_common():
        if jumlah < 5:
            break
        if len(kata) < 2 or kata in SLANG or kata in stop:
            continue
        if kamus.contains(kata) or kamus.contains(stemmer.stem(kata)):
            continue
        hasil.append((kata, jumlah))
        if len(hasil) >= n:
            break
    return hasil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW), help="folder CSV mentah (default: data/raw)")
    ap.add_argument("--out", default=str(INTERIM), help="folder keluaran (default: data/interim)")
    ap.add_argument("--stem", action="store_true", help="stemming Sastrawi pada text_klasik (lambat)")
    ap.add_argument("--slang-top", type=int, default=60, help="jumlah kandidat slang yang dicetak")
    ap.add_argument("--seed", type=int, default=42, help="seed untuk memilih wakil duplikat")
    ap.add_argument("--konsensus", type=float, default=0.8,
                    help="porsi minimum sentimen mayoritas agar teks kembar dianggap berlabel jelas (default 0.8)")
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("*.csv"))
    if not files:
        sys.exit(f"Tidak ada CSV di {args.raw}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stemmer, stop_sastrawi = muat_sastrawi()
    if args.stem and stemmer is None:
        sys.exit("--stem butuh Sastrawi: pip install Sastrawi")
    stop = ((stop_sastrawi or set(FALLBACK_STOP)) | EXTRA_STOP) - PROTECTED
    if emoji is None:
        print("PERINGATAN: paket `emoji` belum terpasang (pip install emoji); emoji dibiarkan apa adanya.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["content"] = df["content"].fillna("").astype(str)
    df["sentiment"] = df["score"].map(SENTIMEN)
    print(f"membersihkan {len(df)} ulasan dari {len(files)} file ...")

    df["text"] = df["content"].map(bersihkan)
    kunci = df["text"].map(buat_kunci)
    df["n_kata"] = kunci.str.split().str.len()
    kata_klasik = kunci.map(lambda k: [w for w in k.split() if w not in stop])
    if args.stem:
        kata_klasik = kata_klasik.map(lambda ws: [stemmer.stem(w) for w in ws])
    df["text_klasik"] = kata_klasik.map(" ".join)

    # status duplikat: sentimen mayoritas per teks, lalu satu wakil acak (seed tetap, supaya tidak
    # condong ke satu aplikasi) dari baris yang sentimennya sama dengan mayoritas
    acak = df.assign(kunci=kunci).sample(frac=1, random_state=args.seed)
    hitung = acak.groupby(["kunci", "sentiment"]).size().rename("n").reset_index()
    hitung["porsi"] = hitung["n"] / hitung.groupby("kunci")["n"].transform("sum")
    terbanyak = hitung.loc[hitung.groupby("kunci")["n"].idxmax()].set_index("kunci")
    acak["mayoritas"] = acak["kunci"].map(terbanyak["sentiment"])
    acak["porsi_mayoritas"] = acak["kunci"].map(terbanyak["porsi"])

    jelas = acak["porsi_mayoritas"] >= args.konsensus
    sama = acak["sentiment"] == acak["mayoritas"]
    urutan = acak[jelas & sama].groupby("kunci").cumcount()
    status = pd.Series("unik", index=acak.index)
    status[~jelas] = "ambigu"
    status[jelas & ~sama] = "minoritas"
    status.loc[urutan[urutan > 0].index] = "duplikat"
    status[acak["kunci"] == ""] = "kosong"
    df["status"] = status.reindex(df.index)

    kolom = ["app_id", "reviewId", "score", "sentiment", "at", "thumbsUpCount", "reviewCreatedVersion",
             "content", "text", "text_klasik", "n_kata", "status"]
    df[kolom].to_csv(out / "reviews_clean.csv", index=False)

    unik = df[df["status"] == "unik"]
    print("\n== status ==")
    print(df["status"].value_counts().to_string())
    print("\n== ulasan UNIK per aplikasi x bintang (bahan sampling) ==")
    print(pd.crosstab(unik["app_id"], unik["score"]).to_string())
    print("\n== ulasan UNIK per aplikasi x sentimen ==")
    print(pd.crosstab(unik["app_id"], unik["sentiment"])[["neg", "neu", "pos"]].to_string())
    print(f"\nshare <= 3 kata: semua ulasan {(df['n_kata'] <= 3).mean():.1%} | ulasan unik {(unik['n_kata'] <= 3).mean():.1%}")
    print("jumlah kata di ulasan unik (median per bintang):")
    print(unik.groupby("score")["n_kata"].median().to_string())

    berubah = df[(df["content"].str.len() > 15) & (df["content"].str.lower().str.strip() != df["text"])]
    print("\n== contoh sebelum -> sesudah (hanya tampil di terminal) ==")
    for _, r in berubah.sample(min(8, len(berubah)), random_state=1).iterrows():
        print(f"  {r['content'][:90]!r}\n  -> {r['text'][:90]!r}")

    if stemmer is not None:
        cand = kandidat_slang(unik["text"].tolist(), stemmer, stop, 300)
        pd.DataFrame(cand, columns=["kata", "jumlah_ulasan"]).to_csv(out / "slang_kandidat.csv", index=False)
        print(f"\n== {args.slang_top} kandidat slang (lengkapnya di {out / 'slang_kandidat.csv'}) ==")
        print(", ".join(f"{k}({c})" for k, c in cand[: args.slang_top]))
    else:
        print("\n(Sastrawi belum terpasang: daftar kandidat slang dilewati. pip install Sastrawi)")

    print(f"\nTersimpan: {out / 'reviews_clean.csv'}")


if __name__ == "__main__":
    main()
