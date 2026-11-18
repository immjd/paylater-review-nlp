"""Topic modeling (BERTopic) pada ulasan NEGATIF: keluhan apa yang paling sering muncul, per aplikasi?

Masuk : data/interim/reviews_clean.csv   (dari preprocess.py; seluruh ulasan, bukan hanya dataset seimbang)
Keluar:
  reports/topik_ringkasan.csv     satu baris per topik: kata kunci, jumlah, persen per aplikasi (TANPA teks ulasan)
  reports/kategori_ringkasan.csv  kategori hasil penggabungan topik bernama sama (butuh reports/topik_nama.csv)
  reports/topik_laporan.txt       laporan yang sama dengan yang tercetak, tanpa contoh ulasan
  data/processed/topik_contoh.txt contoh ulasan per topik + sampel outlier (teks mentah, tidak ikut git; TIDAK dicetak ke terminal)
  data/processed/topik_dokumen.csv topik tiap teks unik (tidak ikut git)
  data/interim/emb_*.npy          cache embedding, supaya menyetel parameter tidak menghitung ulang

Cara kerjanya
  1. "Keluhan" = ulasan berbintang 1-2 (sentiment == neg). Ini label LEMAH: sebagian ulasan berbintang 1-2
     isinya bukan keluhan. Ulasan yang terlalu pendek (--min-kata) dibuang karena hanya menghasilkan topik sampah.
  2. Topik DICARI pada teks unik (tanpa salinan), supaya ulasan salin-tempel tidak membengkokkan struktur topik.
  3. Topik dihitung ke SEMUA ulasan negatif, termasuk salinannya: tiap salinan mewarisi topik dari teks yang sama
     dan dihitung ke aplikasinya sendiri. Jadi persen = porsi ulasan, bukan porsi teks berbeda (kolom pct_teks_unik
     disediakan sebagai pembanding).
  4. Teks untuk topik dibuat ulang dari `content`: EMOJI DIBUANG (preprocess.py mengubah emoji jadi kata seperti
     "wajah menangis", yang membuat ulasan pendek lolos batas panjang dan membuat topik berdasarkan emoji, bukan isi),
     frasa soal bintang ("kasih bintang 1 dulu") dibuang, lalu dibersihkan dengan fungsi yang sama dengan preprocess.py.
     Embedding dari teks itu; kata kunci dari versi tanpa stopword (negasi dipertahankan).

Catatan: tiap aplikasi di-scrape pada rentang waktu berbeda (lihat README), jadi bandingkan PERSEN, bukan jumlah.

Jalankan dari akar proyek:  python src\\topic_model.py
"""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paths import INTERIM, PROCESSED, ROOT
from preprocess import RE_ANGKA, RE_EMAIL, RE_TELP, bersihkan, buat_kunci, muat_sastrawi
from slang import EXTRA_STOP, FALLBACK_STOP, PROTECTED

try:
    import emoji
except ImportError:
    emoji = None

APP = {"com.finaccel.android": "Kredivo", "io.silvrr.installment": "Akulaku", "id.co.myhomecredit": "HomeCredit"}

# Kata yang muncul di hampir semua keluhan sehingga tidak membedakan topik. Mulai dari daftar pendek;
# tambah lewat --stop-tambah setelah melihat hasil pertama. Negasi (tidak, belum, ...) sengaja TIDAK dibuang.
STOP_TOPIK = {
    "kredivo", "akulaku", "homecredit", "home", "credit", "finaccel", "paylater",
    "aplikasi", "aplikasinya", "apk", "app", "tolong", "mohon", "banget", "sangat", "sekali", "bintang", "rating",
}

OUT, CONTOH = [], []


def cetak(teks="", ke=None):
    print(teks)
    (OUT if ke is None else ke).append(str(teks))


def aman(teks, maks=230):
    s = RE_EMAIL.sub("<email>", str(teks))
    s = RE_TELP.sub("<telepon>", s)
    s = RE_ANGKA.sub("<nomor>", s)
    s = " ".join(s.split())
    return s if len(s) <= maks else s[: maks - 3] + "..."


# ---------------------------------------------------------------- teks untuk topik
# "maaf saya kasih bintang 1 dulu" membentuk klaster berdasarkan GAYA kalimat, bukan keluhannya.
RE_BINTANG = re.compile(
    r"\b(?:(?:kasih|ngasih|kasi|ngasi|memberi|memberikan|beri)\s+)?(?:bintang|bintangnya|rating|ratingnya)\b"
    r"(?:\s+(?:\d|satu|dua|tiga|empat|lima))?(?:\s+dulu)?")


def teks_topik(content):
    """content mentah -> teks bersih tanpa emoji dan tanpa frasa pemberian bintang."""
    s = str(content)
    if emoji is not None:
        s = emoji.replace_emoji(s, replace=" ")
    s = RE_BINTANG.sub(" ", bersihkan(s))
    return " ".join(s.split())


# ---------------------------------------------------------------- data
def muat(args):
    df = pd.read_csv(args.inp, dtype={"reviewId": str}).fillna({"content": "", "text": "", "text_klasik": ""})
    perlu = {"app_id", "reviewId", "sentiment", "status", "content", "text", "text_klasik", "n_kata"}
    if not perlu <= set(df.columns):
        raise SystemExit(f"{args.inp} kekurangan kolom: {sorted(perlu - set(df.columns))}")
    neg = df[(df["sentiment"] == "neg") & df["status"].isin(["unik", "duplikat"])].copy()
    neg["kunci"] = neg["text"].map(buat_kunci)           # kunci yang sama dengan yang dipakai preprocess.py (untuk salinan)
    _, stop_sastrawi = muat_sastrawi()
    stop = ((stop_sastrawi or set(FALLBACK_STOP)) | EXTRA_STOP) - PROTECTED
    neg["teks_topik"] = neg["content"].map(teks_topik)
    kt = neg["teks_topik"].map(buat_kunci)
    neg["n_topik"] = kt.str.split().str.len()
    neg["klasik_topik"] = kt.map(lambda k: " ".join(w for w in k.split() if w not in stop))
    unik = neg[(neg["status"] == "unik") & (neg["n_topik"] >= args.min_kata)].reset_index(drop=True)
    if unik["kunci"].duplicated().any():
        raise SystemExit("kunci ganda di antara teks unik: jalankan ulang preprocess.py")
    if len(unik) < 50:
        raise SystemExit(f"hanya {len(unik)} teks unik yang lolos penyaringan: terlalu sedikit untuk topic modeling")
    return df, neg, unik


# ---------------------------------------------------------------- embedding
def buat_embedding(docs, nama, seed):
    h = hashlib.md5(nama.encode())
    for d in docs:
        h.update(d.encode("utf-8", "ignore"))
        h.update(b"\x00")
    slug = nama.replace("/", "_").replace("\\", "_")
    cache = INTERIM / f"emb_{slug}_{h.hexdigest()[:10]}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(docs):
            print(f"embedding dimuat dari cache: {cache.name}")
            return emb
    t0 = time.time()
    if nama == "tfidf-svd":      # tanpa unduhan model; kualitas jauh lebih rendah; untuk uji cepat saja
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        x = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(docs)
        emb = normalize(TruncatedSVD(n_components=min(64, x.shape[1] - 1), random_state=seed).fit_transform(x))
    else:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("Jalankan dulu: pip install sentence-transformers bertopic")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"menghitung embedding dengan {nama} di {device} ({len(docs):,} teks) ...")
        model = SentenceTransformer(nama, device=device)
        emb = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    print(f"embedding selesai dalam {time.time() - t0:.0f} detik, disimpan: {cache.name}")
    return emb


# ---------------------------------------------------------------- model topik
def latih_topik(unik, emb, args):
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as e:
        raise SystemExit(f"Paket topic modeling belum terpasang ({e}).\nJalankan: pip install bertopic sentence-transformers")
    n = len(unik)
    mcs = args.min_cluster or max(30, round(0.004 * n))
    stop = sorted(STOP_TOPIK | {w.strip() for w in args.stop_tambah.split(",") if w.strip()})
    docs = unik["klasik_topik"].tolist()
    umap_model = UMAP(n_neighbors=args.umap_neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=args.seed)
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=min(args.min_samples, mcs), metric="euclidean",
                  cluster_selection_method=args.seleksi, prediction_data=True)
    # BERTopic menjalankan vectorizer pada SATU dokumen gabungan per topik, sehingga min_df di sana berarti
    # "muncul di minimal sekian TOPIK" dan membuang kata khas satu topik. Karena itu kosakata disaring dulu
    # di tingkat ulasan (min_df = jumlah teks unik), lalu dikunci lewat vocabulary=.
    dasar = CountVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=args.min_df).fit(docs)
    vec = CountVectorizer(ngram_range=(1, 2), stop_words=stop, vocabulary=dasar.vocabulary_)
    print(f"kosakata kata kunci: {len(dasar.vocabulary_):,} kata/bigram (muncul di >= {args.min_df} teks unik)")
    ctf = ClassTfidfTransformer(reduce_frequent_words=True)
    tm = BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=hdb, vectorizer_model=vec, ctfidf_model=ctf,
                  nr_topics=(args.nr_topics or None), top_n_words=10, calculate_probabilities=False, verbose=False)
    t0 = time.time()
    topik, _ = tm.fit_transform(docs, embeddings=emb)
    topik = np.asarray(topik)
    banyak = len(set(topik) - {-1})
    print(f"BERTopic selesai dalam {time.time() - t0:.0f} detik: {banyak} topik, min_cluster_size={mcs}, "
          f"outlier {100 * (topik == -1).mean():.0f}%")
    if args.kurangi_outlier and (topik == -1).any():
        baru = tm.reduce_outliers(docs, topik.tolist(), strategy="embeddings", embeddings=emb)
        tm.update_topics(docs, topics=baru, vectorizer_model=vec, ctfidf_model=ctf)
        topik = np.asarray(baru)
        print(f"setelah --kurangi-outlier: outlier {100 * (topik == -1).mean():.0f}%")
    kata = {t: [w for w, _ in tm.get_topic(t)] for t in sorted(set(topik)) if t != -1}
    return topik, kata, mcs


def contoh_terdekat(unik, emb, topik, t, k=3):
    idx = np.where(topik == t)[0]
    pusat = emb[idx].mean(0)
    sim = emb[idx] @ pusat / (np.linalg.norm(pusat) + 1e-12)
    return idx[np.argsort(-sim)[:k]]


# ---------------------------------------------------------------- laporan
def laporan(df, neg, unik, topik, kata, emb, args):
    apps = [a for a in APP if a in set(neg["app_id"])] + sorted(set(neg["app_id"]) - set(APP))
    nama_app = {a: APP.get(a, a) for a in apps}
    nama_topik = {}
    if args.nama and Path(args.nama).exists():
        nm = pd.read_csv(args.nama)
        nama_topik = {int(r.topik): str(r.nama) for r in nm.itertuples() if pd.notna(r.nama)}

    unik = unik.assign(topik=topik)
    peta = dict(zip(unik["kunci"], unik["topik"]))
    neg = neg.assign(topik=neg["kunci"].map(peta))
    dianalisis = neg[neg["topik"].notna()].copy()
    dianalisis["topik"] = dianalisis["topik"].astype(int)

    cetak("=" * 78)
    cetak("CAKUPAN")
    cetak("=" * 78)
    tot_app = neg.groupby("app_id").size().reindex(apps, fill_value=0)
    dia_app = dianalisis.groupby("app_id").size().reindex(apps, fill_value=0)
    cetak(f"  {'aplikasi':<12}{'ulasan negatif':>16}{'dianalisis':>12}{'%':>6}")
    for a in apps:
        p = 100 * dia_app[a] / tot_app[a] if tot_app[a] else 0
        cetak(f"  {nama_app[a]:<12}{int(tot_app[a]):>16,}{int(dia_app[a]):>12,}{p:>5.0f}%")
    cetak(f"  'ulasan negatif' = bintang 1-2 berstatus unik/duplikat. 'dianalisis' = teksnya (atau salinannya) masuk model:")
    cetak(f"  minimal {args.min_kata} kata (emoji tidak dihitung). Teks lebih pendek tidak dianalisis; hasil di bawah TIDAK mewakili keluhan satu-dua kata.")
    cetak(f"  Teks unik yang dimodelkan: {len(unik):,} | topik: {len(kata)} | outlier (tak masuk topik): {100 * (topik == -1).mean():.0f}% teks unik")

    ct = pd.crosstab(dianalisis["topik"], dianalisis["app_id"]).reindex(columns=apps, fill_value=0)
    tot = ct.sum(axis=1)
    pct_app = ct.div(ct.sum(axis=0), axis=1) * 100
    pct_all = tot / tot.sum() * 100
    nu = unik["topik"].value_counts()
    urut = [t for t in tot.index if t != -1]
    urut.sort(key=lambda t: -tot[t])
    if -1 in tot.index:
        urut.append(-1)

    baris = []
    for t in urut:
        row = {"topik": t, "nama": nama_topik.get(t, "(lainnya / tak masuk topik)" if t == -1 else ""),
               "kata_kunci": ", ".join(kata.get(t, [])[:8]),
               "n_teks_unik": int(nu.get(t, 0)), "pct_teks_unik": round(100 * nu.get(t, 0) / len(unik), 1),
               "n_ulasan": int(tot[t]), "pct_ulasan": round(float(pct_all[t]), 1)}
        for a in apps:
            row[f"n_{nama_app[a]}"] = int(ct.loc[t, a])
            row[f"pct_{nama_app[a]}"] = round(float(pct_app.loc[t, a]), 1)
        baris.append(row)
    ring = pd.DataFrame(baris)

    cetak("\n" + "=" * 78)
    cetak("TOPIK KELUHAN (persen = porsi ulasan negatif yang dianalisis, termasuk salinan)")
    cetak("=" * 78)
    kepala = f"  {'id':>3} {'%semua':>7} {'ulasan':>7}  " + " ".join(f"{nama_app[a][:9]:>9}" for a in apps) + "   kata kunci"
    cetak(kepala)
    for r in baris:
        nm = r["nama"] or ", ".join(kata.get(r["topik"], [])[:6])
        if r["topik"] == -1:
            nm = r["nama"]
        cetak(f"  {r['topik']:>3} {r['pct_ulasan']:>6.1f}% {r['n_ulasan']:>7,}  "
              + " ".join(f"{r['pct_' + nama_app[a]]:>8.1f}%" for a in apps) + f"   {nm}")

    cetak("\n" + "=" * 78)
    cetak("3 KELUHAN TERATAS PER APLIKASI (tanpa 'lainnya')")
    cetak("=" * 78)
    for a in apps:
        kolom = pct_app[a].drop(index=-1, errors="ignore").sort_values(ascending=False).head(3)
        cetak(f"  {nama_app[a]}:")
        for t, v in kolom.items():
            nm = nama_topik.get(t) or ", ".join(kata.get(t, [])[:5])
            cetak(f"    {v:5.1f}%  topik {t}: {nm}")
    cetak("\n  Catatan: tiap aplikasi di-scrape pada rentang waktu berbeda; bandingkan persen, bukan jumlah.")
    cetak("  Topik adalah KELOMPOK TEKS MIRIP dari model, bukan kategori resmi: nama dan batasnya perlu dibaca manual.")

    kategori = None
    if nama_topik:
        pemetaan = pd.Series({t: nama_topik.get(t, "(belum diberi nama)") for t in urut if t != -1})
        ct_k = ct.drop(index=-1, errors="ignore")
        grup = ct_k.groupby(pemetaan).sum()
        tot_app_all = ct.sum(axis=0)                       # penyebut: semua ulasan yang dianalisis, termasuk outlier
        pct_k = grup.div(tot_app_all, axis=1) * 100
        tot_k = grup.sum(axis=1)
        urutan_k = list(tot_k.sort_values(ascending=False).index)
        daftar_topik = pemetaan.groupby(pemetaan).apply(lambda x: "+".join(str(i) for i in x.index))
        cetak("\n" + "=" * 78)
        cetak("KATEGORI KELUHAN (topik bernama sama digabung; persen dari semua ulasan negatif yang dianalisis)")
        cetak("=" * 78)
        cetak(f"  {'%semua':>7} {'ulasan':>7}  " + " ".join(f"{nama_app[a][:9]:>9}" for a in apps) + "   kategori [topik]")
        baris_k = []
        for k in urutan_k:
            n_k = int(tot_k[k])
            row = {"kategori": k, "topik": daftar_topik[k], "n_ulasan": n_k, "pct_ulasan": round(100 * n_k / tot_app_all.sum(), 1)}
            for a in apps:
                row[f"n_{nama_app[a]}"] = int(grup.loc[k, a])
                row[f"pct_{nama_app[a]}"] = round(float(pct_k.loc[k, a]), 1)
            baris_k.append(row)
            cetak(f"  {row['pct_ulasan']:>6.1f}% {n_k:>7,}  " + " ".join(f"{row['pct_' + nama_app[a]]:>8.1f}%" for a in apps)
                  + f"   {k} [{daftar_topik[k]}]")
        if -1 in ct.index:
            n_o = int(ct.loc[-1].sum())
            row = {"kategori": "(lainnya / tak masuk topik)", "topik": "-1", "n_ulasan": n_o, "pct_ulasan": round(100 * n_o / tot_app_all.sum(), 1)}
            for a in apps:
                row[f"n_{nama_app[a]}"] = int(ct.loc[-1, a])
                row[f"pct_{nama_app[a]}"] = round(100 * ct.loc[-1, a] / tot_app_all[a], 1)
            baris_k.append(row)
            cetak(f"  {row['pct_ulasan']:>6.1f}% {n_o:>7,}  " + " ".join(f"{row['pct_' + nama_app[a]]:>8.1f}%" for a in apps) + "   (lainnya / tak masuk topik)")
        cetak("\n  3 kategori teratas per aplikasi (tanpa 'lainnya'):")
        for a in apps:
            atas = pct_k[a].sort_values(ascending=False).head(3)
            cetak(f"    {nama_app[a]}: " + "; ".join(f"{k} {v:.1f}%" for k, v in atas.items()))
        kategori = pd.DataFrame(baris_k)

    # contoh ulasan: HANYA ke berkas lokal (tidak ke terminal dan tidak ke laporan yang ikut git)
    rng = np.random.default_rng(args.seed)

    def baris(i):
        return f"   - ({nama_app.get(unik.loc[i, 'app_id'], '')}, {int(unik.loc[i, 'score'])} bintang) {aman(unik.loc[i, 'content'])}"

    CONTOH.append("CONTOH ULASAN PER TOPIK. Email/telepon/angka disamarkan, NAMA ORANG TIDAK: jangan dipublikasikan mentah.")
    CONTOH.append("[khas] = paling dekat ke pusat topik (ulasan paling tipikal). [acak] = sampel acak dari topik yang sama (menunjukkan sebaran).")
    for t in [x for x in urut if x != -1]:
        CONTOH.append(f"\n[topik {t}] {nama_topik.get(t, '')} | {', '.join(kata.get(t, [])[:8])} | {int((topik == t).sum()):,} teks unik")
        khas = contoh_terdekat(unik, emb, topik, t, 3)
        sisa = np.setdiff1d(np.where(topik == t)[0], khas)
        acak = rng.choice(sisa, size=min(args.contoh_acak, len(sisa)), replace=False) if len(sisa) else []
        CONTOH.extend("  [khas]" + baris(i) for i in khas)
        CONTOH.extend("  [acak]" + baris(i) for i in acak)
    idx = np.where(topik == -1)[0]
    if len(idx):
        pilih = rng.choice(idx, size=min(args.contoh_outlier, len(idx)), replace=False)
        CONTOH.append(f"\n[outlier = tidak masuk topik mana pun] {len(idx):,} teks unik; sampel acak {len(pilih)}:")
        CONTOH.extend("  [acak]" + baris(i) for i in pilih)
    return ring, unik, dianalisis, kategori


def main():
    ap = argparse.ArgumentParser(description="BERTopic pada ulasan negatif.")
    ap.add_argument("--inp", default=str(INTERIM / "reviews_clean.csv"))
    ap.add_argument("--embed-model", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="nama model sentence-transformers, atau 'tfidf-svd' untuk uji cepat tanpa unduhan")
    ap.add_argument("--min-kata", type=int, default=5, help="buang ulasan lebih pendek dari ini (jumlah kata)")
    ap.add_argument("--min-cluster", type=int, default=0, help="min_cluster_size HDBSCAN; 0 = otomatis (0,4%% dari jumlah teks, minimal 30)")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--nr-topics", type=int, default=15, help="gabungkan topik menjadi sekian; 0 = jangan digabung")
    ap.add_argument("--min-df", type=int, default=5, help="kata/bigram harus muncul di minimal sekian TEKS UNIK untuk bisa jadi kata kunci")
    ap.add_argument("--seleksi", choices=["eom", "leaf"], default="eom",
                    help="cara HDBSCAN memilih klaster: eom = klaster besar (default), leaf = klaster kecil yang lebih homogen")
    ap.add_argument("--contoh-acak", type=int, default=4, help="jumlah contoh acak per topik di berkas contoh")
    ap.add_argument("--contoh-outlier", type=int, default=15, help="jumlah contoh acak dari kelompok outlier")
    ap.add_argument("--stop-tambah", default="", help="stopword tambahan untuk kata kunci, pisah koma")
    ap.add_argument("--kurangi-outlier", action="store_true", help="paksa teks outlier ke topik terdekat")
    ap.add_argument("--nama", default=None, help="CSV opsional: kolom topik,nama. Topik bernama sama digabung jadi satu kategori. "
                    "Default: reports/topik_nama[_tag].csv")
    ap.add_argument("--out-kategori", default=str(ROOT / "reports" / "kategori_ringkasan.csv"))
    ap.add_argument("--out-csv", default=str(ROOT / "reports" / "topik_ringkasan.csv"))
    ap.add_argument("--out-txt", default=str(ROOT / "reports" / "topik_laporan.txt"))
    ap.add_argument("--out-contoh", default=str(PROCESSED / "topik_contoh.txt"))
    ap.add_argument("--out-dok", default=str(PROCESSED / "topik_dokumen.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="akhiran nama berkas keluaran (mis. leaf), supaya beberapa percobaan tidak saling menimpa")
    args = ap.parse_args()
    if args.nama is None:
        args.nama = str(ROOT / "reports" / (f"topik_nama_{args.tag}.csv" if args.tag else "topik_nama.csv"))
    if args.tag:
        for k in ("out_csv", "out_txt", "out_contoh", "out_dok", "out_kategori"):
            pth = Path(getattr(args, k))
            setattr(args, k, str(pth.with_name(f"{pth.stem}_{args.tag}{pth.suffix}")))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    df, neg, unik = muat(args)
    print(f"ulasan negatif (unik+salinan): {len(neg):,} | teks unik >= {args.min_kata} kata: {len(unik):,}")
    emb = buat_embedding(unik["teks_topik"].tolist(), args.embed_model, args.seed)
    topik, kata, mcs = latih_topik(unik, emb, args)
    cetak(f"Pengaturan: embed={args.embed_model} | min_kata={args.min_kata} | min_cluster_size={mcs} | nr_topics={args.nr_topics or 'bebas'} "
          f"| outlier dikurangi={'ya' if args.kurangi_outlier else 'tidak'} | seed={args.seed}")
    ring, unik, dianalisis, kategori = laporan(df, neg, unik, topik, kata, emb, args)

    for p in (args.out_csv, args.out_txt, args.out_contoh, args.out_dok, args.out_kategori):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    ring.to_csv(args.out_csv, index=False, encoding="utf-8")
    if kategori is not None:
        Path(args.out_kategori).parent.mkdir(parents=True, exist_ok=True)
        kategori.to_csv(args.out_kategori, index=False, encoding="utf-8")
    Path(args.out_txt).write_text("\n".join(OUT) + "\n", encoding="utf-8")
    Path(args.out_contoh).write_text("\n".join(CONTOH) + "\n", encoding="utf-8")
    unik[["reviewId", "app_id", "score", "at", "topik", "text"]].to_csv(args.out_dok, index=False, encoding="utf-8")
    print(f"\nTersimpan:\n  {args.out_csv}\n  {args.out_txt}\n  {args.out_contoh}  (contoh ulasan; lokal, tidak ikut git)\n  {args.out_dok}  (lokal, tidak ikut git)")
    print("\nLangkah berikutnya: beri nama tiap topik di reports\\topik_nama.csv (kolom: topik,nama), lalu jalankan ulang skrip ini.")


if __name__ == "__main__":
    main()
