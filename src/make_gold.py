"""Gold set: ulasan test yang dilabeli berdasarkan ISI TEKS, bukan bintang.

Tahap (jalankan dari akar proyek)
  1) python src\\make_gold.py                  membuat data/labeled/gold_untuk_dilabeli.xlsx
     (lembar berisi teks saja: tanpa bintang, tanpa aplikasi, tanpa prediksi model, supaya pelabelan buta)
     Kamu mengisi kolom label_manual (ada dropdown: neg / neu / pos / x).
  2) python src\\make_gold.py --ekspor-teks    OPSIONAL: membuat data/labeled/gold_label_llm.xlsx (kolom id, teks,
     label_llm, alasan; tanpa bintang dan tanpa labelmu) untuk diisi Claude, misalnya lewat add-in Excel.
  3) python src\\make_gold.py --gabung         memeriksa isian, lalu mencetak
       - kesepakatan label manual vs label LLM pada baris yang dilabeli keduanya (+ Cohen's kappa)
       - seberapa sering label bintang cocok dengan label gold
       - nilai baseline pada gold set: dengan label bintang vs dengan label gold
     dan menyimpan data/labeled/gold.csv. Boleh dijalankan kapan saja, juga saat baru sebagian terisi.

Aturan label gold per ulasan: label manual kalau ada; kalau tidak ada, label LLM (kolom `sumber` mencatat
asalnya). Baris yang ditandai x oleh manusia selalu dibuang. Selalu laporkan sumber label apa adanya.

Sampel diambil dari split TEST (stratified per aplikasi x bintang) lalu diacak, jadi k baris pertama
selalu merupakan sampel acak yang sah. Lembar tidak akan ditimpa kalau sudah ada (pakai --timpa).
File di data/labeled/ tidak ikut git.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from paths import LABELED, PROCESSED
from preprocess import RE_ANGKA, RE_EMAIL, RE_TELP

NAMA = ["neg", "neu", "pos"]
KODE = NAMA + ["x"]
IDX = {n: i for i, n in enumerate(NAMA)}
RE_KONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

PANDUAN = [
    ("Label berdasarkan ISI TEKS saja. Abaikan perkiraanmu soal bintang (bintangnya memang tidak ditampilkan).", ""),
    ("neg", "Pengguna mengeluh, kecewa, atau menyampaikan masalah. Termasuk keluhan tanpa kata emosi "
            "(\"tidak bisa login\", \"pengajuan ditolak\", \"akun diblokir\") dan pertanyaan bernada keluhan "
            "(\"kenapa pengajuan ditolak padahal bayar tepat waktu?\")."),
    ("pos", "Memuji, puas, berterima kasih, atau merekomendasikan."),
    ("neu", "Tidak menilai: pertanyaan netral, permintaan atau saran tanpa keluhan (\"ingin ada penawaran limit\"), "
            "harapan (\"semoga bisa cair\"), atau pujian dan keluhan yang benar-benar seimbang."),
    ("x", "Tidak bisa dinilai: teks tidak bermakna, terlalu terpotong, atau bahasa yang tidak kamu pahami. "
          "Baris x dibuang dari evaluasi."),
    ("Campuran", "Kalau ada pujian DAN keluhan, pilih yang paling dominan. Hanya kalau seimbang, pilih neu."),
    ("Kolom catatan", "Boleh diisi untuk kasus yang kamu ragukan; berguna nanti untuk analisis dan README."),
]


def samarkan(teks):
    s = RE_KONTROL.sub(" ", str(teks))
    s = RE_EMAIL.sub("<email>", s)
    s = RE_TELP.sub("<telepon>", s)
    s = RE_ANGKA.sub("<nomor>", s)
    return " ".join(s.split())


def buat(args):
    xlsx = Path(args.xlsx)
    if xlsx.exists() and not args.timpa:
        raise SystemExit(f"{xlsx} sudah ada (mungkin sudah kamu isi) dan TIDAK ditimpa.\n"
                         "Pakai --timpa hanya kalau memang mau membuat ulang dari nol.")
    df = pd.read_csv(args.dataset, dtype={"reviewId": str}).fillna({"content": "", "text": "", "text_klasik": ""})
    te = df[df["split"] == "test"]
    porsi = min(1.0, args.n / len(te))
    bagian = [g.sample(n=max(1, round(len(g) * porsi)), random_state=args.seed)
              for _, g in te.groupby(["app_id", "score"])]
    s = pd.concat(bagian).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    s.insert(0, "id", [f"g{i + 1:04d}" for i in range(len(s))])

    kunci = s[["id", "reviewId", "app_id", "score", "sentiment"]].rename(columns={"sentiment": "label_bintang"})
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    kunci.to_csv(args.kunci, index=False)

    wb = Workbook()
    ws = wb.active
    ws.title = "label"
    ws.append(["id", "teks", "label_manual", "catatan"])
    for i, t in zip(s["id"], s["content"]):
        ws.append([i, samarkan(t), None, None])
    n = len(s)
    dv = DataValidation(type="list", formula1='"neg,neu,pos,x"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Isian tidak valid", error="Isi dengan neg, neu, pos, atau x.")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{n + 1}")
    for col, lebar in zip("ABCD", (9, 105, 15, 36)):
        ws.column_dimensions[col].width = lebar
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDE8F7")
    for row in ws.iter_rows(min_row=2, max_row=n + 1):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        for c in (row[0], row[2], row[3]):
            c.alignment = Alignment(vertical="top")
    ws.freeze_panes = "A2"

    pj = wb.create_sheet("panduan")
    pj.column_dimensions["A"].width = 16
    pj.column_dimensions["B"].width = 110
    for a, b in PANDUAN:
        pj.append([a, b])
    for row in pj.iter_rows():
        row[0].font = Font(bold=True)
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(xlsx)
    print(f"Lembar label ({n} ulasan, diacak) : {xlsx}")
    print(f"Kunci bintang (JANGAN dibuka)    : {args.kunci}")
    print("Isi kolom label_manual dengan neg / neu / pos / x (ada dropdown). Panduan ada di sheet 'panduan'.")


def nilai(y_true, y_pred):
    return (f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
            accuracy_score(y_true, y_pred))


def ekspor_teks(args):
    out = Path(args.llm)
    if out.exists() and not args.timpa:
        raise SystemExit(f"{out} sudah ada dan TIDAK ditimpa (mungkin sudah berisi label LLM).\n"
                         "Pakai --timpa hanya kalau memang mau membuat ulang dari nol.")
    lembar = pd.read_excel(args.xlsx, sheet_name="label", dtype=str)[["id", "teks"]].fillna("")
    wb = Workbook()
    ws = wb.active
    ws.title = "label"
    ws.append(["id", "teks", "label_llm", "alasan"])
    for i, t in zip(lembar["id"], lembar["teks"]):
        ws.append([i, t, None, None])
    n = len(lembar)
    dv = DataValidation(type="list", formula1='"neg,neu,pos,x"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Isian tidak valid", error="Isi dengan neg, neu, pos, atau x.")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{n + 1}")
    for col, lebar in zip("ABCD", (9, 105, 14, 40)):
        ws.column_dimensions[col].width = lebar
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="E6F2E6")
    for row in ws.iter_rows(min_row=2, max_row=n + 1):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        for c in (row[0], row[2], row[3]):
            c.alignment = Alignment(vertical="top")
    ws.freeze_panes = "A2"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"{n} ulasan (hanya id dan teks, tanpa bintang dan tanpa labelmu): {out}")
    print("Isi kolom label_llm (neg / neu / pos / x), lalu simpan. Kolom alasan opsional.")


SINONIM = {"negatif": "neg", "negative": "neg", "netral": "neu", "neutral": "neu", "positif": "pos", "positive": "pos"}


def baca_llm(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        d = pd.read_excel(p, sheet_name=0, dtype=str)
    else:  # csv: toleran terhadap BOM dan pemisah ; (Excel berbahasa Indonesia)
        d = pd.read_csv(p, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    d.columns = [str(c).strip().lower() for c in d.columns]
    if not {"id", "label_llm"} <= set(d.columns):
        raise SystemExit(f"{p} harus punya kolom id dan label_llm")
    d["id"] = d["id"].astype(str).str.strip()
    d["label_llm"] = d["label_llm"].fillna("").str.strip().str.lower().replace(SINONIM)
    aneh = d[~d["label_llm"].isin(KODE + [""])]
    if len(aneh):
        daftar = ", ".join(f"{i}='{v}'" for i, v in zip(aneh["id"].head(15), aneh["label_llm"].head(15)))
        raise SystemExit(f"{len(aneh)} label_llm tidak dikenal di {p} (hanya neg / neu / pos / x): {daftar}")
    return d[["id", "label_llm"]].drop_duplicates("id")


def nilai_baseline(g, judul):
    yb, yg = g["label_bintang"].map(IDX), g["label_gold"].map(IDX)
    f_b, a_b = nilai(yb, g["pred"])
    f_g, a_g = nilai(yg, g["pred"])
    f_kelas = f1_score(yg, g["pred"], average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n== baseline TF-IDF + LR: {judul} ({len(g)} ulasan) ==")
    print(f"  dinilai dengan label BINTANG : macro-F1 {f_b:.3f} | accuracy {a_b:.3f}")
    print(f"  dinilai dengan label GOLD    : macro-F1 {f_g:.3f} | accuracy {a_g:.3f}")
    print("  F1 per kelas (label gold)    : " + " | ".join(f"{n} {v:.3f}" for n, v in zip(NAMA, f_kelas)))
    print("  confusion matrix vs gold (baris = gold, kolom = prediksi):")
    print(pd.DataFrame(confusion_matrix(yg, g["pred"], labels=[0, 1, 2]), index=NAMA, columns=NAMA).to_string())


def gabung(args):
    lembar = pd.read_excel(args.xlsx, sheet_name="label", dtype=str)
    lembar["label_manual"] = lembar["label_manual"].fillna("").str.strip().str.lower()
    lembar["catatan"] = lembar["catatan"].fillna("")
    aneh = lembar[~lembar["label_manual"].isin(KODE + [""])]
    if len(aneh):
        daftar = ", ".join(f"{i}='{v}'" for i, v in zip(aneh["id"].head(15), aneh["label_manual"].head(15)))
        raise SystemExit(f"{len(aneh)} isian tidak dikenal (hanya neg / neu / pos / x): {daftar}")

    llm = baca_llm(args.llm)
    lembar = lembar.merge(llm, on="id", how="left") if llm is not None else lembar.assign(label_llm="")
    lembar["label_llm"] = lembar["label_llm"].fillna("")

    # label gold: manual kalau ada; kalau tidak, label LLM. x dari manusia selalu menang (baris dibuang)
    pakai_manual = lembar["label_manual"].isin(NAMA)
    pakai_llm = (lembar["label_manual"] == "") & lembar["label_llm"].isin(NAMA)
    lembar["label_gold"] = ""
    lembar["sumber"] = ""
    lembar.loc[pakai_manual, "label_gold"] = lembar["label_manual"]
    lembar.loc[pakai_manual, "sumber"] = "manual"
    lembar.loc[pakai_llm, "label_gold"] = lembar["label_llm"]
    lembar.loc[pakai_llm, "sumber"] = "llm"
    n_x = int(((lembar["label_manual"] == "x") | ((lembar["label_manual"] == "") & (lembar["label_llm"] == "x"))).sum())
    dipakai = lembar[lembar["label_gold"] != ""]
    n_kosong = len(lembar) - len(dipakai) - n_x
    if dipakai.empty:
        raise SystemExit("Belum ada baris yang berlabel neg / neu / pos.")
    print(f"baris di lembar {len(lembar)} | dipakai {len(dipakai)} (manual {int(pakai_manual.sum())}, "
          f"llm {int(pakai_llm.sum())}) | x (dibuang) {n_x} | belum berlabel {n_kosong}")

    if llm is not None:
        irisan = lembar[pakai_manual & lembar["label_llm"].isin(NAMA)]
        if len(irisan):
            kp = cohen_kappa_score(irisan["label_manual"], irisan["label_llm"], labels=NAMA)
            sama = (irisan["label_manual"] == irisan["label_llm"]).mean()
            print(f"\n== kesepakatan manual vs LLM pada {len(irisan)} ulasan yang dilabeli keduanya ==")
            print(pd.crosstab(irisan["label_manual"], irisan["label_llm"]).reindex(index=NAMA, columns=NAMA, fill_value=0).to_string())
            print(f"sepakat {sama:.1%} | Cohen's kappa {kp:.3f}   (baris = manual, kolom = LLM)")
            selisih = irisan[irisan["label_manual"] != irisan["label_llm"]][["id", "teks", "label_manual", "label_llm", "catatan"]]
            selisih.to_csv(args.selisih, index=False)
            print(f"{len(selisih)} ulasan berbeda label disimpan di {args.selisih} (bahan untuk melihat pola perbedaan)")
        else:
            print("\n(belum ada ulasan yang dilabeli manual DAN LLM sekaligus, kesepakatan belum bisa dihitung)")

    kunci = pd.read_csv(args.kunci, dtype={"reviewId": str})
    ds = pd.read_csv(args.dataset, dtype={"reviewId": str}).fillna({"text": "", "text_klasik": ""})
    gold = (dipakai.merge(kunci, on="id", validate="1:1")
            .merge(ds[["reviewId", "text", "text_klasik", "n_kata"]], on="reviewId", validate="m:1"))
    gold = gold.rename(columns={"teks": "content"})
    kolom = ["id", "reviewId", "app_id", "score", "label_bintang", "label_gold", "sumber", "label_manual",
             "label_llm", "content", "text", "text_klasik", "n_kata", "catatan"]
    out = Path(args.gold)
    gold[kolom].to_csv(out, index=False)

    print("\n== distribusi label gold ==")
    dist = gold["label_gold"].value_counts().reindex(NAMA, fill_value=0)
    print(pd.DataFrame({"jumlah": dist, "persen": (dist / dist.sum()).round(3)}).to_string())

    print("\n== label bintang (baris) x label gold (kolom), jumlah ==")
    print(pd.crosstab(gold["label_bintang"], gold["label_gold"]).reindex(index=NAMA, columns=NAMA, fill_value=0).to_string())
    print("\n== bintang 1-5 (baris) x label gold: persen per baris ==")
    print(pd.crosstab(gold["score"], gold["label_gold"], normalize="index").reindex(columns=NAMA, fill_value=0).round(2).to_string())
    cocok = (gold["label_bintang"] == gold["label_gold"]).mean()
    kappa = cohen_kappa_score(gold["label_bintang"], gold["label_gold"], labels=NAMA)
    print(f"\nlabel bintang cocok dengan label gold: {cocok:.1%} | Cohen's kappa {kappa:.3f}")

    pred_file = Path(args.pred)
    if pred_file.exists():
        pr = pd.read_csv(pred_file, dtype={"reviewId": str})[["reviewId", "pred"]]
        g = gold.merge(pr, on="reviewId", how="inner")
        nilai_baseline(g, "semua gold")
        manual = g[g["sumber"] == "manual"]
        if llm is not None and 0 < len(manual) < len(g):
            nilai_baseline(manual, "hanya yang berlabel manual")
    else:
        print(f"\n(file prediksi baseline tidak ditemukan: {pred_file})")
    if n_kosong:
        print(f"\nCatatan: masih {n_kosong} baris belum berlabel; angka di atas akan berubah saat sisanya selesai.")
    print(f"\nTersimpan: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gabung", action="store_true", help="periksa isian dan hitung statistik")
    ap.add_argument("--ekspor-teks", action="store_true", help="buat xlsx id+teks tanpa label (untuk dilabeli Claude)")
    ap.add_argument("--n", type=int, default=600, help="jumlah ulasan di lembar (default 600)")
    ap.add_argument("--timpa", action="store_true", help="izinkan membuat ulang lembar yang sudah ada (xlsx manual atau xlsx LLM)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=str(PROCESSED / "dataset.csv"))
    ap.add_argument("--xlsx", default=str(LABELED / "gold_untuk_dilabeli.xlsx"))
    ap.add_argument("--kunci", default=str(LABELED / "gold_kunci.csv"))
    ap.add_argument("--gold", default=str(LABELED / "gold.csv"))
    ap.add_argument("--llm", default=str(LABELED / "gold_label_llm.xlsx"), help="xlsx atau csv berisi id dan label_llm")
    ap.add_argument("--selisih", default=str(LABELED / "gold_selisih.csv"))
    ap.add_argument("--pred", default=str(PROCESSED / "pred_baseline_test.csv"))
    args = ap.parse_args()
    if args.ekspor_teks:
        ekspor_teks(args)
    elif args.gabung:
        gabung(args)
    else:
        buat(args)


if __name__ == "__main__":
    main()
