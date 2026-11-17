"""Bandingkan baseline TF-IDF + LR dengan IndoBERT (beberapa seed) pada test (label bintang) dan gold set.

Skrip ini hanya MEMBACA prediksi yang sudah ada, tidak melatih apa pun.

Masuk (default)
  data/processed/dataset.csv                      split test, n_kata, app_id
  data/processed/pred_baseline_test.csv           dari train_baseline.py
  data/processed/hasil_indobert/pred_indobert_s*_test.csv   dari notebooks/02_finetune_indobert.ipynb
  data/labeled/gold.csv                           dari make_gold.py --gabung  (opsional)
Keluar
  reports/perbandingan_model.txt   (laporan yang sama dengan yang tercetak, UTF-8)
  reports/perbandingan_model.json  (angka mentah, untuk README / grafik)

Isi laporan, untuk test (label bintang) DAN gold (label dari teks):
  1. macro-F1 dan accuracy: baseline, tiap seed, rata-rata seed, ensemble seed
  2. F1 per kelas
  3. Selisih IndoBERT - baseline dengan selang kepercayaan 95% (bootstrap berpasangan: ulasan yang sama
     dipakai ulang untuk semua model, jadi selisih tidak ikut tergoyang oleh sampel yang mudah/sulit)
  4. Accuracy menurut panjang ulasan (menguji dugaan: IndoBERT paling unggul di ulasan panjang)
  5. macro-F1 per aplikasi
  6. Confusion matrix di gold

Selang kepercayaan hanya mencerminkan pengambilan sampel ulasan. Variasi antar-seed dilaporkan terpisah
(rata-rata +/- simpangan), dan label gold berasal dari satu pelabel.

Jalankan dari akar proyek:  python src\\bandingkan_model.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from paths import LABELED, PROCESSED, ROOT

NAMA = ["neg", "neu", "pos"]
IDX = {n: i for i, n in enumerate(NAMA)}
APP = {"com.finaccel.android": "Kredivo", "id.co.myhomecredit": "HomeCredit", "io.silvrr.installment": "Akulaku"}
BIN = [-1, 3, 7, 15, np.inf]
BIN_NAMA = ["1-3 kata", "4-7 kata", "8-15 kata", ">15 kata"]

OUT = []


def cetak(teks=""):
    print(teks)
    OUT.append(str(teks))


# ---------------------------------------------------------------- metrik + bootstrap
def _cm(yy, pp):
    """Confusion matrix per resample. yy, pp: (B, n) bilangan bulat 0..2 -> (B, 3, 3), baris = benar."""
    b = yy.shape[0]
    kode = yy * 3 + pp + (np.arange(b) * 9)[:, None]
    return np.bincount(kode.ravel(), minlength=b * 9).reshape(b, 3, 3)


def _macro(c):
    tp = np.einsum("bii->bi", c).astype(float)
    pembagi = c.sum(2) + c.sum(1)
    f1 = np.divide(2 * tp, pembagi, out=np.zeros_like(tp), where=pembagi > 0)   # F1 = 0 bila kelas tak ada (seperti zero_division=0)
    return f1.mean(1)


def bootstrap(y, preds, nboot, rng, chunk=500):
    """Resample ulasan dengan pengembalian; indeks yang SAMA dipakai untuk semua model (berpasangan)."""
    n = len(y)
    hasil = {k: {"macro": np.empty(nboot), "acc": np.empty(nboot)} for k in preds}
    for a in range(0, nboot, chunk):
        b = min(chunk, nboot - a)
        idx = rng.integers(0, n, size=(b, n))
        yy = y[idx]
        for k, p in preds.items():
            c = _cm(yy, p[idx])
            hasil[k]["macro"][a:a + b] = _macro(c)
            hasil[k]["acc"][a:a + b] = np.einsum("bii->b", c) / n
    return hasil


def f1_total(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)


def f1_kelas(y, p):
    return f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)


def ms(x, d=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{d}f} +/- {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def selang(diff_boot):
    lo, hi = np.percentile(diff_boot, [2.5, 97.5])
    return float(lo), float(hi), float((diff_boot > 0).mean())


def putusan(lo, hi):
    if lo > 0:
        return "IndoBERT lebih tinggi; selang 95% tidak melewati 0"
    if hi < 0:
        return "baseline lebih tinggi; selang 95% tidak melewati 0"
    return "belum bisa dibedakan dari derau pada ukuran sampel ini (selang 95% melewati 0)"


# ---------------------------------------------------------------- memuat data
def muat(args):
    ds = pd.read_csv(args.dataset, dtype={"reviewId": str})
    te = ds[ds["split"] == "test"][["reviewId", "app_id", "score", "label", "n_kata"]].reset_index(drop=True)
    if te.empty:
        raise SystemExit(f"tidak ada baris split == 'test' di {args.dataset}")

    base = pd.read_csv(args.pred_baseline, dtype={"reviewId": str})
    base = base.drop_duplicates("reviewId").set_index("reviewId")
    hilang = set(te["reviewId"]) - set(base.index)
    if hilang:
        raise SystemExit(f"{len(hilang)} ulasan test tidak ada di {args.pred_baseline}")
    base = base.reindex(te["reviewId"])
    if not (base["label"].values == te["label"].values).all():
        raise SystemExit("label di pred_baseline_test.csv tidak sama dengan dataset.csv (versi dataset berbeda?)")
    p_base = base["pred"].astype(int).values

    berkas = {}
    for f in Path(args.pred_dir).glob("pred_indobert_s*_test.csv"):
        m = re.search(r"_s(\d+)_test", f.name)
        if m:
            berkas[int(m.group(1))] = f
    if not berkas:
        raise SystemExit(f"tidak ada pred_indobert_s*_test.csv di {args.pred_dir}")
    seeds = sorted(berkas)
    p_seed, proba = {}, {}
    for s in seeds:
        d = pd.read_csv(berkas[s], dtype={"reviewId": str}).drop_duplicates("reviewId").set_index("reviewId")
        if set(te["reviewId"]) - set(d.index):
            raise SystemExit(f"{berkas[s].name} tidak mencakup semua ulasan test")
        d = d.reindex(te["reviewId"])
        if not (d["label"].values == te["label"].values).all():
            raise SystemExit(f"label di {berkas[s].name} tidak sama dengan dataset.csv")
        p_seed[s] = d["pred"].astype(int).values
        proba[s] = d[["p_neg", "p_neu", "p_pos"]].values
    p_ens = np.mean([proba[s] for s in seeds], axis=0).argmax(1)

    n_kata = te["n_kata"].values
    app = te["app_id"].values
    y_test = te["label"].astype(int).values
    semua = {
        "nama": "TEST (label bintang)", "kunci": "test", "y": y_test, "base": p_base, "seeds": p_seed,
        "ens": p_ens, "n_kata": n_kata, "app": app,
    }
    himpunan = [semua]

    gp = Path(args.gold)
    if gp.exists():
        g = pd.read_csv(gp, dtype={"reviewId": str})
        g = g[g["label_gold"].isin(NAMA)].reset_index(drop=True)
        posisi = pd.Series(np.arange(len(te)), index=te["reviewId"])
        tidak = set(g["reviewId"]) - set(posisi.index)
        if tidak:
            raise SystemExit(f"{len(tidak)} ulasan gold tidak ada di split test (gold harus bagian dari test)")
        pos = posisi.loc[g["reviewId"]].values
        himpunan.append({
            "nama": "GOLD (label dari teks)", "kunci": "gold", "y": g["label_gold"].map(IDX).astype(int).values,
            "base": p_base[pos], "seeds": {s: p_seed[s][pos] for s in seeds}, "ens": p_ens[pos],
            "n_kata": n_kata[pos], "app": app[pos],
        })
    else:
        cetak(f"(gold set tidak ditemukan di {gp}: bagian gold dilewati)")

    # cek silang dengan angka yang dicetak notebook
    rj = Path(args.pred_dir) / "ringkasan.json"
    if rj.exists():
        try:
            nb = {int(r["seed"]): r["f1_test_bintang"] for r in json.loads(rj.read_text(encoding="utf-8"))["hasil"]}
            beda = [abs(f1_total(y_test, p_seed[s]) - nb[s]) for s in seeds if s in nb]
            if beda and max(beda) < 5e-4:
                cetak(f"Cek silang dengan notebook: macro-F1 test per seed COCOK (selisih maks {max(beda):.5f}).")
            elif beda:
                cetak(f"PERINGATAN: macro-F1 test per seed BEDA dari notebook (selisih maks {max(beda):.4f}). "
                      "Periksa apakah dataset.csv dan hasil_indobert berasal dari run yang sama.")
        except (KeyError, ValueError, TypeError):
            cetak("(ringkasan.json ada tapi formatnya tidak dikenali; cek silang dilewati)")
    return himpunan, seeds


# ---------------------------------------------------------------- laporan
def laporan(h, seeds, args, rng, hasil_json):
    y, base, ps, ens = h["y"], h["base"], h["seeds"], h["ens"]
    n = len(y)
    kj = {"n": int(n)}
    cetak("\n" + "=" * 78)
    cetak(f"{h['nama']}  |  n = {n}")
    cetak("=" * 78)

    # 1. ringkasan
    f_b, a_b = f1_total(y, base), accuracy_score(y, base)
    f_s = {s: f1_total(y, ps[s]) for s in seeds}
    a_s = {s: accuracy_score(y, ps[s]) for s in seeds}
    f_e, a_e = f1_total(y, ens), accuracy_score(y, ens)
    cetak("\n[1] macro-F1 dan accuracy")
    cetak(f"  {'model':<30}{'macro-F1':>18}{'accuracy':>18}")
    cetak(f"  {'baseline TF-IDF + LR':<30}{f_b:>18.3f}{a_b:>18.3f}")
    for s in seeds:
        cetak(f"  {'IndoBERT seed ' + str(s):<30}{f_s[s]:>18.3f}{a_s[s]:>18.3f}")
    cetak(f"  {'IndoBERT rata-rata ' + str(len(seeds)) + ' seed':<30}{ms(list(f_s.values())):>18}{ms(list(a_s.values())):>18}")
    cetak(f"  {'IndoBERT ensemble (rata2 proba)':<30}{f_e:>18.3f}{a_e:>18.3f}")
    kj["macro_f1"] = {"baseline": f_b, "indobert_seed": {str(s): f_s[s] for s in seeds},
                      "indobert_mean": float(np.mean(list(f_s.values()))),
                      "indobert_std": float(np.std(list(f_s.values()), ddof=1)) if len(seeds) > 1 else None,
                      "ensemble": f_e}
    kj["accuracy"] = {"baseline": a_b, "indobert_mean": float(np.mean(list(a_s.values()))), "ensemble": a_e}

    # 2. per kelas
    k_b = f1_kelas(y, base)
    k_s = np.mean([f1_kelas(y, ps[s]) for s in seeds], axis=0)
    k_e = f1_kelas(y, ens)
    cetak("\n[2] F1 per kelas")
    cetak(f"  {'model':<30}" + "".join(f"{c:>10}" for c in NAMA) + f"{'(n per kelas)':>26}")
    cetak(f"  {'baseline TF-IDF + LR':<30}" + "".join(f"{v:>10.3f}" for v in k_b)
          + f"{'  ' + ' / '.join(str(int((y == i).sum())) for i in range(3)):>26}")
    cetak(f"  {'IndoBERT rata-rata seed':<30}" + "".join(f"{v:>10.3f}" for v in k_s))
    cetak(f"  {'IndoBERT ensemble':<30}" + "".join(f"{v:>10.3f}" for v in k_e))
    kj["f1_per_kelas"] = {"baseline": dict(zip(NAMA, map(float, k_b))), "indobert_mean": dict(zip(NAMA, map(float, k_s))),
                          "ensemble": dict(zip(NAMA, map(float, k_e)))}

    # 3. selisih berpasangan
    preds = {"base": base, "ens": ens, **{f"s{s}": ps[s] for s in seeds}}
    bt = bootstrap(y, preds, args.boot, rng)
    ib_macro = np.mean([bt[f"s{s}"]["macro"] for s in seeds], axis=0)
    ib_acc = np.mean([bt[f"s{s}"]["acc"] for s in seeds], axis=0)
    cetak(f"\n[3] Selisih IndoBERT - baseline (bootstrap berpasangan, {args.boot:,} resample)")
    kj["selisih"] = {}
    barisan = [
        ("macro-F1", "IndoBERT rata-rata seed", float(np.mean(list(f_s.values())) - f_b), ib_macro - bt["base"]["macro"], "macro_mean"),
        ("macro-F1", "IndoBERT ensemble", float(f_e - f_b), bt["ens"]["macro"] - bt["base"]["macro"], "macro_ens"),
        ("accuracy", "IndoBERT rata-rata seed", float(np.mean(list(a_s.values())) - a_b), ib_acc - bt["base"]["acc"], "acc_mean"),
    ]
    for metrik, nama, titik, boot, kunci in barisan:
        lo, hi, p_pos = selang(boot)
        cetak(f"  {metrik:<9}{nama:<26} {titik:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   P(selisih>0) = {p_pos:.3f}")
        cetak(f"{'':>11}-> {putusan(lo, hi)}")
        kj["selisih"][kunci] = {"titik": titik, "ci_lo": lo, "ci_hi": hi, "p_positif": p_pos}
    if len(seeds) > 1:
        cetak(f"  (variasi antar-seed sendiri: macro-F1 IndoBERT {ms(list(f_s.values()))}; "
              "selang di atas hanya mencerminkan pengambilan sampel ulasan)")

    # 4. per panjang
    kategori = pd.cut(pd.Series(h["n_kata"]), bins=BIN, labels=BIN_NAMA)
    cetak("\n[4] Accuracy menurut panjang ulasan (IndoBERT = rata-rata seed)")
    cetak(f"  {'panjang':<12}{'n':>6}{'%':>7}{'baseline':>10}{'IndoBERT':>10}{'selisih':>9}{'  95% CI selisih':>22}")
    kj["per_panjang"] = {}
    for nm in BIN_NAMA:
        m = (kategori == nm).values
        if m.sum() < 5:
            continue
        pb = {"base": base[m], **{f"s{s}": ps[s][m] for s in seeds}}
        b2 = bootstrap(y[m], pb, args.boot, rng)
        ab = accuracy_score(y[m], base[m])
        ai = float(np.mean([accuracy_score(y[m], ps[s][m]) for s in seeds]))
        d = np.mean([b2[f"s{s}"]["acc"] for s in seeds], axis=0) - b2["base"]["acc"]
        lo, hi, _ = selang(d)
        tanda = "  *" if (lo > 0 or hi < 0) else ""
        cetak(f"  {nm:<12}{int(m.sum()):>6}{100 * m.mean():>6.0f}%{ab:>10.3f}{ai:>10.3f}{ai - ab:>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{tanda}")
        kj["per_panjang"][nm] = {"n": int(m.sum()), "baseline": float(ab), "indobert": ai, "selisih": ai - ab,
                                 "ci_lo": lo, "ci_hi": hi}
    cetak("  (* = selang 95% tidak melewati 0. Tiap baris diuji sendiri-sendiri dengan sampel kecil, jadi")
    cetak("   anggap pola ini petunjuk, bukan bukti, kecuali selisihnya jelas dan konsisten antar baris.)")

    # 5. per aplikasi
    cetak("\n[5] macro-F1 per aplikasi (IndoBERT = rata-rata seed)")
    cetak(f"  {'aplikasi':<14}{'n':>6}{'baseline':>10}{'IndoBERT':>10}{'selisih':>9}")
    kj["per_app"] = {}
    for a in sorted(set(h["app"])):
        m = h["app"] == a
        fb = f1_total(y[m], base[m])
        fi = float(np.mean([f1_total(y[m], ps[s][m]) for s in seeds]))
        nama = APP.get(a, a)
        cetak(f"  {nama:<14}{int(m.sum()):>6}{fb:>10.3f}{fi:>10.3f}{fi - fb:>+9.3f}")
        kj["per_app"][nama] = {"n": int(m.sum()), "baseline": float(fb), "indobert": fi}

    # 6. confusion matrix (hanya gold)
    if h["kunci"] == "gold":
        cetak("\n[6] Confusion matrix terhadap label gold (baris = gold, kolom = prediksi)")
        for judul, p in (("baseline TF-IDF + LR", base), ("IndoBERT ensemble", ens)):
            cetak(f"  {judul}")
            cm = pd.DataFrame(confusion_matrix(y, p, labels=[0, 1, 2]), index=NAMA, columns=NAMA)
            cetak("    " + cm.to_string().replace("\n", "\n    "))
    hasil_json[h["kunci"]] = kj


def main():
    ap = argparse.ArgumentParser(description="Bandingkan baseline TF-IDF + LR dengan IndoBERT.")
    ap.add_argument("--dataset", default=str(PROCESSED / "dataset.csv"))
    ap.add_argument("--pred-baseline", default=str(PROCESSED / "pred_baseline_test.csv"))
    ap.add_argument("--pred-dir", default=str(PROCESSED / "hasil_indobert"))
    ap.add_argument("--gold", default=str(LABELED / "gold.csv"))
    ap.add_argument("--out-txt", default=str(ROOT / "reports" / "perbandingan_model.txt"))
    ap.add_argument("--out-json", default=str(ROOT / "reports" / "perbandingan_model.json"))
    ap.add_argument("--boot", type=int, default=10000, help="jumlah resample bootstrap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    himpunan, seeds = muat(args)
    cetak(f"IndoBERT: {len(seeds)} seed ({', '.join(map(str, seeds))}); bootstrap {args.boot:,} resample, seed acak {args.seed}.")
    rng = np.random.default_rng(args.seed)
    hasil_json = {"seeds": seeds, "boot": args.boot}
    for h in himpunan:
        laporan(h, seeds, args, rng, hasil_json)

    cetak("\nCatatan: label test = bintang (lemah, ~62% cocok dengan isi teks); label gold = satu pelabel manual (596 ulasan).")
    cetak("Evaluasi memakai teks yang unik (duplikat sudah dibuang), sehingga ulasan sangat pendek lebih jarang dari di data mentah.")

    for p in (args.out_txt, args.out_json):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(OUT) + "\n", encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(hasil_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTersimpan: {args.out_txt}\n           {args.out_json}")


if __name__ == "__main__":
    main()
