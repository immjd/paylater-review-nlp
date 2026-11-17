"""Baseline sentimen: TF-IDF + Logistic Regression (3 kelas: neg / neu / pos).

Masuk : data/processed/dataset.csv   (dari make_dataset.py)
Keluar: models/baseline_tfidf_lr.joblib            model terbaik (vectorizer + classifier)
        reports/baseline_metrics.json              semua angka penting
        reports/baseline_fitur_teratas.csv         kata paling berpengaruh per kelas
        reports/figures/baseline_confusion_matrix.png
        data/processed/pred_baseline_test.csv      prediksi per ulasan test (bahan analisis error)

Alur
  1. Beberapa konfigurasi TF-IDF x beberapa nilai C dicoba. Pemilihan HANYA memakai macro-F1 di set VAL.
  2. Konfigurasi terbaik dilatih ulang di TRAIN saja, lalu dinilai di TEST satu kali.
  3. Dua pembanding murah dicetak supaya angka baseline punya konteks:
       - tebakan acak (DummyClassifier)
       - model yang hanya melihat panjang teks (kata-kata tidak dipakai sama sekali)

Label di sini masih label LEMAH dari bintang (1-2 neg, 3 neu, 4-5 pos), jadi angka test mengukur
kecocokan dengan bintang, bukan dengan makna teks. Angka dengan label manual (gold set) menyusul.

Jalankan dari akar proyek:  python src\\train_baseline.py
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from paths import FIGURES, MODELS, PROCESSED  # noqa: E402

NAMA = ["neg", "neu", "pos"]
TAMPIL = ["negatif\n(1-2 bintang)", "netral\n(3 bintang)", "positif\n(4-5 bintang)"]
KONFIG = {
    "kata_klasik_1": dict(kolom="text_klasik", vec=dict(analyzer="word", ngram_range=(1, 1))),
    "kata_klasik_12": dict(kolom="text_klasik", vec=dict(analyzer="word", ngram_range=(1, 2))),
    "kata_text_12": dict(kolom="text", vec=dict(analyzer="word", ngram_range=(1, 2))),
    "char_text_25": dict(kolom="text", vec=dict(analyzer="char_wb", ngram_range=(2, 5))),
}
C_GRID = [0.1, 0.3, 1, 3, 10]

# warna grafik: ramp biru (satu hue, terang -> gelap) dan tinta netral
RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#104281"]
PERMUKAAN, INK, INK2, REDUP = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def buat_model(cfg, C, seed):
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, **cfg["vec"])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    return Pipeline([("tfidf", vec), ("lr", clf)])


def metrik(y, pred):
    f1_kelas = f1_score(y, pred, average=None, labels=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "f1_per_kelas": {n: float(v) for n, v in zip(NAMA, f1_kelas)},
    }


def gambar_cm(cm, judul, subjudul, path):
    persen = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=160, facecolor=PERMUKAAN)
    ax.set_facecolor(PERMUKAAN)
    ax.imshow(persen, cmap=LinearSegmentedColormap.from_list("biru", RAMP), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            warna = "#ffffff" if persen[i, j] > 0.5 else INK
            ax.text(j, i, f"{persen[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", color=warna, fontsize=11)
    ax.set_xticks(range(3), TAMPIL, fontsize=9)
    ax.set_yticks(range(3), TAMPIL, fontsize=9)
    ax.set_xlabel("prediksi model", color=INK2, fontsize=10)
    ax.set_ylabel("label sebenarnya (dari bintang)", color=INK2, fontsize=10)
    ax.tick_params(length=0, colors=INK2)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color=PERMUKAAN, linewidth=3)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.04, 0.955, judul, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.04, 0.905, subjudul, ha="left", va="top", fontsize=9, color=INK2)
    fig.text(0.04, 0.025, "angka: persen per baris (recall) dan jumlah ulasan", ha="left", fontsize=8, color=REDUP)
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.2, right=0.96)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=PERMUKAAN)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default=str(PROCESSED / "dataset.csv"))
    ap.add_argument("--models", default=str(MODELS), help="folder model")
    ap.add_argument("--reports", default=str(FIGURES.parent), help="folder laporan (json, csv)")
    ap.add_argument("--figures", default=str(FIGURES), help="folder gambar")
    ap.add_argument("--pred", default=str(PROCESSED), help="folder prediksi test")
    ap.add_argument("--top", type=int, default=15, help="jumlah fitur teratas per kelas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype={"reviewId": str}).fillna({"text": "", "text_klasik": ""})
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    assert min(len(tr), len(va), len(te)) > 0, "salah satu split kosong"
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    # 1. pilih konfigurasi dengan macro-F1 di VAL (test belum disentuh)
    baris = []
    print("\nmencoba konfigurasi (macro-F1 di val) ...")
    for nama, cfg in KONFIG.items():
        for C in C_GRID:
            m = buat_model(cfg, C, args.seed).fit(tr[cfg["kolom"]], tr["label"])
            f1 = f1_score(va["label"], m.predict(va[cfg["kolom"]]), average="macro")
            baris.append((nama, C, f1))
        print(f"  {nama:<15} selesai")
    grid = pd.DataFrame(baris, columns=["konfigurasi", "C", "macro_f1_val"])
    print("\n== macro-F1 val per konfigurasi x C ==")
    print(grid.pivot(index="konfigurasi", columns="C", values="macro_f1_val").round(3).to_string())
    terbaik = grid.sort_values("macro_f1_val", ascending=False, kind="stable").iloc[0]
    nama_b, C_b = terbaik["konfigurasi"], float(terbaik["C"])
    cfg_b = KONFIG[nama_b]
    print(f"\nterbaik: {nama_b}, C={C_b} (macro-F1 val {terbaik['macro_f1_val']:.3f})")

    # 2. latih ulang di TRAIN saja, nilai di TEST satu kali
    model = buat_model(cfg_b, C_b, args.seed).fit(tr[cfg_b["kolom"]], tr["label"])
    m_val = metrik(va["label"], model.predict(va[cfg_b["kolom"]]))
    proba = model.predict_proba(te[cfg_b["kolom"]])
    pred = proba.argmax(axis=1)
    m_test = metrik(te["label"], pred)

    # 3. pembanding
    dummy = DummyClassifier(strategy="stratified", random_state=args.seed).fit(tr[["n_kata"]], tr["label"])
    m_dummy = metrik(te["label"], dummy.predict(te[["n_kata"]]))
    panjang = LogisticRegression(class_weight="balanced", max_iter=1000)
    panjang.fit(np.log1p(tr[["n_kata"]]), tr["label"])
    m_panjang = metrik(te["label"], panjang.predict(np.log1p(te[["n_kata"]])))

    print("\n== ringkasan (macro-F1 / accuracy di TEST) ==")
    print(f"  tebakan acak            : {m_dummy['macro_f1']:.3f} / {m_dummy['accuracy']:.3f}")
    print(f"  hanya panjang teks      : {m_panjang['macro_f1']:.3f} / {m_panjang['accuracy']:.3f}")
    print(f"  TF-IDF + LR ({nama_b}, C={C_b:g}): {m_test['macro_f1']:.3f} / {m_test['accuracy']:.3f}"
          f"   (val: {m_val['macro_f1']:.3f} / {m_val['accuracy']:.3f})")
    print("\n== laporan per kelas (test) ==")
    print(classification_report(te["label"], pred, target_names=NAMA, digits=3))

    cm = confusion_matrix(te["label"], pred, labels=[0, 1, 2])
    print("confusion matrix (baris = sebenarnya, kolom = prediksi; urutan neg, neu, pos):")
    print(pd.DataFrame(cm, index=NAMA, columns=NAMA).to_string())

    te = te.assign(pred=pred, benar=(pred == te["label"]))
    per_app = {a: float(f1_score(g["label"], g["pred"], average="macro")) for a, g in te.groupby("app_id")}
    per_bintang = te.groupby("score")["benar"].mean()
    print("\n== macro-F1 test per aplikasi ==")
    for a, v in per_app.items():
        print(f"  {a:<24} {v:.3f}")
    print("== ketepatan test per bintang (sentimen yang diprediksi sama dengan label bintang) ==")
    for s, v in per_bintang.items():
        print(f"  bintang {s}: {v:.3f}")

    # fitur teratas per kelas: selalu dari model berbasis KATA terbaik (potongan huruf dari model
    # char n-gram tidak terbaca manusia). Model ini hanya untuk interpretasi, tidak dinilai di test.
    grid_kata = grid[grid["konfigurasi"].map(lambda n: KONFIG[n]["vec"]["analyzer"] == "word")]
    kb = grid_kata.sort_values("macro_f1_val", ascending=False, kind="stable").iloc[0]
    nama_k, C_k = kb["konfigurasi"], float(kb["C"])
    if (nama_k, C_k) == (nama_b, C_b):
        model_kata = model
    else:
        model_kata = buat_model(KONFIG[nama_k], C_k, args.seed).fit(tr[KONFIG[nama_k]["kolom"]], tr["label"])
    fitur = np.array(model_kata.named_steps["tfidf"].get_feature_names_out())
    koef = model_kata.named_steps["lr"].coef_
    teratas = []
    for k, nama in enumerate(NAMA):
        for rank, i in enumerate(np.argsort(koef[k])[::-1][: args.top], start=1):
            teratas.append((nama, rank, fitur[i], float(koef[k][i])))
    teratas = pd.DataFrame(teratas, columns=["kelas", "peringkat", "fitur", "koefisien"])
    print(f"\n== {args.top} fitur teratas per kelas (model kata: {nama_k}, C={C_k:g}) ==")
    for nama in NAMA:
        print(f"  {nama}: " + ", ".join(teratas[teratas["kelas"] == nama]["fitur"].tolist()))

    # simpan
    models, reports, pred_dir = Path(args.models), Path(args.reports), Path(args.pred)
    for d in (models, reports, pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models / "baseline_tfidf_lr.joblib")
    teratas.to_csv(reports / "baseline_fitur_teratas.csv", index=False)
    keluar = te[["reviewId", "app_id", "score", "sentiment", "label", "pred", "content", "text"]].copy()
    keluar["pred_sentiment"] = keluar["pred"].map(dict(enumerate(NAMA)))
    for k, nama in enumerate(NAMA):
        keluar[f"p_{nama}"] = proba[:, k].round(4)
    keluar.to_csv(pred_dir / "pred_baseline_test.csv", index=False)
    ringkas = {
        "konfigurasi": nama_b, "kolom_teks": cfg_b["kolom"], "vectorizer": {k: str(v) for k, v in cfg_b["vec"].items()},
        "C": C_b, "model_fitur_teratas": {"konfigurasi": nama_k, "C": C_k}, "seed": args.seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "val": m_val, "test": m_test, "pembanding_test": {"acak": m_dummy, "hanya_panjang_teks": m_panjang},
        "test_per_aplikasi_macro_f1": per_app,
        "test_per_bintang_ketepatan": {int(s): float(v) for s, v in per_bintang.items()},
        "confusion_matrix_test": cm.tolist(),
        "grid_val": grid.round(5).to_dict(orient="records"),
    }
    (reports / "baseline_metrics.json").write_text(json.dumps(ringkas, indent=2, ensure_ascii=False), encoding="utf-8")
    gambar = Path(args.figures) / "baseline_confusion_matrix.png"
    gambar_cm(cm, "Baseline TF-IDF + Logistic Regression",
              f"set test, n={len(te)} | macro-F1 {m_test['macro_f1']:.3f} | accuracy {m_test['accuracy']:.3f}", gambar)
    print(f"\nTersimpan: {models / 'baseline_tfidf_lr.joblib'}\n           {reports / 'baseline_metrics.json'}"
          f"\n           {gambar}\n           {pred_dir / 'pred_baseline_test.csv'}")
    print("\nCatatan: label test masih dari bintang (label lemah). Angka dengan label manual menyusul.")


if __name__ == "__main__":
    main()
