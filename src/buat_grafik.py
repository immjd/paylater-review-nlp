"""Grafik keluhan per aplikasi untuk README, dari reports/kategori_keluhan.csv (hanya angka agregat, aman di-commit).

Keluaran (reports/figures/):
  keluhan_per_aplikasi.png  : 3 panel (satu per aplikasi), kategori diurutkan dari yang paling sering DI aplikasi itu.
                              Garis = selang kepercayaan 95% (Wilson). Batang berarsir = kategori dengan presisi
                              hasil validasi manual < --ambang (angkanya kemungkinan terlalu tinggi).
  akses_kredit.png          : keluhan inti "akses kredit" dan "bayar lancar tapi ditolak/limit turun/tidak bisa pinjam lagi".

Penggunaan:  python src/buat_grafik.py
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import ROOT

APP = ["Kredivo", "Akulaku", "HomeCredit"]
PENDEK = {
    "ditolak": "Pengajuan ditolak", "limit_turun": "Limit turun / kecil / tidak naik", "tidak_bisa_pinjam_lagi": "Tidak bisa pinjam / pakai limit",
    "penagihan": "Penagihan kasar / debt collector", "spam_telepon": "Spam telepon marketing", "login_akun": "Login / OTP / akun diblokir",
    "captcha": "CAPTCHA", "app_error": "Aplikasi error / lemot", "biaya": "Denda / bunga / biaya", "pencairan_pembayaran": "Dana / pembayaran tidak masuk",
    "data_privasi": "Data pribadi / privasi", "tagihan_asing": "Tagihan padahal tidak pinjam", "skor_kredit": "Skor kredit / SLIK",
    "customer_service": "Customer service", "belanja_refund": "Belanja / pulsa / refund", "lucky_draw": "Lucky draw / undian", "penipuan": "Penipuan / bohong",
    "akses_kredit": "Akses kredit bermasalah\n(ditolak + limit turun + tidak bisa pinjam)",
    "lancar_tapi_ditutup": "Bayar lancar, tapi akses kredit\nditutup / limit turun",
}
WARNA, WARNA_PUDAR = "#2a78d6", "#a9c8f0"        # satu rona (palet referensi dataviz); identitas dibawa label sumbu, bukan warna
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def presisi_terakhir(folder):
    """Presisi per kategori dari putaran validasi TERBARU yang menilai kategori itu."""
    hasil = {}
    for f in sorted(glob.glob(str(folder / "validasi_kategori*.csv"))):   # urutan nama = urutan putaran
        v = pd.read_csv(f)
        v = v[v["ukuran"] == "presisi"]
        for _, r in v.iterrows():
            hasil[r["kategori"]] = float(r["nilai"])
    return hasil


def riwayat(folder, kunci):
    """'putaran 1: 12/12, putaran 2: 10/10' dari berkas validasi (kosong bila kategori itu tidak pernah dinilai)."""
    bagian = []
    for i, f in enumerate(sorted(glob.glob(str(folder / "validasi_kategori*.csv"))), 1):
        v = pd.read_csv(f)
        v = v[(v["ukuran"] == "presisi") & (v["kategori"] == kunci)]
        if len(v):
            bagian.append(f"putaran {i}: {int(v['n_benar'].iloc[0])}/{int(v['n_dinilai'].iloc[0])}")
    return ", ".join(bagian) or "belum divalidasi"


def gaya(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def batang(ax, d, a, rendah, label=lambda k: PENDEK.get(k, k)):
    y = range(len(d))
    for i, (k, r) in enumerate(d.iterrows()):
        pudar = k in rendah
        ax.barh(i, r[f"pct_{a}"], height=0.62, color=WARNA_PUDAR if pudar else WARNA, hatch="////" if pudar else None,
                edgecolor="white" if not pudar else WARNA, linewidth=0.6)
        ax.plot([r[f"lo_{a}"], r[f"hi_{a}"]], [i, i], color=INK, linewidth=1.1, solid_capstyle="butt")
        ax.text(r[f"hi_{a}"] + 0.4, i, f"{r[f'pct_{a}']:.1f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([label(k) + (" *" if k in rendah else "") for k in d.index])
    ax.invert_yaxis()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "reports" / "kategori_keluhan.csv"))
    ap.add_argument("--validasi-dir", default=str(ROOT / "reports"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures"))
    ap.add_argument("--top", type=int, default=8, help="jumlah kategori per aplikasi")
    ap.add_argument("--ambang", type=float, default=0.8, help="presisi di bawah ini ditandai berarsir (*)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    k = pd.read_csv(args.csv).set_index("kunci")
    pres = presisi_terakhir(Path(args.validasi_dir))
    rendah = {c for c, v in pres.items() if v < args.ambang}
    dasar = k[k["jenis"] == "dasar"]

    # 1. tiga panel: peringkat kategori di tiap aplikasi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharex=True)
    xmax = 0
    for ax, a in zip(axes, APP):
        d = dasar.sort_values(f"pct_{a}", ascending=False).head(args.top)
        batang(ax, d, a, rendah)
        gaya(ax)
        n = int(k.loc["spesifik", f"n_{a}"] + k.loc["hanya_umum", f"n_{a}"] + k.loc["tanpa_kategori", f"n_{a}"])
        ax.set_title(f"{a} (n = {n:,})".replace(",", "."), loc="left", fontsize=10.5, color=INK, fontweight="bold")
        xmax = max(xmax, d[f"hi_{a}"].max())
    for ax in axes:
        ax.set_xlim(0, xmax * 1.22)
        ax.set_xlabel("% ulasan negatif", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.005,
             "n = ulasan negatif yang dianalisis (minimal 5 kata). Garis = selang 95% (Wilson). Satu ulasan boleh masuk beberapa kategori, jadi persen tidak dijumlahkan.\n"
             "Pola aturan menghitung batas bawah: sebagian ulasan negatif memuat keluhan spesifik yang belum tertangkap (lihat README).\n"
             f"* berarsir = presisi validasi manual di bawah {int(args.ambang*100)}%, angka kemungkinan terlalu tinggi. Periode ulasan tiap aplikasi berbeda: bandingkan persen, bukan jumlah.",
             fontsize=7.6, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "keluhan_per_aplikasi.png", dpi=200, facecolor="white")
    plt.close(fig)

    # 2. keluhan inti: akses kredit, per aplikasi
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), sharex=True)
    xmax = 0
    for ax, key in zip(axes, ["akses_kredit", "lancar_tapi_ditutup"]):
        r = k.loc[key]
        for i, a in enumerate(APP):
            ax.barh(i, r[f"pct_{a}"], height=0.55, color=WARNA, edgecolor="white")
            ax.plot([r[f"lo_{a}"], r[f"hi_{a}"]], [i, i], color=INK, linewidth=1.1, solid_capstyle="butt")
            ax.text(r[f"hi_{a}"] + 0.8, i, f"{r[f'pct_{a}']:.1f}%", va="center", fontsize=9, color=INK)
            xmax = max(xmax, r[f"hi_{a}"])
        ax.set_yticks(range(3)); ax.set_yticklabels(APP); ax.invert_yaxis(); gaya(ax)
        ax.set_title(PENDEK[key], loc="left", fontsize=10, color=INK, fontweight="bold")
    for ax in axes:
        ax.set_xlim(0, xmax * 1.2); ax.set_xlabel("% ulasan negatif", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.005, f"Garis = selang 95% (Wilson). Ulasan sampel yang benar (validasi manual) - akses kredit: {riwayat(Path(args.validasi_dir), 'akses_kredit')}; "
             f"lancar-tapi-ditutup: {riwayat(Path(args.validasi_dir), 'lancar_tapi_ditutup')}.",
             fontsize=7.6, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "akses_kredit.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Tersimpan di {out}: keluhan_per_aplikasi.png, akses_kredit.png")
    if rendah:
        print("Kategori berarsir (presisi < %d%%): %s" % (args.ambang * 100, ", ".join(sorted(rendah))))


if __name__ == "__main__":
    main()
