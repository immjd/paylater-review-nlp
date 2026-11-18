"""Demo: sentimen ulasan + keluhan yang terdeteksi.

Jalankan dari akar proyek (lokal, belum dipublikasikan ke mana pun):
    streamlit run app\\app.py

Model yang dipakai
  - Baseline TF-IDF + Logistic Regression : models/baseline_tfidf_lr.joblib      (dari src/train_baseline.py)
  - IndoBERT (opsional)                   : models/indobert_final/               (dari notebooks/02_finetune_indobert.ipynb)
                                            atau repo Hugging Face Hub lewat variabel INDOBERT_REPO
Kalau salah satu tidak ada, aplikasi tetap jalan dengan model yang tersedia.

Teks yang diketik tidak disimpan oleh aplikasi ini.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))

from kategori_keluhan import GABUNGAN, LABEL, SPESIFIK, kategorikan, pemicu  # noqa: E402
from paths import MODELS  # noqa: E402
from preprocess import bersihkan, buat_kunci, muat_sastrawi  # noqa: E402
from slang import EXTRA_STOP, FALLBACK_STOP, PROTECTED  # noqa: E402
from topic_model import teks_topik  # noqa: E402

DIR_MODEL = Path(os.environ.get("MODELS_DIR", MODELS))
NAMA = ["neg", "neu", "pos"]
TAMPIL = {"neg": "Negatif", "neu": "Netral", "pos": "Positif"}
SIMBOL = {"neg": "▼", "neu": "●", "pos": "▲"}          # makna tidak hanya lewat warna
WARNA = {"neg": "#e34948", "neu": "#8a8985", "pos": "#2a78d6"}   # pasangan divergen merah <-> biru, titik tengah abu-abu (aman untuk buta warna); dipakai untuk teks dan batang
MAX_LEN = 128

# Contoh karangan sendiri (bukan kutipan ulasan asli).
CONTOH = {
    "Sudah lunas, tidak bisa pinjam lagi": "Selalu bayar tepat waktu, sudah lunas semua, tapi sekarang tidak bisa pinjam lagi dan limit saya hilang.",
    "Telepon terus tiap hari": "Aplikasi bagus tapi saya ditelepon terus setiap hari menawarkan produk padahal sudah bilang tidak berminat.",
    "Aplikasi membantu": "Pengajuan cepat disetujui, limit naik, cicilan ringan. Sangat membantu untuk kebutuhan mendadak.",
    "Login susah": "Tidak bisa login, kode OTP tidak masuk dan verifikasi wajah gagal terus.",
}

CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 2.2rem;}
.batang {height: 10px; border-radius: 5px; background: rgba(128,128,128,.25); overflow: hidden; margin: 2px 0 10px 0;}
.batang > span {display: block; height: 100%; border-radius: 5px;}
.baris {display: flex; justify-content: space-between; font-size: 0.9rem;}
.label-besar {font-size: 2rem; font-weight: 600; line-height: 1.2; margin: 2px 0 2px 0;}
.kecil {font-size: 0.82rem; opacity: .75;}
</style>
"""


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Memuat model baseline ...")
def muat_baseline():
    import joblib
    f = DIR_MODEL / "baseline_tfidf_lr.joblib"
    if not f.exists():
        return None
    meta = AKAR / "reports" / "baseline_metrics.json"
    kolom = "text"
    if meta.exists():
        kolom = json.loads(meta.read_text(encoding="utf-8")).get("kolom_teks", "text")
    _, stop_sastrawi = muat_sastrawi()
    stop = ((stop_sastrawi or set(FALLBACK_STOP)) | EXTRA_STOP) - PROTECTED
    return joblib.load(f), kolom, stop


@st.cache_resource(show_spinner="Memuat IndoBERT ...")
def muat_indobert():
    sumber = os.environ.get("INDOBERT_REPO") or str(DIR_MODEL / "indobert_final")
    if not os.environ.get("INDOBERT_REPO") and not Path(sumber).exists():
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    tok = AutoTokenizer.from_pretrained(sumber)
    model = AutoModelForSequenceClassification.from_pretrained(sumber)
    model.eval()
    return tok, model


def proba_baseline(paket, teks):
    model, kolom, stop = paket
    bersih = bersihkan(teks)
    if kolom == "text_klasik":   # sama seperti src/preprocess.py (tanpa stemming)
        bersih = " ".join(w for w in buat_kunci(bersih).split() if w not in stop)
    p = model.predict_proba([bersih])[0]
    return dict(zip(NAMA, [float(x) for x in p]))


def proba_indobert(paket, teks):
    import torch
    tok, model = paket
    enc = tok(bersihkan(teks), truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    return dict(zip(NAMA, p))


def kelas_teratas(p):
    return max(p, key=p.get)


def kesepakatan(pb, pi):
    """Ringkasan satu kalimat: apakah dua model sepakat, dan apakah ada yang ragu. (jenis, teks)"""
    a, b = kelas_teratas(pb), kelas_teratas(pi)
    ragu = min(pb[a], pi[b]) < 0.5
    if a == b:
        if ragu:
            return "ragu", f"Kedua model condong ke **{TAMPIL[a]}**, tetapi salah satunya kurang yakin (di bawah 50%)."
        return "sepakat", f"Kedua model sepakat: **{TAMPIL[a]}**."
    return "beda", (f"Kedua model **berbeda pendapat**: baseline {TAMPIL[a]}, IndoBERT {TAMPIL[b]}. "
                    "Ulasan seperti ini sering ambigu, campuran, atau berupa sindiran.")


def batang_html(p):
    baris = []
    for k in NAMA:
        persen = p[k] * 100
        baris.append(
            f'<div class="baris"><span>{SIMBOL[k]} {TAMPIL[k]}</span><span>{persen:.0f}%</span></div>'
            f'<div class="batang"><span style="width:{persen:.1f}%;background:{WARNA[k]}"></span></div>')
    return "".join(baris)


def kartu_model(judul, p):
    atas = kelas_teratas(p)
    with st.container(border=True):
        st.markdown(f"**{judul}**")
        st.markdown(f'<div class="label-besar" style="color:{WARNA[atas]}">{SIMBOL[atas]} {TAMPIL[atas]}</div>'
                    f'<div class="kecil">{p[atas]:.0%} yakin</div>', unsafe_allow_html=True)
        st.markdown(batang_html(p), unsafe_allow_html=True)
        if p[atas] < 0.5:
            st.caption("Model ragu: tidak ada kelas di atas 50%.")


# ------------------------------------------------------------------ halaman
st.set_page_config(page_title="Suara Pengguna Paylater", page_icon=None, layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("Suara Pengguna Paylater")
st.caption("Sentimen dan keluhan pada ulasan aplikasi paylater berbahasa Indonesia (Kredivo, Akulaku, Home Credit).")

baseline, indobert = muat_baseline(), muat_indobert()
status = [f"Baseline: {'siap' if baseline else 'belum ada'}", f"IndoBERT: {'siap' if indobert else 'belum ada'}"]
st.caption("Model  |  " + "  |  ".join(status))

tab_coba, tab_tentang = st.tabs(["Coba sendiri", "Keluhan per aplikasi", "Tentang & keterbatasan"])

with tab_coba:
    if "teks" not in st.session_state:
        st.session_state["teks"] = ""
    st.markdown("**Tulis atau tempel sebuah ulasan**, atau pilih contoh:")
    kolom_contoh = st.columns(len(CONTOH))
    for kol, (nama, isi) in zip(kolom_contoh, CONTOH.items()):
        if kol.button(nama, use_container_width=True):
            st.session_state["teks"] = isi
    teks = st.text_area("Ulasan", key="teks", height=110, max_chars=1500, label_visibility="collapsed",
                        placeholder="Contoh: sudah bayar tepat waktu tapi tidak bisa pinjam lagi ...")
    analisis = st.button("Analisis", type="primary")

    if analisis:
        if not teks.strip():
            st.warning("Isi dulu teks ulasannya.")
        elif baseline is None and indobert is None:
            st.error("Belum ada model. Jalankan `python src\\train_baseline.py` (baseline), "
                     "lalu jalankan notebook `02_finetune_indobert.ipynb` untuk menyimpan IndoBERT di `models\\indobert_final`.")
        else:
            pb = proba_baseline(baseline, teks) if baseline is not None else None
            pi = proba_indobert(indobert, teks) if indobert is not None else None

            st.subheader("Sentimen")
            if pb is not None and pi is not None:
                jenis, kalimat = kesepakatan(pb, pi)
                {"sepakat": st.success, "ragu": st.info, "beda": st.warning}[jenis](kalimat)
            kol_model = st.columns(2)
            with kol_model[0]:
                if pb is not None:
                    kartu_model("Baseline TF-IDF + Logistic Regression", pb)
                else:
                    st.info("Model baseline belum ada di `models\\`.")
            with kol_model[1]:
                if pi is not None:
                    kartu_model("IndoBERT (fine-tuned)", pi)
                else:
                    st.info("IndoBERT belum ada di `models\\indobert_final`; yang ditampilkan hanya baseline.")
            st.caption("Akurasi kedua model hanya sekitar 0,63-0,67 pada data uji, dan kelas netral paling sulit. "
                       "Persentase menunjukkan keyakinan model, bukan jaminan benar. Anggap hasil ini perkiraan.")

            st.subheader("Keluhan yang terdeteksi")
            t = teks_topik(teks)
            d = kategorikan(pd.Series([t]))
            ditemukan = [k for k in SPESIFIK + GABUNGAN if bool(d[k].iloc[0])]
            if ditemukan:
                for k in ditemukan:
                    pm = pemicu(t, k)
                    with st.container(border=True):
                        st.markdown(f"**{LABEL[k]}**")
                        if pm:
                            st.caption(f"Terpicu oleh: `{pm}`")
            elif bool(d["umum_negatif"].iloc[0]):
                st.write("Hanya kekecewaan umum (mis. kata makian atau 'tidak berguna'), tanpa masalah spesifik yang dikenali pola.")
            else:
                st.write("Tidak ada keluhan spesifik yang cocok dengan pola. Ini bukan berarti tidak ada keluhan: "
                         "pola hanya menangkap sebagian keluhan.")
            st.caption("Keluhan dideteksi dengan pola aturan, bukan model. Analisis keluhan pada proyek ini hanya untuk ulasan 5 kata ke atas."
                       + (" Teks ini kurang dari 5 kata." if len(t.split()) < 5 else ""))

# with tab_hasil:
#     st.write("Persen ulasan negatif (minimal 5 kata) yang menyebut tiap keluhan. Satu ulasan boleh masuk beberapa kategori.")
#     for nama in ("keluhan_per_aplikasi.png", "akses_kredit.png"):
#         f = AKAR / "reports" / "figures" / nama
#         if f.exists():
#             st.image(str(f), use_container_width=True)
#         else:
#             st.info(f"{nama} belum ada. Jalankan `python src\\buat_grafik.py`.")

with tab_tentang:
    st.markdown(
        """
**Data.** Ulasan publik Google Play untuk Kredivo, Akulaku, dan Home Credit.

**Model.** Baseline TF-IDF + Logistic Regression dibandingkan dengan IndoBERT hasil fine-tuning. Selisihnya kecil
(sekitar 0,03 macro-F1): signifikan terhadap label bintang, tetapi belum bisa dibedakan dari derau pada label manual.

**Keluhan.** Dideteksi dengan pola aturan, bukan model. Pola divalidasi manual (presisi sekitar 80-100% untuk
sebagian besar kategori) dan hanya menangkap sebagian keluhan; sekitar seperlima ulasan negatif masih memuat keluhan
yang belum tertangkap.

**Keterbatasan.** Label training berasal dari bintang (lemah), label uji manual hanya satu penilai, dan periode ulasan
tiap aplikasi berbeda. Sindiran dan ulasan campuran (sebagian memuji, sebagian mengeluh) sulit bagi kedua model.

**Privasi.** Teks yang kamu ketik tidak disimpan oleh aplikasi ini.
"""
    )
