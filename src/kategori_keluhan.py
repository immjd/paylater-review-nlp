"""Kategori keluhan berbasis ATURAN (kata kunci) pada ulasan negatif: berapa persen ulasan tiap aplikasi menyebut tiap keluhan?

Kenapa ada skrip ini selain topic_model.py:
  BERTopic bagus untuk MENEMUKAN tema, tapi satu keluhan bisa tersebar di banyak topik dan satu topik bisa berisi
  beberapa keluhan (kata kunci topik pernah menyesatkan, contoh acaknya campuran). Persen per topik jadi tidak bisa
  dipakai untuk menjawab "keluhan apa yang paling sering?". Di sini tiap kategori didefinisikan eksplisit lewat pola
  kata (lihat KATEGORI di bawah, bisa diaudit baris per baris), dihitung pada SEMUA ulasan negatif yang dianalisis,
  lalu KETEPATANNYA diperiksa manual pada sampel (--cek, lalu --hitung-cek).

Populasi: persis sama dengan topic_model.py (fungsi muat): bintang 1-2, status unik/duplikat, minimal --min-kata kata
setelah emoji dan frasa bintang dibuang. Teks yang dicocokkan = teks_topik (sudah dinormalisasi: "ga" -> "tidak" dst).

Cara baca:
  * MULTI-LABEL: satu ulasan boleh masuk beberapa kategori, jadi persen antar kategori TIDAK dijumlahkan.
  * "Keluhan spesifik" = masuk minimal satu kategori selain "Makian/kekecewaan umum".
  * Pola buatan tangan = tidak sempurna. Presisi (cek manual) dan keluhan yang terlewat (cek manual pada ulasan tanpa kategori)
    harus dilaporkan bersama angka persennya.

Keluar:
  reports/kategori_keluhan.csv   satu baris per kategori: jumlah, persen, per aplikasi + selang kepercayaan Wilson 95%
  reports/kategori_keluhan.txt   laporan tercetak, tanpa teks ulasan
  data/processed/kategori_contoh.txt  contoh ulasan per kategori + ulasan tanpa kategori (teks mentah; lokal, tidak ikut git)
  data/labeled/validasi_kategori.xlsx  (hanya dengan --cek N) sampel untuk diperiksa manual; berisi teks ulasan, TIDAK ikut git
  reports/validasi_kategori.csv  (hanya dengan --hitung-cek) presisi per kategori dari sampel yang sudah kamu nilai

Jalankan dari akar proyek:
  python src\\kategori_keluhan.py                       # hitung
  python src\\kategori_keluhan.py --cek 12              # + buat sampel validasi
  python src\\kategori_keluhan.py --hitung-cek data\\labeled\\validasi_kategori.xlsx
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from paths import INTERIM, LABELED, PROCESSED, ROOT
from topic_model import APP, STOP_TOPIK, aman, muat

# ---------------------------------------------------------------- pola
# Teks sudah huruf kecil dan slang sudah dinormalisasi (ga/gk/gak/nggak -> "tidak", dc -> "debt collector", ...).
SEP = r"[ ,;:/\-]+"            # pemisah antar kata; tanda akhir kalimat (. ! ?) SENGAJA tidak termasuk: jendela tidak melewati kalimat
KATA = r"[\w<>']+"


def dekat(a, b, n=4):
    """a dan b muncul berdekatan (maksimal n kata di antaranya, urutan bebas, dalam satu kalimat)."""
    mid = "(?:" + SEP + KATA + "){0," + str(n) + "}?" + SEP
    return r"\b(?:" + a + ")" + mid + "(?:" + b + r")\b|\b(?:" + b + ")" + mid + "(?:" + a + r")\b"


NEG = r"tidak|belum|gagal|susah|sulit"
TELP = r"telepon|menelepon|ditelepon|telpon|menelpon|ditelpon|telp|panggilan|sms|whatsapp|miscall|missed call"
# Hanya bentuk pasif/oleh-pihak-aplikasi: "tolong hubungi saya" atau "susah menghubungi CS" bukan penagihan.
HUBUNGI = r"dihubungi|ditelepon|ditelpon|dikontak|dichat|disms|menelepon|menelpon"
BRAND = r"aplikasi|akun|app|apk|home credit|homecredit|kredivo|akulaku|aplikasinya|akunnya"
KERJA = (r"pinjam|meminjam|mengajukan|ajukan|pengajuan|pakai|dipakai|memakai|menggunakan|digunakan|transaksi|"
         r"bertransaksi|belanja|mencicil|cicil|checkout|mencairkan|cairkan")
AKHIR = r"(?=\s*(?:[.,!?;]|$))"  # akhir kalimat/klausa
# Penanda "ini soal pembayaran/utang" vs penanda "ini telepon marketing". Dipakai membedakan penagihan dari spam telepon:
# telepon yang menyebut jatuh tempo/lunas/telat = penagihan (menurut penilaian manual), telepon menawarkan produk = spam.
PAY = (r"jatuh tempo|tagih\w*|telat|terlambat|keterlambatan|lunas|dilunasi|tunggakan|menunggak|sudah (?:di ?)?bayar|sudah membayar|"
       r"belum (?:di ?)?bayar|tepat waktu|tercatat pembayaran|padahal bayar|pembayaran(?: [\w<>']+){0,2}? (?:tercatat|lancar)")
MKT = (r"menawarkan|menawari|ditawarkan|tawar\w*|promosi|promo|iklan|marketing|telemarketing|sales|tidak minat|tidak berminat|tidak butuh|"
       r"belum butuh|tidak ada kebutuhan|tidak mau|spam|nomor baru|nomer baru")
CS_NEG = (r"tidak|belum|lambat|lama|susah|sulit|buruk|jelek|payah|parah|kecewa|bohong|asal|cuek|lemot|ribet|berbelit|muter|"
          r"mengecewakan|percuma|sampah|nol")


# Ejaan yang belum tertangkap kamus slang proyek; hanya dipakai di skrip ini (data olahan tidak berubah).
NORMAL_LOKAL = {
    "gx": "tidak", "gaguna": "tidak guna", "gaminat": "tidak minat", "ditelfon": "ditelepon", "ditlfon": "ditelepon", "tlfon": "telepon",
    "tlpn": "telepon", "tlpon": "telepon", "nlp": "menelepon", "nlpn": "menelepon", "nelponin": "menelepon", "menelpon": "menelepon",
    "menelponi": "menelepon", "menelfon": "menelepon", "telponin": "menelepon", "tlat": "telat", "kenak": "kena", "kunas": "lunas",
    "nmr": "nomor", "pmbyrn": "pembayaran", "pembyrn": "pembayaran", "donlot": "download", "dowlod": "download", "dowload": "download",
    "autoll": "auto tolak", "autotolak": "auto tolak", "repon": "respon", "diteleponin": "ditelepon", "ditelponin": "ditelepon",
    "diteleponi": "ditelepon", "kaga": "tidak", "palah": "malah", "keblokir": "diblokir", "belom": "belum", "ndak": "tidak", "nda": "tidak",
    "ssh": "susah", "nelfon": "menelepon", "telfonin": "menelepon", "ditelfonin": "ditelepon", "nelepon": "menelepon", "ditlpn": "ditelepon",
    "paylatter": "paylater",
}
RE_NORMAL = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(NORMAL_LOKAL, key=len, reverse=True))) + r")\b")


# "g" tunggal sebagai singkatan "tidak" ("g bisa di buka"): hanya bila diikuti kata yang lazim sesudah negasi
RE_G = re.compile(r"\bg (?=(?:bisa|bs|ada|dapat|dpt|jelas|guna|gunanya|mau|punya|pernah|tau|tahu|usah|pake|pakai|di|dikasih|dikasi|boleh|sempat|lolos|masuk|cair)\b)")


def norm(t):
    t = re.sub(r"[\"“”«»]", " ", t)          # tanda kutip nyasar ("muncul\" penawarannya") memutus pola berurutan
    t = " ".join(RE_NORMAL.sub(lambda m: NORMAL_LOKAL[m.group(0)], t).split())
    return RE_G.sub("tidak ", t)


# (label, [pola, ...]). Urutan = urutan di berkas laporan sebelum diurutkan menurut persen.
KATEGORI = {
    "ditolak": ("Pengajuan ditolak", [
        dekat(r"tidak|belum", r"(?:di ?)?(?:acc|approve|approved|lolos|diterima)|(?:di|d) ?setuju|disetujui", 2),   # "acc" dinormalisasi jadi "setuju"; "tidak setuju" (= tak sependapat) sengaja tidak dihitung
        dekat(r"pengajuan|mengajukan|ajuan", r"gagal|tidak berhasil|tidak lolos|tidak disetujui|dibatalkan", 3),
        r"\b(?:di ?batal ?kan|dibatalin)(?: [\w<>']+){0,2}? (?:terus|melulu|sepihak)\b",
    ]),
    "limit_turun": ("Limit turun / hilang / kecil / tidak naik", [
        dekat(r"limit\w*|plafon\w*|plafond|dana cicil|kredit", r"turun|menurun|berkurang|dikurangi|diturunkan|dipotong|"
              r"hilang|ilang|menghilang|nol|kecil|mengecil|rendah|minim|menyusut|anjlok|drop|habis|dibekukan|dikunci|terkunci|mentok|stagnan|sedikit|sitik|receh", 4),
        dekat(r"limit\w*|plafon\w*|plafond", r"(?:tidak|belum|bukannya|malah)(?: pernah)? (?:naik|nambah|menambah|bertambah|ditambah|dinaikkan|menaikkan|menaikan|ada kenaikan|ada penambahan)", 5),
        dekat(r"limit\w*|plafon\w*|plafond", r"(?:tidak|belum)(?: pernah)? (?:ada|muncul|dapat|dapet|dikasih|diberi|punya|keluar)", 2),
        r"\b(?:malah|justru) (?:turun|berkurang|menurun)\b",
        r"\blimit\w*(?: [\w<>']+){0,2}? hanya \d",
        r"\b(?:tidak|belum) (?:dapat|dapet|ada|muncul|dikasih|diberi|ditawari|ditawarkan)(?: [\w<>']+){0,2}? (?:penawaran|tawaran|fasilitas|kenaikan|(?:promosi|promo) (?:pinjaman|limit|kredit|paylater))\w*\b",
        dekat(r"qris|paylater|pay later|fitur|metode", r"hilang|(?:tidak|belum) ada lagi|dihapus|dihilangkan|ditiadakan", 3),
        r"\blimit (?:tetap|segitu|segini|tidak sesuai|tidak masuk akal)\b",
        dekat(r"limit\w*|plafon\w*|plafond|dana cicil", r"kurang", 3),
        r"\b(?:tidak|belum) (?:dapat|dapet) (?:apa apa|apa-apa|apapun|apa pun)\b",
        dekat(r"limit\w*|plafon\w*|plafond|kredit", r"(?:tidak|belum) (?:di ?)?(?:pulih\w*|dikembalikan|kembali|balik|normal)", 4),
        r"\b(?:meminjam|pinjam\w*|mengajukan)(?: [\w<>']+){0,4}? (?:malah|hanya) (?:dapat |dikasih |diberi |cair )?(?:\d|rp)",
    ]),
    "tidak_bisa_pinjam_lagi": ("Tidak bisa mengajukan / pinjam / pakai limit", [
        r"\b(?:tidak|belum|susah|sulit) (?:bisa|dapat|boleh|diizinkan|diperbolehkan)(?: [\w<>']+){0,2}? (?:" + KERJA + r")(?: [\w<>']+){0,3}? lagi\b",
        r"\b(?:" + KERJA + r")(?: [\w<>']+){0,2}? lagi(?: (?:malah|kok|tapi|tetap|selalu|saja))? (?:tidak|belum) (?:bisa|dapat|boleh)\b",
        r"\b(?:tidak|belum) (?:di ?)?(?:kasih|kasi|beri|diberi|berikan|izinkan|ijinkan|bolehkan|diizinkan|diperbolehkan)\b(?! (?:tahu|kabar|info|informasi|notifikasi|konfirmasi))",
        r"\b(?:tidak|belum|susah|sulit) (?:bisa|dapat|boleh)(?: [\w<>']+){0,2}? (?:pinjam|meminjam|mengajukan|ajukan|pengajuan|mencairkan|cairkan|kredit|paylater|pay later|limit|(?:di ?)?cicil|mencicil)\b",
        r"\b(?:tidak|belum) (?:bisa|dapat|boleh) lagi(?: [\w<>']+){0,2}? (?:ambil|mengambil|pinjam|meminjam|kredit|ajukan|mengajukan|pakai|memakai|menggunakan|transaksi)\b",
        r"\b(?:tidak|belum) (?:bisa|dapat|boleh)(?: [\w<>']+){0,1}? (?:ambil|mengambil)(?: [\w<>']+){0,2}? lagi\b",
        dekat(r"limit\w*|paylater|pay later|kredit|fitur|qris|cicilan", r"(?:tidak|belum) (?:bisa|dapat|boleh)(?: [\w<>']+){0,1}? (?:(?:di ?)?(?:pakai|pake|gunakan|gunain|pergunakan|cairkan)|memakai|menggunakan|cair)", 4),
        r"\blimit\w*.{0,40}?\b(?:tidak|belum) (?:bisa|dapat|boleh) (?:di ?)?(?:pakai|pake|gunakan|digunakan)\b",
        r"\b(?:malah|tetap|masih) (?:tidak|belum) (?:bisa|dapat) (?:di ?)?(?:pakai|pake|gunakan|digunakan)\b",
        r"\b(?:tidak|belum) (?:bisa|dapat) (?:di ?)?(?:pakai|pake|gunakan|digunakan)\b(?: [\w<>']+){0,2}? (?:sama sekali|lagi|sekarang|sampai sekarang)",
        r"\b(?:pajangan|hiasan)\b",
        dekat(r"susah|sulit|dipersulit|dipersusah", r"pinjam|meminjam|kredit|mengajukan|ajukan|pengajuan|cair|dicairkan|mencairkan|limit\w*", 2),
        dekat(r"pinjam\w*|meminjam", r"hanya (?:sekali|satu kali|1 kali|1x)", 4),
        r"\b(?:pengajuan|pinjaman|ajuan)(?: [\w<>']+){0,2}? (?:tidak|belum) (?:bisa|dapat|boleh)\b",
        r"\b(?:menunggu|nunggu|tunggu)(?: [\w<>']+){0,1}? (?:sebulan|30 hari|satu bulan|1 bulan|seminggu|3 bulan|berbulan)(?: [\w<>']+){0,5}? (?:pengajuan|pinjam|meminjam|ajukan|mengajukan|berikutnya|lagi)\b",
    ]),
    "penagihan": ("Penagihan / debt collector / teror soal utang", [
        r"\b(?:debt collector|penagih\w*|nagih\w*|menagih|ditagih|tagih|tukang tagih|kontak darurat|nomor darurat)\b",
        dekat(r"teror|diteror|meneror|intimidasi|mengintimidasi|mengancam|diancam|ancam\w*|mempermalukan|dipermalukan", PAY, 6),
        dekat(r"menelepon|telepon|ditelepon|menghubungi|dihubungi|hubungi|telpon|menelpon|ditelpon|mengirim pesan|whatsapp|sms",
              r"keluarga|saudara|teman|orang tua|ortu|atasan|bos|kantor|tempat kerja|tetangga|kerabat|istri|suami|anak|rekan", 4),
        dekat(r"data|foto|ktp|identitas|nama", r"disebar|disebarkan|menyebar|menyebarkan|disebarluaskan|dipajang", 3),
        dekat(HUBUNGI, PAY, 6),
    ]),
    "spam_telepon": ("Spam & telepon marketing berlebihan", [
        r"\b(?:spam|spamming)\b",
        dekat(r"mengganggu|menganggu|terganggu|gangguan|meresahkan|resah|bawel|rese|ganggu|jengkel|menjengkelkan", TELP, 4),
        dekat(r"jangan|stop|berhenti|hentikan", TELP, 3),
        dekat(TELP, r"tidak minat|tidak butuh|tidak berminat|tidak mau|tidak perlu", 5),
        dekat(TELP, r"terus|melulu|setiap hari|tiap hari|saban hari|berkali-kali|berkali kali|puluhan kali|tanpa henti|tiap jam|setiap jam|"
              r"tengah malam|malam hari|subuh|seharian|bertubi-tubi|bertubi tubi|tidak berhenti|nonstop|non stop", 3),
        dekat(TELP, r"menawarkan|menawari|ditawarkan|tawar|tawaran|tawarannya|penawaran|promosi|iklan|marketing|telemarketing|sales", 4),
        dekat(r"dihubungi|dikontak|dichat", r"terus|melulu|sepanjang hari|setiap hari|tiap hari|saban hari|berkali-kali|berkali kali", 3),
    ]),
    "login_akun": ("Login / OTP / verifikasi wajah / akun diblokir", [
        dekat(r"(?:tidak|belum) (?:bisa|dapat|berhasil|lolos|aktif|terverifikasi|terdaftar|kunjung|selesai|bisa masuk)|gagal|susah|sulit", r"login|log in|logout|masuk (?:ke |di |dalam |ke dalam )?(?:" + BRAND + r")|daftar|mendaftar|pendaftaran|registrasi|"
              r"mendaftarkan|membuat akun|buat akun|aktivasi|mengaktifkan|diaktifkan|verifikasi|diverifikasi|memverifikasi", 4),
        r"(?<!dana )(?<!uang )(?<!saldo )(?<!transfer )(?<!pembayaran )(?<!bayar )\b(?:tidak|susah|sulit|gagal) (?:bisa )?masuk" + AKHIR,
        r"\b(?:tidak|belum|susah|sulit) (?:biasa|bisa|bs) masuk (?:ke |di |dalam )?(?:" + BRAND + r")",   # "tidak biasa masuk" = salah ketik umum
        dekat(r"login|log in", r"lama|lambat|lemot|ribet|berkali-kali|berkali kali|terus|ulang|muter|berputar|melulu", 3),
        r"\b(?:otp|kode otp|kode verifikasi|sms otp)\b",
        dekat(r"kode", r"(?:tidak|belum) (?:masuk|terkirim|datang|dikirim|muncul|diterima)", 3),
        dekat(r"akun\w*|nomor", r"(?:di ?)?(?:blokir|blok|bekukan|kunci|tangguhkan|suspend|nonaktifkan)\w*|(?:di ?)?(?:banned|ban)\b|terkunci|non ?akti[fv]\w*|nonaktif|"
              r"(?:di ?)?tutup\w*|dihapus|tidak aktif", 5),
        dekat(r"(?:di ?)?(?:blokir|blok)\w*|nonaktif\w*|(?:di ?)?tutup", r"permanen|selamanya|seumur hidup", 3),
        r"\b(?:pin|password|kata sandi|sandi)(?: saya)? lupa\b",
        r"\b(?:lupa|reset|ganti|ubah|mengganti|mengubah)(?: [\w<>']+){0,2}? (?:pin|password|kata sandi|sandi)\b",
        dekat(r"verifikasi|selfie|liveness|foto wajah|foto ktp|scan wajah|pindai wajah|wajah",
              r"gagal|tidak lolos|tidak berhasil|tidak terdeteksi|tidak terbaca|ditolak|berkali-kali|berkali kali|terus|tidak bisa|tidak sesuai|tidak cocok|error|susah|sulit|ribet|lama|lelet|payah", 4),
        dekat(r"ganti|gonta ganti|pindah", r"nomor|hp|handphone|ponsel|perangkat|device|smartphone", 2),
    ]),
    "captcha": ("CAPTCHA / verifikasi 'bukan robot'", [
        r"\b(?:captcha|kapca|capca|recaptcha|re captcha|bukan robot|verifikasi robot|puzzle|teka-teki|teka teki|pilih gambar|centang kotak|bukan bot)\b",
        dekat(r"robot", r"verifikasi|centang|buktikan|membuktikan|bukan|gambar|mobil|lampu|jembatan|zebra|pilih", 3),
    ]),
    "app_error": ("Aplikasi error / tidak bisa dibuka / lemot", [
        dekat(NEG, r"dibuka|buka|membuka|terbuka|diunduh|mengunduh|download|unduh|diinstal|install|menginstal|diinstall|"
              r"dijalankan|menjalankan|dimuat|loading|terhubung|tersambung|diakses|akses|diupdate|update|memperbarui", 3),
        r"\b(?:force close|forceclose|force stop|keluar sendiri|keluar terus|crash|crashing|error|eror|bug|buggy|lemot|lag|ngelag|"
        r"hang|macet|stuck|nyangkut|layar putih|layar hitam|blank|tidak responsif|tutup sendiri|maintenance|pemeliharaan|"
        r"server down|server error|koneksi|jaringan|tidak stabil|gangguan sistem|sistem error|sistem gangguan|sistem down)\b",
        r"\bloading (?:terus|lama|melulu|doang|saja|berputar|muter|tidak selesai)\b",
        dekat(r"update|pembaruan|versi", r"wajib|harus|dipaksa|paksa|terus|melulu|setiap", 3),
    ]),
    "biaya": ("Denda / bunga / biaya", [
        r"\b(?:denda|dendanya|didenda|bunga|berbunga|riba|ribawi|mencekik|memeras|pemerasan|penalti|pinalti|admin fee|"
        r"potongan|dipotong|memotong|potong|mahal|menjerat|jeratan)\b",
        r"\bbiaya\w*\b",
        r"\basuransi\b",
        dekat(r"kena|dikenakan|dikenai|dipotong|potong", r"admin|biaya|denda|asuransi|bunga|potongan|persen", 3),
        r"\b(?:telat|terlambat|keterlambatan)(?: [\w<>']+){0,6}? \d+ ?(?:%|persen)",
        r"\bpinjam\w*(?: [\w<>']+){0,4}? hanya (?:dapat|terima|cair|masuk)\b",
        dekat(r"cicilan|tagihan|angsuran", r"naik|membengkak|bengkak|berlipat|melonjak|tidak sesuai|salah|berbeda|lebih besar|kebesaran|tinggi", 3),
        r"\b(?:melebihi|lebih dari) (?:harga|pokok|pinjaman)\b",
        r"\b(?:\d+ ?x|\d kali|dua kali|tiga kali) lipat\b",
    ]),
    "pencairan_pembayaran": ("Dana / pembayaran tidak masuk, proses lama", [
        dekat(r"pencairan|mencairkan|cair|dana|uang|saldo|transfer|pembayaran|cicilan|angsuran|tagihan|virtual account|va",
              r"belum masuk|tidak masuk|tidak cair|belum cair|belum ada|belum dicairkan|tidak terdeteksi|tidak tercatat|belum tercatat|tidak terupdate|belum terupdate|"
              r"tidak terbaca|belum terbaca|tidak berhasil|tertahan|nyangkut|belum diproses|tidak diproses|belum diterima|tidak diterima|"
              r"double|dua kali|ganda|gagal|pending|tertunda|terpotong|raib", 4),
        r"\b(?:sudah|telah)(?: [\w<>']+){0,2}? (?:bayar|dibayar|lunas|dilunasi|membayar)(?: [\w<>']+){0,4}? "
        r"(?:masih|tetap|malah|kok)(?: [\w<>']+){0,3}? (?:ditagih|ada tagihan|tertagih|belum lunas|ditelepon|ditelpon)\b",
        dekat(r"lunas|dilunasi|melunasi|sudah (?:di ?)?bayar|membayar|tagihan|pembayaran",
              r"masih (?:di ?)?(?:tagih\w*|tertagih|menggantung|tercatat|tertera|terdaftar|menunggak|belum (?:terbayar|terbayarkan|lunas|berubah|update|terupdate|hilang))|"
              r"belum (?:terbayar|terbayarkan|terupdate|berubah|terdeteksi)", 8),
        dekat(r"proses|verifikasi|persetujuan|review|pencairan", r"lama|berhari-hari|berhari hari|seminggu|sebulan|"
              r"berminggu-minggu|berminggu minggu|berbulan|tidak ada kabar|tidak ada kejelasan|tidak ada konfirmasi|tidak selesai|tidak kunjung|belum kunjung", 4),
        r"\b(?:tidak|belum) ada (?:keterangan|status|bukti) (?:lunas|bayar|pembayaran)\b",
        r"\bbayar (?:dobel|double|dua kali|2 ?x)\b",
        dekat(r"metode pembayaran|e ?wallet|gopay|ovo|shopeepay", r"tambah\w*|terbatas|kurang|tidak ada|belum ada|minta", 4),
        r"\b(?:tidak|belum) (?:di ?)?(?:setor\w*|dibayarkan|dibayarin|disetorkan)\b",
    ]),
    "data_privasi": ("Data pribadi / hapus data / izin akses", [
        r"\b(?:hapus|menghapus|dihapus|penghapusan|hapuskan|hapusin)(?: [\w<>']+){0,2}? (?:data|akun|rekening|identitas|ktp|foto|informasi)\b",
        r"\b(?:data pribadi|data diri|privasi|privacy|penyalahgunaan|disalahgunakan|menyalahgunakan|salah gunakan|bocor|kebocoran|"
        r"dibocorkan|membocorkan|jual data|menjual data|diperjualbelikan|mata-mata|memata-matai|menyadap|disadap|dilacak)\b",
        dekat(r"akses|izin|ijin|permission|mengakses|meminta|minta|diminta", r"kontak|galeri|foto|lokasi|sms|mikrofon|kamera|file|penyimpanan|media|telepon|panggilan", 3),
        dekat(r"data|ktp|identitas|selfie|nomor", r"tersimpan|disimpan|menyimpan|permanen|tidak bisa dihapus|dipakai pinjol|dipakai orang|digunakan orang|"
              r"dicuri|mencuri|pencurian|dijual|menjual", 4),
        dekat(r"jaga|menjaga|lindungi|melindungi|amankan|mengamankan", r"data|privasi|identitas", 3),
    ]),
    "tagihan_asing": ("Tagihan muncul padahal tidak pernah pinjam", [
        dekat(r"tagihan|ditagih|utang|hutang", r"tiba tiba|(?:tidak|belum) pernah (?:pinjam|meminjam|pakai|memakai|transaksi|bertransaksi|mengajukan)|"
              r"baru daftar|(?:tidak|belum) (?:ada )?(?:pinjam|meminjam|transaksi)|tanpa (?:pinjam|transaksi)", 6),
        r"\bbaru (?:daftar|mendaftar|download|install)\b.{0,60}?\btagihan\b",
        r"\b(?:sudah|sdh) tidak (?:pakai|memakai|menggunakan)(?: [\w<>']+){0,5}? tagihan\b",
        r"\bmalah (?:ada )?tagihan\b",
        r"\b(?:transaksi sendiri|transaksi tiba tiba|ada transaksi (?:yang )?(?:bukan|tidak)|transaksi (?:yang )?(?:bukan|tidak) (?:saya|dikenal|dikenali))\b",
    ]),
    "skor_kredit": ("Skor kredit / SLIK jelek atau turun", [
        dekat(r"skor|score|slik|bi checking|bi cheking|riwayat kredit|kolektibilitas|sid ojk",
              r"turun|jelek|buruk|merah|anjlok|menurun|berkurang|rusak|tercatat|muncul|macet|kolek ?[2-5]", 5),
    ]),
    "customer_service": ("Customer service tidak membantu", [
        dekat(r"customer service|customer care|layanan pelanggan|layanan konsumen|layanan bantuan|pusat bantuan|live chat|livechat|"
              r"chatbot|chat bot|call center|callcenter|contact center|hotline|helpdesk|help desk", CS_NEG, 6),
        dekat(r"pelayanan|layanan", r"tidak ramah|tidak profesional|tidak membantu|lambat", 3),
        dekat(r"tidak|belum|lambat|lama|susah|sulit|lebih|kurang", r"direspon|merespon|respon|respons|dibalas|membalas|balasan|ditanggapi|"
              r"menanggapi|tanggapan|solusi|diselesaikan", 3),
        dekat(r"jawab\w*|balas\w*|dibalas|dijawab|respon\w*", r"robot|bot|otomatis", 3),
        r"\b(?:komplain|complain|mengadu|pengaduan)\b",
        r"\borang(?: orang)?(?: [\w<>']+){0,1}? (?:tidak ramah|kasar|jutek|judes|cuek)\b",
        dekat(r"admin|petugas|operator|agen|staf|karyawan|pegawai", r"tidak ramah|kasar|tidak membantu|tidak profesional|lambat|lama|tidak merespon|cuek|asal|tidak jelas|jutek|judes", 3),
    ]),
    "belanja_refund": ("Belanja / pulsa / listrik / barang / refund", [
        dekat(r"pulsa|kuota|token listrik|listrik|pln|ppob|bpjs|pdam|paket data|top ?up|topup", r"gagal|salah|lama|tidak masuk|belum masuk|tidak sampai|tidak terkirim|belum terkirim|hangus", 4),
        r"\b(?:refund|pengembalian dana|pengembalian uang|dikembalikan|retur|diretur|pesanan|orderan|penjual|seller|"
        r"marketplace|kurir|pengiriman|ongkir|ongkos kirim|resi|voucher|kupon|cashback|diskon|garansi)\b",
        dekat(r"barang\w*|pesanan|produk|order\w*|paket", r"tidak sesuai|rusak|cacat|palsu|bekas|salah|belum sampai|tidak sampai(?! (?:separuh|setengah|\d|rp|limit|plafon|batas))|tidak datang|lama|dikembalikan|retur|batal|dibatalkan|kembali|hilang", 4),
        dekat(r"pesanan|order|barang|transaksi", r"dibatalkan|pembatalan|batal|membatalkan", 3),
    ]),
    "lucky_draw": ("Lucky draw / undian / hadiah", [
        r"\b(?:lucky draw|luckydraw|lucky|undian|diundi|hadiah|doorprize|pemenang|grand prize|jackpot)\b",
    ]),
    "penipuan": ("Penipuan / bohong / janji palsu", [
        r"(?<!kayak )(?<!seperti )(?<!mirip )(?<!sama )(?<!bedanya sama )(?<!jangan )\b(?:penipu|penipuan|menipu|tertipu|ditipu|tipu|tipuan|scam|scammer|bodong|hoax|hoaks|abal-abal|abal abal|bohong|"
        r"membohongi|dibohongi|kebohongan|php|iklan palsu|janji palsu|omong kosong|menjebak|dijebak|"
        r"(?:tidak|belum) terdaftar(?: di)? ojk|iming ?iming|dirayu|rayu|janji manis)\b",
        r"(?<!kayak )(?<!seperti )(?<!mirip )(?<!sama )(?<!bedanya sama )(?<!jangan )\b(?:aplikasi|apk|ini|kredivo|akulaku|homecredit|home credit)(?: [\w<>']+){0,2}? ilegal\b",
        r"\btidak sesuai (?:iklan|janji|kenyataan)\b",
    ]),
    "umum_negatif": ("Memuat kata makian / kekecewaan umum (boleh bersama keluhan lain)", [
        r"\b(?:najis|norak|menjijikkan|jijik|memuakkan|sampah|tidak berguna|tidak guna|tidak bermanfaat|jelek|buruk|parah|payah|mengecewakan|kecewa|menyesal|nyesel|tidak jelas|"
        r"ribet|menyusahkan|bikin susah|bikin rugi|rugi|kapok|zonk|kesal|menyebalkan|tidak recommended|tidak direkomendasikan|tidak layak|"
        r"bobrok|hancur|amburadul|ampas|goblok|bodoh|tolol|anjing|bangsat|brengsek|sialan|menyesali|uninstall)\b",
        r"\b(?:hapus|menghapus|dihapus)(?: [\w<>']+){0,1}? (?:aplikasi|app|apk|aplikasinya)\b",
        dekat(r"jangan|hindari|menghindari", r"download|install|instal|pakai|gunakan|pinjam|meminjam|daftar|unduh", 2),
    ]),
}


RE_PAY, RE_MKT = re.compile(r"\b(?:" + PAY + r")\b"), re.compile(r"\b(?:" + MKT + r")\b")

# Kata "ditolak" dihitung KECUALI konteksnya penolakan pembayaran/transfer oleh bank (mis. "bayar ke BCA ditolak"), bukan pengajuan.
RE_TOLAK = re.compile(r"\b(?:ditolak|menolak|penolakan|tertolak|tolak|nolak)\w*\b")
RE_KONTEKS_BAYAR = re.compile(r"\b(?:pembayaran|transfer|bank|bca|bri|bni|mandiri|kartu|virtual account|top ?up|topup|qris|ovo|gopay|shopeepay)\b")
RE_KONTEKS_KREDIT = re.compile(r"\b(?:pengajuan|mengajukan|ajukan|ajuan|pinjam|pinjaman|meminjam|limit\w*|kredit|daftar|mendaftar|akun|aplikasi|cicilan|paylater)\b")


def cari_tolak(t):
    for m in RE_TOLAK.finditer(t):
        w = " ".join(t[:m.start()].split()[-4:]) + " " + " ".join(t[m.end():].split()[:4])
        if RE_KONTEKS_BAYAR.search(w) and not RE_KONTEKS_KREDIT.search(w):
            continue
        return m
    return None


# "Bayar baik": dipakai untuk kategori gabungan "lancar tapi ditutup". 'lancar' tidak dihitung bila didahului "tidak ".
LANCAR = (r"\b(?:lunas|dilunasi|melunasi|pelunasan|tepat waktu|selalu bayar|rajin bayar|disiplin|sudah dibayar|sudah di bayar|sudah bayar|"
          r"bayar terus|bayar lancar|lancar bayar|pembayaran lancar|(?:tidak|belum) pernah (?:telat|terlambat|menunggak|nunggak|gagal bayar|galbay)|"
          r"(?<!tidak )(?<!belum )lancar)\b")
TBARE = r"\b(?:tidak|belum|susah|sulit) (?:bisa|dapat|boleh)(?: [\w<>']+){0,2}? (?:" + KERJA + r")\b"

RX = {k: re.compile("|".join(f"(?:{p})" for p in pola)) for k, (_, pola) in KATEGORI.items()}
RX_LANCAR, RX_TBARE = re.compile(LANCAR), re.compile(TBARE)
LABEL = {k: v[0] for k, v in KATEGORI.items()}
SPESIFIK = [k for k in KATEGORI if k != "umum_negatif"]
LABEL.update({
    "akses_kredit": "GABUNGAN: akses kredit tertutup (ditolak / limit turun / tidak bisa pinjam lagi)",
    "lancar_tapi_ditutup": "GABUNGAN: bayar lancar / lunas, tetapi akses kredit tertutup",
    "spesifik": "RINGKASAN: menyebut minimal satu keluhan spesifik",
    "hanya_umum": "RINGKASAN: hanya makian / kekecewaan umum, tanpa keluhan spesifik",
    "tanpa_kategori": "RINGKASAN: tidak masuk kategori mana pun",
})
GABUNGAN = ["akses_kredit", "lancar_tapi_ditutup"]
RINGKASAN = ["spesifik", "hanya_umum", "tanpa_kategori"]
LAINNYA = "(tanpa kategori spesifik)"

OUT = []


def cetak(teks=""):
    print(teks)
    OUT.append(str(teks))


def wilson(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- inti
def kategorikan(teks):
    """Series teks -> DataFrame boolean per kategori (dasar + gabungan + ringkasan)."""
    teks = teks.map(norm)
    d = pd.DataFrame({k: teks.map(lambda t, rx=rx: rx.search(t) is not None) for k, rx in RX.items()}, index=teks.index)
    d["ditolak"] = d["ditolak"] | teks.map(lambda t: cari_tolak(t) is not None)
    # Telepon yang menyebut jatuh tempo/lunas/telat dan tidak menawarkan apa pun = penagihan, bukan spam marketing.
    mkt = teks.map(lambda t: RE_MKT.search(t) is not None)
    pay = teks.map(lambda t: RE_PAY.search(t) is not None)
    d["spam_telepon"] = d["spam_telepon"] & (mkt | ~pay)
    lancar = teks.map(lambda t: RX_LANCAR.search(t) is not None)
    tbare = teks.map(lambda t: RX_TBARE.search(t) is not None)
    d["tidak_bisa_pinjam_lagi"] = d["tidak_bisa_pinjam_lagi"] | (tbare & lancar)
    d["akses_kredit"] = d[["ditolak", "limit_turun", "tidak_bisa_pinjam_lagi"]].any(axis=1)
    d["lancar_tapi_ditutup"] = d["akses_kredit"] & lancar
    d["spesifik"] = d[SPESIFIK].any(axis=1)
    d["hanya_umum"] = d["umum_negatif"] & ~d["spesifik"]
    d["tanpa_kategori"] = ~d["spesifik"] & ~d["umum_negatif"]
    return d


def pemicu(teks, kunci):
    """Potongan teks yang membuat kategori `kunci` cocok (untuk membantu menilai sampel)."""
    teks = norm(teks)
    def cocok(k):
        return RX[k].search(teks) or (cari_tolak(teks) if k == "ditolak" else None)

    if kunci in RX:
        m = cocok(kunci)
    elif kunci == "akses_kredit":
        m = next((mm for k in ("ditolak", "limit_turun", "tidak_bisa_pinjam_lagi") if (mm := cocok(k))), None)
    elif kunci == "lancar_tapi_ditutup":
        m1, m2 = RX_LANCAR.search(teks), next((mm for k in ("ditolak", "limit_turun", "tidak_bisa_pinjam_lagi") if (mm := cocok(k))), None)
        return " | ".join(x.group(0) for x in (m1, m2) if x)
    else:
        m = None
    return m.group(0) if m else ""


def kata_terlewat(teks_klasik, k=15, minimal=5):
    """Kata/bigram paling sering di ulasan yang tidak masuk kategori apa pun: petunjuk pola yang kurang.
    Dihitung per TEKS UNIK (salinan tidak menggelembungkan) dan hanya yang muncul di >= `minimal` teks, supaya nama atau
    kata langka dari satu ulasan tidak ikut tercetak ke laporan yang masuk git."""
    uni, bi = Counter(), Counter()
    for t in teks_klasik:
        w = [x for x in t.split() if len(x) > 2 and x not in STOP_TOPIK and not x.startswith("<")]
        uni.update(set(w))
        bi.update(set(zip(w, w[1:])))
    return ([(w, c) for w, c in uni.most_common(k) if c >= minimal],
            [(" ".join(g), c) for g, c in bi.most_common(k) if c >= minimal])


def laporan(analisis, args):
    apps = [a for a in APP if a in set(analisis["app_id"])] + sorted(set(analisis["app_id"]) - set(APP))
    nama = {a: APP.get(a, a) for a in apps}
    d = kategorikan(analisis["teks_topik"])
    cols = list(KATEGORI) + GABUNGAN + RINGKASAN
    n_app = analisis.groupby("app_id").size().reindex(apps)
    uniq = (analisis["status"] == "unik").to_numpy()
    n_all, n_uniq = len(analisis), int(uniq.sum())

    cetak("=" * 86)
    cetak("KATEGORI KELUHAN BERBASIS ATURAN (ulasan negatif yang dianalisis; MULTI-LABEL, persen tidak dijumlahkan)")
    cetak("=" * 86)
    cetak(f"  Dianalisis: {n_all:,} ulasan ({n_uniq:,} teks unik) | " + " | ".join(f"{nama[a]} {int(n_app[a]):,}" for a in apps))
    cetak("  Populasi sama dengan topic_model.py. Tiap aplikasi di-scrape pada rentang waktu berbeda: bandingkan PERSEN.")

    baris = []
    for k in cols:
        v = d[k].to_numpy()
        row = {"kunci": k, "kategori": LABEL[k], "jenis": "dasar" if k in SPESIFIK else ("gabungan" if k in GABUNGAN else "ringkasan"),
               "n_ulasan": int(v.sum()), "pct_ulasan": round(100 * v.mean(), 1),
               "n_teks_unik": int(v[uniq].sum()), "pct_teks_unik": round(100 * v[uniq].mean(), 1)}
        for a in apps:
            m = (analisis["app_id"] == a).to_numpy()
            kk, nn = int(v[m].sum()), int(m.sum())
            lo, hi = wilson(kk, nn)
            row.update({f"n_{nama[a]}": kk, f"pct_{nama[a]}": round(100 * kk / nn, 1),
                        f"lo_{nama[a]}": round(100 * lo, 1), f"hi_{nama[a]}": round(100 * hi, 1)})
        baris.append(row)
    ring = pd.DataFrame(baris)

    def tulis(sub, judul):
        cetak("\n" + judul)
        cetak(f"  {'%semua':>7} {'ulasan':>7}  " + " ".join(f"{nama[a][:9]:>9}" for a in apps) + "   kategori")
        for r in sub.itertuples(index=False):
            rd = r._asdict()
            cetak(f"  {r.pct_ulasan:>6.1f}% {r.n_ulasan:>7,}  " + " ".join(f"{rd['pct_' + nama[a]]:>8.1f}%" for a in apps) + f"   {r.kategori}")

    dasar = ring[ring["jenis"] == "dasar"].sort_values("pct_ulasan", ascending=False)
    tulis(dasar, "KATEGORI KELUHAN SPESIFIK (urut dari paling sering)")
    tulis(ring[ring["jenis"] == "gabungan"], "KATEGORI GABUNGAN (tumpang-tindih dengan kategori dasar)")
    tulis(ring[ring["jenis"] == "ringkasan"], "RINGKASAN")
    cetak("\n  Selang kepercayaan 95% (Wilson, ulasan dianggap independen; salinan membuatnya sedikit terlalu sempit) ada di reports/kategori_keluhan.csv.")
    cetak("\n  3 kategori dasar teratas per aplikasi:")
    for a in apps:
        atas = dasar.sort_values(f"pct_{nama[a]}", ascending=False).head(3)
        cetak(f"    {nama[a]}: " + "; ".join(f"{r.kategori} {getattr(r, 'pct_' + nama[a]):.1f}%" for r in atas.itertuples(index=False)))

    if args.kata_lainnya > 0:
        sisa_semua = int(d["tanpa_kategori"].sum())
        sisa = analisis.loc[d["tanpa_kategori"].to_numpy() & uniq, "klasik_topik"]
        uni, bi = kata_terlewat(sisa, args.kata_lainnya, args.kata_min)
        cetak(f"\n  Ulasan tanpa kategori apa pun: {sisa_semua:,} ({len(sisa):,} teks unik). Kata paling sering (jumlah TEKS UNIK, minimal {args.kata_min}) - "
              "petunjuk pola yang masih kurang:")
        cetak("    kata  : " + ", ".join(f"{w} ({c})" for w, c in uni))
        cetak("    bigram: " + ", ".join(f"{w} ({c})" for w, c in bi))
    return ring, d


def tulis_contoh(analisis, d, args):
    """Contoh ulasan per kategori + ulasan tanpa kategori ke berkas LOKAL (teks mentah, PII disamarkan; tidak ikut git)."""
    rng = np.random.default_rng(args.seed + 1)
    uni_idx = np.flatnonzero((analisis["status"] == "unik").to_numpy())
    baris = []

    def bagian(judul, mask, n, kunci=None, lebar=190):
        cand = uni_idx[mask[uni_idx]]
        baris.append(f"[{judul}] | {int(mask.sum()):,} ulasan ({len(cand):,} teks unik)")
        for i in (rng.choice(cand, size=min(n, len(cand)), replace=False) if len(cand) else []):
            r = analisis.iloc[i]
            pm = f"   <<{pemicu(r['teks_topik'], kunci)}>>" if kunci else ""
            baris.append(f"  ({APP.get(r['app_id'], r['app_id'])}, {int(r['score'])} bintang) {aman(r['content'], lebar)}{pm}")
        baris.append("")

    bagian("tanpa_kategori", d["tanpa_kategori"].to_numpy(), args.contoh_tanpa)          # bagian pertama: paling penting untuk menyetel pola
    bagian("hanya_umum", d["hanya_umum"].to_numpy(), min(args.contoh_tanpa, 12))
    for k in SPESIFIK + GABUNGAN:
        bagian(k, d[k].to_numpy(), args.contoh, kunci=k)
    Path(args.out_contoh).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_contoh).write_text("\n".join(baris) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- validasi manual
def buat_cek(analisis, d, args):
    """Sampel per kategori (hanya teks unik) untuk dinilai manual: apakah ulasan ini benar mengeluhkan hal tersebut?"""
    out = Path(args.out_cek)
    if out.exists():   # pengaman: jangan menimpa hasil kerja menilai manual
        try:
            lama_x = pd.read_excel(out, sheet_name="sampel", dtype=str)
            terisi = int(lama_x["benar"].fillna("").astype(str).str.strip().ne("").sum())
        except Exception:
            terisi = 0
        if terisi:
            raise SystemExit(f"{out} sudah berisi {terisi} penilaian di kolom 'benar' dan TIDAK akan ditimpa. "
                             "Pakai --out-cek dengan nama lain (mis. data\\labeled\\validasi_kategori_putaran2.xlsx).")
    rng = np.random.default_rng(args.seed)
    uni_idx = np.flatnonzero((analisis["status"] == "unik").to_numpy())
    if args.kecuali:   # sampel putaran 2: jangan pakai ulasan yang sudah dinilai di putaran 1
        lama = []
        for f in [x.strip() for x in args.kecuali.split(",") if x.strip()]:
            lama += pd.read_excel(f, sheet_name="sampel", dtype=str)["id"].dropna().tolist()
        uni_idx = uni_idx[~np.isin(analisis["reviewId"].to_numpy()[uni_idx], lama)]
    baris = []

    def ambil(mask, n, kunci, label):
        cand = uni_idx[mask[uni_idx]]
        pilih = rng.choice(cand, size=min(n, len(cand)), replace=False) if len(cand) else []
        for i in pilih:
            r = analisis.iloc[i]
            baris.append({"id": r["reviewId"], "kategori_diuji": kunci, "kategori_arti": label, "app": APP.get(r["app_id"], r["app_id"]),
                          "bintang": int(r["score"]), "teks": aman(r["content"], 600),
                          "pemicu": pemicu(r["teks_topik"], kunci), "benar": "", "kategori_manual": ""})
        return len(cand)

    jumlah = {}
    pilih = [k.strip() for k in args.cek_hanya.split(",") if k.strip()]
    salah = [k for k in pilih if k not in SPESIFIK + GABUNGAN]
    if salah:
        raise SystemExit(f"--cek-hanya: kategori tidak dikenal {salah}. Pilihan: {', '.join(SPESIFIK + GABUNGAN)}")
    for k in SPESIFIK + GABUNGAN:      # 'umum_negatif' tidak dinilai: yang penting adalah ulasan tanpa kategori spesifik di bawah
        if pilih and k not in pilih:
            continue
        jumlah[k] = ambil(d[k].to_numpy(), args.cek, k, LABEL[k])
    if args.cek_lainnya > 0:
        jumlah[LAINNYA] = ambil(~d["spesifik"].to_numpy(), args.cek_lainnya, LAINNYA, "Tidak ada kategori spesifik yang cocok")
    x = pd.DataFrame(baris)
    Path(args.out_cek).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out_cek, engine="openpyxl") as w:
        x.to_excel(w, index=False, sheet_name="sampel")
        pet = pd.DataFrame({"petunjuk": [
            "Kolom 'benar': isi 1 / 0 / ? untuk setiap baris.",
            "Baris kategori biasa: 1 = ulasan ini MEMANG mengeluhkan hal pada kolom 'kategori_arti'; 0 = bukan (pola salah tangkap); ? = ragu.",
            f"Baris '{LAINNYA}': 1 = sebenarnya ADA keluhan spesifik yang seharusnya masuk salah satu kategori (boleh tulis kuncinya di 'kategori_manual' bila mau, "
            "mis. ditolak; tidak wajib); 0 = memang tidak ada keluhan spesifik (hanya makian / terlalu pendek / tidak jelas).",
            "Nilai keluhan yang DITULIS ulasannya, bukan menebak maksudnya. Satu ulasan boleh mengandung beberapa keluhan.",
            "Simpan sebagai .xlsx (jangan CSV) lalu jalankan: python src\\kategori_keluhan.py --hitung-cek data\\labeled\\validasi_kategori.xlsx",
            "Berkas ini berisi teks ulasan asli: JANGAN di-commit (folder data/ sudah di-gitignore)."]})
        pet.to_excel(w, index=False, sheet_name="petunjuk")
        ws = w.sheets["sampel"]
        for kol, lebar in zip("ABCDEFGHI", (12, 26, 40, 11, 8, 90, 38, 8, 22)):
            ws.column_dimensions[kol].width = lebar
        from openpyxl.styles import Alignment
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        w.sheets["petunjuk"].column_dimensions["A"].width = 150
    cetak(f"\nSampel validasi: {len(x)} baris -> {args.out_cek}  (lokal, tidak ikut git)")
    kurang = [k for k, n in jumlah.items() if n < args.cek and k != LAINNYA]
    if kurang:
        cetak("  Kategori dengan kandidat lebih sedikit dari sampel (semua diambil): " + ", ".join(f"{k} ({jumlah[k]})" for k in kurang))


def baca_nilai(v):
    s = str(v).strip().lower()
    if s in ("1", "1.0", "y", "ya", "benar", "true"):
        return 1
    if s in ("0", "0.0", "n", "tidak", "salah", "false"):
        return 0
    if s in ("?", "ragu"):
        return -1
    return None


def hitung_cek(path, args):
    x = pd.read_excel(path, sheet_name="sampel", dtype=str)
    perlu = {"kategori_diuji", "benar"}
    if not perlu <= set(x.columns):
        raise SystemExit(f"{path}: kolom wajib {sorted(perlu)}, ditemukan {list(x.columns)}")
    x["nilai"] = x["benar"].map(baca_nilai)
    ref = None
    if Path(args.out_csv).exists():
        ref = pd.read_csv(args.out_csv).set_index("kunci")
    cetak("=" * 86)
    cetak("VALIDASI MANUAL KATEGORI (presisi = dari ulasan yang dicocokkan pola, berapa yang memang mengeluhkan hal itu)")
    cetak("=" * 86)
    cetak(f"  {'kategori':<26}{'dinilai':>8}{'benar':>7}{'presisi':>9}{'  selang 95%':<16}{'ragu':>5}{'kosong':>7}   perkiraan % terkoreksi")
    hasil = []
    for k, g in x.groupby("kategori_diuji", sort=False):
        n_kosong, n_ragu = int(g["nilai"].isna().sum()), int((g["nilai"] == -1).sum())
        v = g["nilai"][g["nilai"].isin([0, 1])]
        n, b = len(v), int(v.sum())
        lo, hi = wilson(b, n)
        p = b / n if n else np.nan
        if k == LAINNYA:
            ket = ""
            if ref is not None and {"spesifik", "hanya_umum", "tanpa_kategori"} <= set(ref.index) and n:
                sisa = 100 - ref.loc["spesifik", "pct_ulasan"]
                ket = f"~{sisa * p:.1f}% ulasan negatif menyebut keluhan spesifik yang terlewat (dari {sisa:.1f}% tanpa kategori spesifik)"
            cetak(f"\n  Ulasan TANPA kategori spesifik: {n} dinilai, {b} ternyata menyebut keluhan spesifik yang terlewat = {p * 100:.0f}% "
                  f"[selang 95% {lo * 100:.0f}-{hi * 100:.0f}%] (ragu {n_ragu}, kosong {n_kosong})")
            if ket:
                cetak(f"    -> {ket}")
        else:
            ket = f"{ref.loc[k, 'pct_ulasan'] * p:.1f}% (dari {ref.loc[k, 'pct_ulasan']:.1f}%)" if (ref is not None and k in ref.index and n) else ""
            cetak(f"  {k:<26}{n:>8}{b:>7}{p * 100:>8.0f}%  [{lo * 100:>3.0f}-{hi * 100:>3.0f}%]  {n_ragu:>5}{n_kosong:>7}   {ket}")
        hasil.append({"kategori": k, "ukuran": "terlewat" if k == LAINNYA else "presisi", "n_dinilai": n, "n_benar": b,
                      "nilai": round(p, 3) if n else np.nan,
                      "lo95": round(lo, 3) if n else np.nan, "hi95": round(hi, 3) if n else np.nan, "n_ragu": n_ragu, "n_kosong": n_kosong})
    cetak("\n  'perkiraan % terkoreksi' = persen tercetak x presisi: kasar, dan selang presisi lebar bila sampel kecil (lihat selang 95%).")
    cetak("  Baris tanpa nilai ('kosong') tidak dihitung; ulasan yang kamu tandai '?' juga tidak dihitung.")
    out = Path(args.out_validasi)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hasil).to_csv(out, index=False, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Kategori keluhan berbasis aturan pada ulasan negatif.")
    ap.add_argument("--inp", default=str(INTERIM / "reviews_clean.csv"))
    ap.add_argument("--min-kata", type=int, default=5)
    ap.add_argument("--kata-lainnya", type=int, default=15, help="cetak sekian kata/bigram teratas dari ulasan tanpa kategori; 0 = jangan")
    ap.add_argument("--kata-min", type=int, default=5, help="kata/bigram hanya dicetak bila muncul di minimal sekian teks unik")
    ap.add_argument("--contoh", type=int, default=8, help="jumlah contoh acak per kategori di berkas contoh")
    ap.add_argument("--contoh-tanpa", type=int, default=40, help="jumlah contoh acak dari ulasan tanpa kategori di berkas contoh")
    ap.add_argument("--out-contoh", default=str(PROCESSED / "kategori_contoh.txt"))
    ap.add_argument("--cek", type=int, default=0, help="buat sampel validasi sebesar sekian ulasan per kategori; 0 = tidak membuat")
    ap.add_argument("--cek-lainnya", type=int, default=50, help="ukuran sampel dari ulasan tanpa kategori spesifik; 0 = tanpa bagian ini")
    ap.add_argument("--cek-hanya", default="", help="batasi sampel ke kategori ini saja (kunci dipisah koma), mis. penagihan,spam_telepon")
    ap.add_argument("--kecuali", default="", help="path .xlsx sampel sebelumnya (boleh beberapa, pisah koma): ulasan di dalamnya tidak diambil lagi")
    ap.add_argument("--hitung-cek", default="", help="path .xlsx sampel yang sudah dinilai: hitung presisi per kategori")
    ap.add_argument("--out-cek", default=str(LABELED / "validasi_kategori.xlsx"))
    ap.add_argument("--out-validasi", default=str(ROOT / "reports" / "validasi_kategori.csv"))
    ap.add_argument("--out-csv", default=str(ROOT / "reports" / "kategori_keluhan.csv"))
    ap.add_argument("--out-txt", default=str(ROOT / "reports" / "kategori_keluhan.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.hitung_cek:
        out = hitung_cek(args.hitung_cek, args)
        Path(args.out_validasi).with_suffix(".txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
        print(f"\nTersimpan: {out}")
        return

    df, neg, _ = muat(args)
    analisis = neg[neg["n_topik"] >= args.min_kata].reset_index(drop=True)
    ring, d = laporan(analisis, args)
    tulis_contoh(analisis, d, args)
    if args.cek > 0:
        buat_cek(analisis, d, args)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    ring.to_csv(args.out_csv, index=False, encoding="utf-8")
    Path(args.out_txt).write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"\nTersimpan:\n  {args.out_csv}\n  {args.out_txt}\n  {args.out_contoh}  (contoh ulasan; lokal, tidak ikut git, TIDAK dicetak)")
    if args.cek > 0:
        print("Langkah berikutnya: nilai kolom 'benar' di berkas sampel, lalu jalankan --hitung-cek.")


if __name__ == "__main__":
    main()
