"""Kamus normalisasi slang + daftar kata yang dilindungi dari penghapusan stopword.

Kamus ini AWAL. Tambahkan entri baru dari data/interim/slang_kandidat.csv (dibuat oleh
preprocess.py): kata yang sering muncul tapi tidak dikenal kamus bahasa Indonesia.
Aturan: kunci selalu huruf kecil, satu kata, tanpa spasi.
"""

SLANG = {
    # negasi
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "ngak": "tidak", "tdk": "tidak", "tak": "tidak", "blm": "belum", "blum": "belum",
    "jgn": "jangan", "gada": "tidak ada", "gaada": "tidak ada", "gabisa": "tidak bisa",
    # kata umum
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "krn": "karena", "karna": "karena", "krna": "karena",
    "utk": "untuk", "untk": "untuk", "tp": "tapi", "tpi": "tapi", "jd": "jadi", "jdi": "jadi",
    "dr": "dari", "dri": "dari", "sm": "sama", "klo": "kalau", "kalo": "kalau", "kalu": "kalau",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "dah": "sudah", "sudh": "sudah",
    "lg": "lagi", "lgi": "lagi", "lgsg": "langsung", "langsg": "langsung",
    "bgt": "banget", "bngt": "banget", "bs": "bisa", "bsa": "bisa",
    "trs": "terus", "trus": "terus", "skrg": "sekarang", "skrng": "sekarang", "skg": "sekarang",
    "byk": "banyak", "bnyk": "banyak", "knp": "kenapa", "knapa": "kenapa",
    "gmn": "bagaimana", "gimana": "bagaimana", "gmna": "bagaimana", "bgmn": "bagaimana",
    "emg": "memang", "emang": "memang", "mmg": "memang", "kyk": "seperti", "kayak": "seperti", "kyak": "seperti",
    "sy": "saya", "saia": "saya", "aq": "aku", "gw": "saya", "gue": "saya", "gua": "saya",
    "org": "orang", "orng": "orang", "pake": "pakai", "hrs": "harus", "lbh": "lebih", "kmrn": "kemarin",
    "bln": "bulan", "hr": "hari", "thn": "tahun", "jt": "juta", "rb": "ribu",
    "tlg": "tolong", "tolng": "tolong", "plis": "tolong", "pls": "tolong",
    "sbg": "sebagai", "dlm": "dalam", "pd": "pada", "ttg": "tentang", "msh": "masih", "bru": "baru",
    "tdi": "tadi", "gtu": "begitu", "gitu": "begitu", "gini": "begini",
    "sampe": "sampai", "smpe": "sampai", "sampek": "sampai", "aja": "saja", "ajah": "saja",
    "doang": "saja", "cuma": "hanya", "cuman": "hanya", "mudh": "mudah", "kcwa": "kecewa",
    # dari kandidat di data (iterasi 2)
    "bet": "banget", "liat": "lihat", "mmbantu": "membantu", "km": "kamu", "kmu": "kamu", "dlu": "dulu",
    "gausah": "tidak usah", "ngajuin": "mengajukan", "minjam": "meminjam", "minjem": "meminjam",
    "pinjem": "pinjam", "mulu": "melulu", "sya": "saya", "ku": "aku", "tgl": "tanggal",
    "bener": "benar", "cepet": "cepat", "dapet": "dapat", "dpt": "dapat", "nunggu": "menunggu",
    "ni": "ini", "ny": "nya", "ko": "kok", "pdhl": "padahal", "jg": "juga", "tetep": "tetap",
    "slalu": "selalu", "nelpon": "menelepon", "lunasin": "melunasi", "ok": "oke",
    # dari kandidat di data (iterasi 3)
    "tlp": "telepon", "telfon": "telepon", "byr": "bayar", "dipake": "dipakai", "ngasih": "memberi",
    "nawarin": "menawarkan", "nyesel": "menyesal", "gajelas": "tidak jelas", "naikin": "menaikkan",
    "ngk": "tidak", "dn": "dan", "knpa": "kenapa", "ngambil": "mengambil", "nunggak": "menunggak",
    "nomer": "nomor", "yaa": "ya", "instal": "install", "uninstal": "uninstall",
    "apl": "aplikasi", "pdhal": "padahal", "trimaksih": "terima kasih", "kgk": "tidak", "kagak": "tidak",
    # singkatan domain
    "dc": "debt collector", "cs": "customer service", "wa": "whatsapp",
    # pujian
    "mantab": "mantap", "mantep": "mantap", "mantapp": "mantap", "bgus": "bagus", "bagu": "bagus",
    "bgs": "bagus", "baguss": "bagus", "trbaik": "terbaik", "tebaik": "terbaik",
    # terima kasih
    "makasih": "terima kasih", "mksh": "terima kasih", "trims": "terima kasih", "tks": "terima kasih",
    "thanks": "terima kasih", "thx": "terima kasih", "tq": "terima kasih",
    "terimakasih": "terima kasih", "trimakasih": "terima kasih",
    # istilah aplikasi / keuangan
    "apk": "aplikasi", "app": "aplikasi", "acc": "setuju", "diacc": "disetujui", "dacc": "disetujui",
    "komen": "komentar",
}

# Kata yang TIDAK boleh dibuang sebagai stopword: negasi dan penanda kontras yang membawa sentimen.
# (Daftar stopword Sastrawi memuat tidak, belum, tanpa, bisa, sudah, ada.)
PROTECTED = {
    "tidak", "tak", "bukan", "belum", "jangan", "kurang", "tanpa", "bisa", "sudah", "ada",
    "padahal", "malah", "pernah",
}

# Tambahan stopword khusus ulasan (partikel percakapan).
EXTRA_STOP = {"ya", "nih", "sih", "dong", "deh", "kok", "lah", "kan", "yah", "nya"}

# Dipakai hanya kalau Sastrawi tidak terpasang.
FALLBACK_STOP = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "saya", "aku", "juga", "lagi",
    "atau", "pada", "adalah", "akan", "oleh", "saja", "hanya", "sebagai", "dalam", "agar", "supaya",
    "karena", "para", "dapat", "harus", "masih", "mereka", "kami", "kita", "anda",
}
