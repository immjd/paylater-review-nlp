"""Satu tempat untuk semua path proyek, supaya skrip tidak bergantung pada
folder tempat perintah dijalankan (cwd). Impor dari skrip lain di src/:

    from paths import RAW, INTERIM, LABELED, PROCESSED
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"              # hasil scraping mentah, satu CSV per aplikasi
INTERIM = DATA / "interim"      # setelah cleaning / normalisasi
LABELED = DATA / "labeled"      # sampel yang dilabeli manual (test set emas)
PROCESSED = DATA / "processed"  # train / val / test final

MODELS = ROOT / "models"        # checkpoint IndoBERT, model baseline
FIGURES = ROOT / "reports" / "figures"
