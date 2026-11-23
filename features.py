import numpy as np
import pandas as pd

TARGET = "species"

# Kolom mentah yang dipakai sebagai fitur (input aplikasi = kolom-kolom ini).
RAW_NUM = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
RAW_CAT = ["island", "sex"]
INPUT_COLS = RAW_CAT + RAW_NUM

# Fitur turunan (dibuat di dalam pipeline, bukan oleh pengguna).
ENGINEERED = ["bill_ratio", "flipper_mass_ratio"]

# Kolom yang sengaja dibuang dari fitur beserta alasannya.
DROPPED = {
    "year": "Tahun pengamatan bukan ciri biologis penguin; hanya penanda waktu "
            "pencatatan. Distribusi spesies per tahun hampir sama sehingga tidak "
            "informatif, dan berisiko membuat model menghafal artefak sampling.",
}


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur rasio. Aman untuk NaN dan pembagian nol."""
    X = X.copy()
    # Bentuk paruh: ramping-panjang (Gentoo/Chinstrap) vs pendek-tebal (Adelie).
    X["bill_ratio"] = X["bill_length_mm"] / X["bill_depth_mm"]
    # Panjang sirip relatif terhadap massa (massa dalam kg agar skalanya wajar).
    X["flipper_mass_ratio"] = X["flipper_length_mm"] / (X["body_mass_g"] / 1000.0)
    X[ENGINEERED] = X[ENGINEERED].replace([np.inf, -np.inf], np.nan)
    return X
