import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import DROPPED, ENGINEERED, INPUT_COLS, RAW_CAT, RAW_NUM, TARGET, add_features

# Semua path relatif terhadap lokasi file ini (tidak ada path absolut).
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "kasus03_penguins.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "penguin_pipeline.joblib"

SEED = 42
TEST_SIZE = 0.2

# Batas kewajaran biologis (untuk audit nilai tidak wajar).
PLAUSIBLE = {
    "bill_length_mm": (25, 70),
    "bill_depth_mm": (8, 25),
    "flipper_length_mm": (150, 260),
    "body_mass_g": (2000, 7500),
}


# --------------------------------------------------------------------------- audit
def audit(df: pd.DataFrame) -> dict:
    """Audit deskriptif: duplikat, missing, nilai tidak wajar, tipe data."""
    report = {
        "n_rows": int(len(df)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "rows_all_measurements_missing": int(df[RAW_NUM].isna().all(axis=1).sum()),
        "implausible_values": {},
        "class_counts": df[TARGET].value_counts().to_dict(),
        "unexpected_categories": {},
    }
    for col, (lo, hi) in PLAUSIBLE.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        report["implausible_values"][col] = int(bad.sum())
    for col in RAW_CAT:
        report["unexpected_categories"][col] = sorted(df[col].dropna().unique().tolist())
    return report


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Pastikan kolom kategorikal bertipe object & nilai dinormalisasi (huruf kecil, tanpa spasi).
    for col in RAW_CAT:
        df[col] = df[col].astype("object").where(df[col].isna(), df[col].astype(str).str.strip().str.lower())
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


# ------------------------------------------------------------------------ pipeline
def make_pipeline(model, engineered: bool = True, extra_num=(), use_island: bool = True) -> Pipeline:
    """Pipeline lengkap: feature engineering -> imputasi/scaling/encoding -> model.

    Semua statistik (median, mean/std, kategori, modus) dipelajari di .fit(),
    sehingga hanya dihitung dari data yang diberikan ke fit (= data train / fold train).
    """
    num_cols = RAW_NUM + (ENGINEERED if engineered else []) + list(extra_num)
    transformers = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        # sex: NaN jadi kategori "unknown" (informasi 'tidak tercatat' dipertahankan,
        # tidak memalsukan jenis kelamin).
        ("sex", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), ["sex"]),
    ]
    if use_island:
        transformers.append(
            ("island", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("oh", OneHotEncoder(handle_unknown="ignore"))]), ["island"]))
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", ColumnTransformer(transformers, remainder="drop")),
        ("clf", model),
    ])


def candidate_models() -> dict:
    """Tiga algoritma + grid hiperparameter kecil (dicari lewat CV di data train)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=SEED),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED),
            {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5],
             "clf__min_samples_leaf": [1, 3]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]},
        ),
    }


# ---------------------------------------------------------------------------- main
def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()

    # 1) AUDIT -------------------------------------------------------------------
    audit_report = audit(df)
    print("=== AUDIT DATA ===")
    print(json.dumps(audit_report, indent=2, default=str))

    # 2) Buang baris tanpa informasi sama sekali (semua pengukuran kosong).
    #    Ini aturan tetap per-baris (bukan statistik dari data), jadi aman dilakukan sebelum split.
    empty = df[RAW_NUM].isna().all(axis=1)
    print(f"\nBaris dibuang karena seluruh pengukuran kosong: {int(empty.sum())}")
    df = df.loc[~empty].reset_index(drop=True)

    # 3) SPLIT dulu, sebelum imputasi/scaling/encoding --------------------------
    X, y = df[INPUT_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print("Missing di train (sebelum imputasi):", X_train.isna().sum()[lambda s: s > 0].to_dict())

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    # 4a) Ablasi fitur (CV di train saja) --------------------------------------
    print("\n=== ABLASI FITUR (CV f1_macro, data train) ===")
    ablation_defs = {
        "A. Mentah (4 numerik + island + sex)": dict(engineered=False),
        "B. A + fitur rasio  [DIPILIH]": dict(engineered=True),
        "C. B + year": dict(engineered=True, extra_num=("year",)),
        "D. B tanpa island": dict(engineered=True, use_island=False),
    }
    ablation_rows = []
    for name, kw in ablation_defs.items():
        row = {"feature_set": name}
        for mname, mdl in {"LogReg": LogisticRegression(max_iter=5000, random_state=SEED),
                           "RandForest": RandomForestClassifier(n_estimators=300, random_state=SEED)}.items():
            Xtr = X_train.copy()
            if "year" in kw.get("extra_num", ()):
                Xtr["year"] = df.loc[X_train.index, "year"]
            s = cross_val_score(make_pipeline(mdl, **kw), Xtr, y_train, cv=cv, scoring="f1_macro")
            row[mname] = round(float(s.mean()), 4)
        ablation_rows.append(row)
    ablation = pd.DataFrame(ablation_rows)
    print(ablation.to_string(index=False))

    # 4b) Pemilihan model: GridSearchCV di train ---------------------------------
    print("\n=== PEMILIHAN MODEL (CV f1_macro, data train) ===")
    searches, cv_rows = {}, []
    for name, (mdl, grid) in candidate_models().items():
        gs = GridSearchCV(make_pipeline(mdl), grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_train, y_train)
        searches[name] = gs
        i = gs.best_index_
        cv_rows.append({
            "model": name,
            "cv_f1_macro_mean": round(float(gs.cv_results_["mean_test_score"][i]), 4),
            "cv_f1_macro_std": round(float(gs.cv_results_["std_test_score"][i]), 4),
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        })
    cv_table = pd.DataFrame(cv_rows).sort_values("cv_f1_macro_mean", ascending=False)
    print(cv_table.to_string(index=False))

    best_name = cv_table.iloc[0]["model"]
    best_pipe = searches[best_name].best_estimator_  # sudah di-refit pada seluruh data train
    print(f"\nModel terpilih (berdasarkan CV): {best_name}")

    # 5) EVALUASI TEST — satu kali, hanya model terpilih ------------------------
    classes = list(best_pipe.classes_)
    pred = best_pipe.predict(X_test)
    proba = best_pipe.predict_proba(X_test)
    rep = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "roc_auc_ovr_macro": float(roc_auc_score(y_test, proba, multi_class="ovr",
                                                 average="macro", labels=classes)),
        "log_loss": float(log_loss(y_test, proba, labels=classes)),
    }
    per_class = {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                 for c in classes}
    cm = confusion_matrix(y_test, pred, labels=classes).tolist()

    print("\n=== EVALUASI DATA TEST (sekali) ===")
    print(json.dumps(test_metrics, indent=2))
    print(classification_report(y_test, pred, labels=classes, zero_division=0))
    print("Confusion matrix (baris=aktual, kolom=prediksi):", classes)
    print(np.array(cm))

    # 6) SIMPAN pipeline + metadata ---------------------------------------------
    metadata = {
        "classes": classes,
        "input_cols": INPUT_COLS,
        "numeric_inputs": {  # rentang dari data TRAIN saja
            c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                "median": float(X_train[c].median())} for c in RAW_NUM},
        "categorical_inputs": {
            "island": sorted(X_train["island"].dropna().unique().tolist()),
            "sex": sorted(X_train["sex"].dropna().unique().tolist()) + ["unknown"],
        },
        "engineered_features": ENGINEERED,
        "dropped_columns": DROPPED,
        "best_model": best_name,
        "best_params": cv_table.iloc[0]["best_params"],
        "cv_table": cv_table.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "per_class": per_class,
        "confusion_matrix": cm,
        "audit": audit_report,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_class_counts": y_train.value_counts().to_dict(),
        "seed": SEED, "test_size": TEST_SIZE,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump({"pipeline": best_pipe, "metadata": metadata}, MODEL_PATH, compress=3)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nTersimpan: {MODEL_PATH.relative_to(BASE_DIR)}  &  models/metadata.json")


if __name__ == "__main__":
    main()
