from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import sklearn
import streamlit as st

from features import add_features  # noqa: F401  (diperlukan joblib untuk memuat pipeline)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "penguin_pipeline.joblib"

LABELS = {
    "bill_length_mm": "Panjang paruh (mm)",
    "bill_depth_mm": "Kedalaman paruh (mm)",
    "flipper_length_mm": "Panjang sirip (mm)",
    "body_mass_g": "Massa tubuh (gram)",
}
SEX_LABELS = {"male": "Jantan", "female": "Betina", "unknown": "Tidak diketahui"}

st.set_page_config(page_title="Klasifikasi Spesies Penguin", page_icon="🐧", layout="wide")


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
st.title("🐧 Klasifikasi Spesies Penguin")

if bundle is None:
    st.error("Model belum ditemukan di `models/penguin_pipeline.joblib`. "
             "Jalankan `python train.py` terlebih dahulu.")
    st.stop()

pipe, meta = bundle["pipeline"], bundle["metadata"]
classes = meta["classes"]

if meta["sklearn_version"] != sklearn.__version__:
    st.warning(f"Model dilatih dengan scikit-learn {meta['sklearn_version']}, sedangkan environment "
               f"ini memakai {sklearn.__version__}. Jika muncul error/hasil aneh, jalankan ulang `train.py`.")

tab_pred, tab_eval, tab_info = st.tabs(["🔮 Prediksi", "📊 Evaluasi model", "🧾 Data & fitur"])

# ------------------------------------------------------------------ TAB PREDIKSI
with tab_pred:
    st.caption("Masukkan data mentah hasil pengamatan. Imputasi, scaling, encoding, dan fitur rasio "
               "diproses otomatis di dalam pipeline.")
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            island = st.selectbox("Pulau pengamatan", meta["categorical_inputs"]["island"],
                                  format_func=str.title)
            sex = st.selectbox("Jenis kelamin", meta["categorical_inputs"]["sex"],
                               format_func=lambda v: SEX_LABELS.get(v, v),
                               index=meta["categorical_inputs"]["sex"].index("unknown"))
        values = {}
        with c2:
            for col, info in meta["numeric_inputs"].items():
                values[col] = st.number_input(LABELS[col], min_value=0.0, value=round(info["median"], 1),
                                              step=0.1 if col != "body_mass_g" else 10.0)
        submitted = st.form_submit_button("Prediksi spesies", type="primary")

    if submitted:
        row = pd.DataFrame([{"island": island, "sex": sex, **values}])[meta["input_cols"]]
        proba = pipe.predict_proba(row)[0]
        order = proba.argsort()[::-1]
        top, top_p = classes[order[0]], proba[order[0]]

        # peringatan jika input di luar rentang data latih
        out_of_range = [LABELS[c] for c, i in meta["numeric_inputs"].items()
                        if not (i["min"] <= values[c] <= i["max"])]
        if out_of_range:
            st.warning("Nilai di luar rentang data latih untuk: " + ", ".join(out_of_range)
                       + ". Prediksi kurang dapat diandalkan.")

        st.subheader(f"Prediksi: **{top}**")
        st.metric("Probabilitas tertinggi", f"{top_p:.1%}")
        if top_p < 0.60:
            st.info("Model kurang yakin (probabilitas tertinggi < 60%). Periksa kembali input.")

        prob_df = pd.DataFrame({"Spesies": classes, "Probabilitas": proba}).sort_values(
            "Probabilitas", ascending=False)
        left, right = st.columns(2)
        with left:
            st.dataframe(prob_df.style.format({"Probabilitas": "{:.2%}"}), hide_index=True,
                         width="stretch")
        with right:
            st.bar_chart(prob_df.set_index("Spesies"), y="Probabilitas")

        with st.expander("Fitur turunan yang dihitung otomatis"):
            derived = add_features(row)[meta["engineered_features"]]
            st.dataframe(derived.round(3), hide_index=True)

# --------------------------------------------------------------------- TAB EVALUASI
with tab_eval:
    tm = meta["test_metrics"]
    st.caption(f"Model terpilih: **{meta['best_model']}** · dilatih pada {meta['n_train']} data, "
               f"dievaluasi satu kali pada {meta['n_test']} data test (stratified split, seed {meta['seed']}).")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{tm['accuracy']:.3f}")
    m[1].metric("Balanced accuracy", f"{tm['balanced_accuracy']:.3f}")
    m[2].metric("F1 macro", f"{tm['f1_macro']:.3f}")
    m[3].metric("ROC-AUC (OvR)", f"{tm['roc_auc_ovr_macro']:.3f}")
    m[4].metric("Log loss", f"{tm['log_loss']:.3f}")

    st.warning(f"Data test hanya {meta['n_test']} baris, sehingga skor sempurna di test tidak menjamin "
               "performa sempurna pada data baru. Gunakan hasil cross-validation (tabel di bawah) "
               "sebagai gambaran yang lebih stabil.")

    st.subheader("Per kelas (data test)")
    pc = pd.DataFrame(meta["per_class"]).T.rename(
        columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Jumlah data"})
    pc["Jumlah data"] = pc["Jumlah data"].astype(int)
    st.dataframe(pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                 width="stretch")

    st.subheader("Confusion matrix (data test)")
    cm = pd.DataFrame(meta["confusion_matrix"], index=classes, columns=classes)
    long = cm.reset_index(names="Aktual").melt("Aktual", var_name="Prediksi", value_name="Jumlah")
    base = alt.Chart(long).encode(x=alt.X("Prediksi:N", sort=classes), y=alt.Y("Aktual:N", sort=classes))
    heat = base.mark_rect().encode(color=alt.Color("Jumlah:Q", legend=None))
    txt = base.mark_text(fontSize=18).encode(
        text="Jumlah:Q",
        color=alt.condition(alt.datum.Jumlah > long["Jumlah"].max() / 2, alt.value("white"), alt.value("black")))
    st.altair_chart((heat + txt).properties(height=280), width="stretch")

    st.subheader("Perbandingan model (cross-validation di data train)")
    st.caption("5-fold × 3 pengulangan, skor F1 macro. Dipakai untuk memilih model; data test tidak dipakai.")
    cvt = pd.DataFrame(meta["cv_table"])
    cvt["best_params"] = cvt["best_params"].astype(str)
    st.dataframe(cvt.rename(columns={"model": "Model", "cv_f1_macro_mean": "F1 macro (rata-rata)",
                                     "cv_f1_macro_std": "Std", "best_params": "Hiperparameter terbaik"}),
                 hide_index=True, width="stretch")

    with st.expander("Apa arti metrik-metrik ini?"):
        st.markdown(
            "- **Accuracy**: proporsi seluruh prediksi yang benar. Bisa menyesatkan jika kelas tidak seimbang "
            "(Chinstrap hanya ±20% data).\n"
            "- **Balanced accuracy**: rata-rata recall tiap kelas, sehingga kelas kecil punya bobot setara.\n"
            "- **Precision**: dari semua yang diprediksi spesies X, berapa persen benar-benar X.\n"
            "- **Recall**: dari semua spesies X yang sebenarnya, berapa persen berhasil terdeteksi.\n"
            "- **F1 (macro)**: rata-rata harmonik precision & recall, dirata-ratakan sama rata antar kelas. "
            "Ini metrik utama untuk memilih model.\n"
            "- **ROC-AUC (OvR)**: kemampuan probabilitas memisahkan satu kelas dari sisanya; 1.0 = pemisahan sempurna.\n"
            "- **Log loss**: kualitas probabilitas; makin kecil makin baik, dan menghukum prediksi salah yang terlalu percaya diri.\n"
            "- **Confusion matrix**: baris = spesies sebenarnya, kolom = prediksi; angka di luar diagonal adalah kesalahan.")

# ------------------------------------------------------------------------ TAB INFO
with tab_info:
    a = meta["audit"]
    st.subheader("Audit data")
    st.markdown(
        f"- Jumlah baris awal: **{a['n_rows']}**; duplikat: **{a['duplicate_rows']}**\n"
        f"- Missing value: `{a['missing']}`\n"
        f"- Baris dengan seluruh pengukuran kosong: **{a['rows_all_measurements_missing']}** (dibuang, tidak ada "
        "informasi yang bisa dipakai atau diimputasi secara bermakna)\n"
        f"- Nilai tidak wajar (di luar batas biologis): `{a['implausible_values']}`\n"
        f"- Distribusi kelas: `{a['class_counts']}`")
    st.subheader("Keputusan fitur")
    st.markdown(
        "- **Fitur dipakai**: island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g\n"
        "- **Fitur baru**: `bill_ratio` (panjang/kedalaman paruh) dan `flipper_mass_ratio` (sirip per kg massa)\n"
        "- **sex kosong** → kategori `unknown` (tidak memalsukan jenis kelamin); **numerik kosong** → median dari data train\n"
        "- **Scaling** (StandardScaler) untuk numerik; **One-hot** untuk island & sex")
    for col, reason in meta["dropped_columns"].items():
        st.markdown(f"- **Dibuang: `{col}`** — {reason}")
    st.subheader("Ablasi fitur (CV di data train, F1 macro)")
    st.dataframe(pd.DataFrame(meta["ablation"]).rename(columns={"feature_set": "Set fitur"}),
                 hide_index=True, width="stretch")
    st.caption("Catatan: `island` sangat informatif (mis. Torgersen hanya berisi Adelie di data ini), tetapi "
               "membuat model bergantung pada lokasi pengamatan; spesies di pulau baru bisa tidak terprediksi baik.")
