### Nama: Taufik Ismail (24552011355)

### Kelas: TIF RM 23A

# Klasifikasi Spesies Penguin

```
penguin_app/
├── app.py                          # aplikasi Streamlit
├── train.py                        # audit -> split -> CV -> evaluasi test (1x) -> simpan
├── features.py                     # definisi kolom + fitur rasio (dipakai train.py & app.py)
├── requirements.txt
├── data/kasus03_penguins.csv
├── models/penguin_pipeline.joblib  # {"pipeline": ..., "metadata": ...}
├── models/metadata.json            # salinan metadata yang mudah dibaca
└── penguin_streamlit_colab.ipynb   # training + menjalankan Streamlit di Colab
```

Lokal:

```
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Colab: buka `penguin_streamlit_colab.ipynb`, jalankan semua sel berurutan, upload CSV saat diminta.

Catatan: `features.py` harus ada di folder yang sama dengan `app.py` karena pipeline memakai `add_features`.
Model dilatih dengan scikit-learn yang versinya dicatat di metadata; jika versi berbeda, jalankan ulang `train.py`.
