import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")

st.title(" Prediksi Kategori Obesitas Berdasarkan Gaya Hidup")

# Cek apakah file model tersedia
if not os.path.exists("best_rf_model.pkl"):
    st.error(" File model 'best_rf_model.pkl' tidak ditemukan. Pastikan sudah mengunggah file tersebut.")
    st.stop()

# Load model dan encoder
pipeline = joblib.load("best_rf_model.pkl")

try:
    model = pipeline["model"]
    encoders = pipeline["encoders"]
    target_encoder = pipeline["target_encoder"]
    features = pipeline["feature_names"]
except KeyError:
    st.error(" Format file model.pkl tidak sesuai. Pastikan file berisi key: 'model', 'encoders', 'target_encoder', dan 'feature_names'.")
    st.stop()

# Form Input Pengguna
st.header("Masukkan Data Anda:")
user_input = {
    "Age": st.slider("Usia", 10, 100, 25),
    "Gender": st.selectbox("Jenis Kelamin", encoders["Gender"].classes_),
    "Height": st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01),
    "Weight": st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1),
    "CALC": st.selectbox("Konsumsi Alkohol", encoders["CALC"].classes_),
    "FAVC": st.selectbox("Mengonsumsi Makanan Tinggi Kalori?", encoders["FAVC"].classes_),
    "FCVC": st.slider("Frekuensi Konsumsi Sayur (1-3)", 1, 3, 2),
    "NCP": st.slider("Jumlah Makan per Hari (1-4)", 1, 4, 3),
    "SCC": st.selectbox("Monitoring Kalori?", encoders["SCC"].classes_),
    "SMOKE": st.selectbox("Merokok?", encoders["SMOKE"].classes_),
    "CH2O": st.slider("Konsumsi Air (liter per hari)", 1, 3, 2),
    "family_history_with_overweight": st.selectbox("Riwayat Obesitas Keluarga?", encoders["family_history_with_overweight"].classes_),
    "FAF": st.slider("Aktivitas Fisik (jam/minggu)", 0, 10, 2),
    "TUE": st.slider("Waktu Layar/Hiburan (jam/hari)", 0, 5, 1),
    "CAEC": st.selectbox("Kebiasaan Camilan", encoders["CAEC"].classes_),
    "MTRANS": st.selectbox("Moda Transportasi", encoders["MTRANS"].classes_)
}

# Encode input
input_df = pd.DataFrame([user_input])
for col in encoders:
    input_df[col] = encoders[col].transform(input_df[col])

# Prediksi
try:
    prediction = model.predict(input_df[features])
    label = target_encoder.inverse_transform(prediction)[0]

    st.subheader(" Hasil Prediksi:")
    st.success(f"Status Obesitas Anda: **{label}**")
except Exception as e:
    st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
