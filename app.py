import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title(" Prediksi Kategori Obesitas")

# Cek model
if not os.path.exists("best_rf_model.pkl"):
    st.error(" File model 'best_rf_model.pkl' tidak ditemukan.")
    st.stop()

# Load model (bukan dictionary!)
model = joblib.load("best_rf_model.pkl")

# Input pengguna (tanpa encoder)
st.header("Masukkan Data:")
input_data = {
    "Age": st.slider("Usia", 10, 100, 25),
    "Height": st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.70),
    "Weight": st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0),
    "FCVC": st.slider("Frekuensi Konsumsi Sayur (1-3)", 1, 3, 2),
    "NCP": st.slider("Jumlah Makan per Hari (1-4)", 1, 4, 3),
    "CH2O": st.slider("Konsumsi Air (liter/hari)", 1, 3, 2),
    "FAF": st.slider("Aktivitas Fisik (jam/minggu)", 0, 10, 2),
    "TUE": st.slider("Waktu Hiburan/Layar (jam/hari)", 0, 5, 1)
}

input_df = pd.DataFrame([input_data])

try:
    prediction = model.predict(input_df)
    st.subheader(" Hasil Prediksi:")
    st.success(f"Prediksi Status Obesitas: **{prediction[0]}**")
except Exception as e:
    st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
