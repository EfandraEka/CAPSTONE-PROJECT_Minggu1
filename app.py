import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title(" Prediksi Kategori Obesitas Berdasarkan Gaya Hidup")

# Load pipeline model
pipeline = joblib.load("model.pkl")
model = pipeline["model"]
encoders = pipeline["encoders"]
target_encoder = pipeline["target_encoder"]
features = pipeline["feature_names"]

# Input dari pengguna
user_input = {
    "Age": st.slider("Usia", 10, 100, 25),
    "Gender": st.selectbox("Jenis Kelamin", encoders["Gender"].classes_),
    "Height": st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.70),
    "Weight": st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0),
    "CALC": st.selectbox("Konsumsi Alkohol", encoders["CALC"].classes_),
    "FAVC": st.selectbox("Makanan Tinggi Kalori?", encoders["FAVC"].classes_),
    "FCVC": st.slider("Frekuensi Konsumsi Sayur (1-3)", 1, 3, 2),
    "NCP": st.slider("Jumlah Makan per Hari (1-4)", 1, 4, 3),
    "SCC": st.selectbox("Monitoring Kalori?", encoders["SCC"].classes_),
    "SMOKE": st.selectbox("Merokok?", encoders["SMOKE"].classes_),
    "CH2O": st.slider("Konsumsi Air (liter/hari)", 1, 3, 2),
    "family_history_with_overweight": st.selectbox("Riwayat Obesitas di Keluarga?", encoders["family_history_with_overweight"].classes_),
    "FAF": st.slider("Aktivitas Fisik (jam/minggu)", 0, 10, 2),
    "TUE": st.slider("Waktu Hiburan/Layar (jam/hari)", 0, 5, 1),
    "CAEC": st.selectbox("Kebiasaan Camilan", encoders["CAEC"].classes_),
    "MTRANS": st.selectbox("Transportasi", encoders["MTRANS"].classes_)
}

# Encode input
input_df = pd.DataFrame([user_input])
for col in encoders:
    input_df[col] = encoders[col].transform(input_df[col])

# Prediksi
prediction = model.predict(input_df[features])
label = target_encoder.inverse_transform(prediction)[0]

st.subheader(" Hasil Prediksi:")
st.success(f"Status Obesitas Anda: **{label}**")
