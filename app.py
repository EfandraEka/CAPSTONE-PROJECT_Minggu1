import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Kategori Obesitas")

# Load model pipeline
model = joblib.load("best_rf_model_clean.pkl")

# Input form
with st.form("obesity_form"):
    st.subheader("Masukkan Data Anda")

    user_input = {
        "Age": st.slider("Usia", 10, 100, 25),
        "Gender": st.selectbox("Jenis Kelamin", ["Female", "Male"]),
        "Height": st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.70),
        "Weight": st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0),
        "CALC": st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"]),
        "FAVC": st.selectbox("Makanan Tinggi Kalori?", ["yes", "no"]),
        "FCVC": st.slider("Frekuensi Konsumsi Sayur (1-3)", 1, 3, 2),
        "NCP": st.slider("Jumlah Makan per Hari (1-4)", 1, 4, 3),
        "SCC": st.selectbox("Monitoring Kalori?", ["yes", "no"]),
        "SMOKE": st.selectbox("Merokok?", ["yes", "no"]),
        "CH2O": st.slider("Konsumsi Air (liter/hari)", 1, 3, 2),
        "family_history_with_overweight": st.selectbox("Riwayat Obesitas Keluarga?", ["yes", "no"]),
        "FAF": st.slider("Aktivitas Fisik (jam/minggu)", 0, 10, 2),
        "TUE": st.slider("Waktu Hiburan/Layar (jam/hari)", 0, 5, 1),
        "CAEC": st.selectbox("Kebiasaan Camilan", ["no", "Sometimes", "Frequently", "Always"]),
        "MTRANS": st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
    }

    submitted = st.form_submit_button("Prediksi")

# Kategori sederhana
kategori_sederhana = {
    "Insufficient_Weight": "Tidak Obesitas",
    "Normal_Weight": "Tidak Obesitas",
    "Overweight_Level_I": "Cenderung Obesitas",
    "Overweight_Level_II": "Cenderung Obesitas",
    "Obesity_Type_I": "Obesitas",
    "Obesity_Type_II": "Obesitas",
    "Obesity_Type_III": "Obesitas"
}

if submitted:
    try:
        # DataFrame dari input tanpa encoding manual
        input_df = pd.DataFrame([user_input])

        # Prediksi langsung (pipeline handle preprocessing)
        prediction = model.predict(input_df)[0]

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.info(f"Kategori: **{prediction.replace('_', ' ')}**")
        st.success(f"Kesimpulan: **{kategori_sederhana.get(prediction, 'Tidak diketahui')}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
