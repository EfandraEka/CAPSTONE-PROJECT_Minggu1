import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title(" Prediksi Kategori Obesitas")

# Load model dan scaler
model_data = joblib.load("best_rf_model_clean.pkl")
model = model_data["model"]
scaler = model_data["scaler"]

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

# Mapping hasil model ke label klasifikasi
label_mapping = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

# Kesimpulan sederhananya
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
        label_maps = {
            "Gender": {"Female": 0, "Male": 1},
            "FAVC": {"no": 0, "yes": 1},
            "SCC": {"no": 0, "yes": 1},
            "SMOKE": {"no": 0, "yes": 1},
            "family_history_with_overweight": {"no": 0, "yes": 1},
            "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
            "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
            "MTRANS": {
                "Public_Transportation": 3,
                "Walking": 4,
                "Automobile": 0,
                "Motorbike": 2,
                "Bike": 1
            }
        }

        input_df = pd.DataFrame([user_input])
        for col, mapping in label_maps.items():
            input_df[col] = input_df[col].map(mapping)

        # Urutkan fitur agar sesuai dengan pelatihan
        feature_order = [
            "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
            "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC",
            "FAF", "TUE", "CALC", "MTRANS"
        ]
        input_df = input_df[feature_order]

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Prediksi
        prediction = model.predict(input_scaled)[0]
        label_penuh = label_mapping.get(prediction, "Tidak diketahui")
        label_sederhana = kategori_sederhana.get(label_penuh, "Tidak diketahui")

        st.subheader("Hasil Prediksi:")
        st.info(f"Kategori: **{label_penuh.replace('_', ' ')}**")
        st.success(f"Kesimpulan: **{label_sederhana}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
