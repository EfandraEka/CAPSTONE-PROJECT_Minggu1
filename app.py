import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
try:
    model = joblib.load("best_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Kesalahan saat memuat model atau scaler: {e}")
    st.stop()

# Judul aplikasi
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("📊 Prediksi Tingkat Obesitas")
st.write("Masukkan data pribadi dan gaya hidup Anda untuk memprediksi tingkat obesitas.")

with st.form("form_obesitas"):
    st.markdown("### 🧍 Data Pribadi dan Gaya Hidup")

    # Input
    age = st.slider("Usia", 10, 100, 25)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    height_cm = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=65.0)

    calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["yes", "no"])
    fcvc = st.slider("Frekuensi makan sayur (0.0 - 3.0)", 0.0, 3.0, 2.0)
    ncp = st.slider("Jumlah makan utama per hari", 1, 5, 3)
    scc = st.selectbox("Apakah Anda memantau konsumsi kalori?", ["yes", "no"])
    smoke = st.selectbox("Apakah Anda merokok?", ["yes", "no"])
    ch2o = st.slider("Konsumsi air per hari (liter)", 0.0, 3.0, 2.0)
    family_history = st.selectbox("Riwayat keluarga obesitas", ["yes", "no"])
    faf = st.slider("Aktivitas fisik mingguan (0.0 - 3.0)", 0.0, 3.0, 1.0)
    tue = st.slider("Penggunaan perangkat teknologi (jam/hari)", 0.0, 4.0, 2.0)
    caec = st.selectbox("Ngemil / fast food", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submit = st.form_submit_button("Prediksi")

if submit:
    # === Konversi dan encoding ===
    height = height_cm / 100  # Konversi ke meter
    bmi = weight / (height ** 2)

    gender = 1 if gender == "Male" else 0
    family_history = 1 if family_history == "yes" else 0
    favc = 1 if favc == "yes" else 0
    scc = 1 if scc == "yes" else 0
    smoke = 1 if smoke == "yes" else 0

    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    mtrans_map = {
        "Public_Transportation": 0,
        "Walking": 1,
        "Automobile": 2,
        "Motorbike": 3,
        "Bike": 4
    }

    calc = calc_map[calc]
    caec = caec_map[caec]
    mtrans = mtrans_map[mtrans]

    # === Buat vektor fitur dengan BMI ditambahkan ===
    data = np.array([[age, gender, height, weight, bmi,
                      calc, favc, fcvc, ncp,
                      scc, smoke, ch2o, family_history,
                      faf, tue, caec, mtrans]])

    # Scaling dan prediksi
    try:
        scaled = scaler.transform(data)
        pred = model.predict(scaled)[0]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.stop()

    # Label dan penjelasan
    label = {
        0: "Insufficient Weight",
        1: "Normal Weight",
        2: "Overweight Level I",
        3: "Overweight Level II",
        4: "Obesity Type I",
        5: "Obesity Type II"
    }

    explanation = {
        "Insufficient Weight": "Berat badan Anda berada di bawah normal.",
        "Normal Weight": "Berat badan Anda ideal. Pertahankan gaya hidup sehat!",
        "Overweight Level I": "Sedikit kelebihan berat badan. Perhatikan pola makan dan aktivitas fisik.",
        "Overweight Level II": "Kelebihan berat badan. Disarankan untuk meningkatkan aktivitas fisik dan mengatur pola makan.",
        "Obesity Type I": "Tingkat obesitas. Konsultasikan dengan ahli gizi atau dokter.",
        "Obesity Type II": "Obesitas tingkat lanjut. Perlu penanganan medis serius."
    }

    st.markdown("---")
    st.metric(label="BMI Anda", value=f"{bmi:.2f}")
    st.success(f"Hasil Prediksi: *{label[pred]}*")
    st.info(explanation[label[pred]])
