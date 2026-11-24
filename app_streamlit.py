import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediksi Harga Rumah California",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_artefact():
    return joblib.load("model_california.joblib")

artefact = load_artefact()
pipeline = artefact["pipeline"]
model_version = artefact["version"]
metrics = artefact["metrics"]

st.title("🏠 Prediksi Nilai Median Rumah California")
st.caption(f"Model: Regresi Linear | Versi: {model_version} | MAE: USD {metrics['mae']*100_000:,.0f}")

st.markdown(
    "Aplikasi ini memperkirakan **nilai median rumah pada tingkat kelompok blok sensus** "
    "di California berdasarkan data sensus historis. Masukkan karakteristik wilayah di bawah ini:"
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        med_inc = st.number_input(
            "MedInc (Pendapatan Median, dalam USD 10.000)",
            min_value=0.1, max_value=20.0, value=3.87, step=0.1
        )
        house_age = st.number_input(
            "HouseAge (Usia Median Rumah, tahun)",
            min_value=1.0, max_value=100.0, value=28.0, step=1.0
        )
        ave_rooms = st.number_input(
            "AveRooms (Rata-rata Jumlah Kamar)",
            min_value=1.0, max_value=50.0, value=5.4, step=0.1
        )
        ave_bedrms = st.number_input(
            "AveBedrms (Rata-rata Kamar Tidur)",
            min_value=0.5, max_value=20.0, value=1.1, step=0.1
        )
    with col2:
        population = st.number_input(
            "Population (Populasi Blok Sensus)",
            min_value=1.0, max_value=50000.0, value=1425.0, step=50.0
        )
        ave_occup = st.number_input(
            "AveOccup (Rata-rata Penghuni per Rumah)",
            min_value=0.5, max_value=50.0, value=3.0, step=0.1
        )
        latitude = st.number_input(
            "Latitude (Garis Lintang)",
            min_value=32.0, max_value=42.5, value=35.63, step=0.01
        )
        longitude = st.number_input(
            "Longitude (Garis Bujur)",
            min_value=-125.0, max_value=-114.0, value=-119.57, step=0.01
        )

    submitted = st.form_submit_button("Prediksi Harga")

if submitted:
    if ave_bedrms > ave_rooms:
        st.error("Input tidak valid: Rata-rata kamar tidur (AveBedrms) tidak boleh lebih besar dari rata-rata total kamar (AveRooms).")
    else:
        input_df = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }])

        pred_unit = float(pipeline.predict(input_df)[0])
        pred_usd = pred_unit * 100_000

        if pred_usd < 0:
            st.warning(
                f"Hasil prediksi mentah bernilai negatif ({pred_unit:.2f} unit), "
                "yang menandakan kombinasi input berada jauh di luar pola data latih."
            )
        else:
            st.success(f"**Perkiraan Nilai Median Rumah:** USD {pred_usd:,.2f}")
            st.info(f"Keluaran mentah model: **{pred_unit:.4f}** (1 unit = USD 100.000)")

        with st.expander("Interpretasi & Keterbatasan Model"):
            st.markdown(
                "- **Interpretasi:** Nilai di atas adalah estimasi median harga rumah untuk kelompok blok sensus "
                "dengan karakteristik serupa pada data historis California.\n"
                "- **Keterbatasan:** Dataset berasal dari sensus historis (tahun 1990), sehingga tidak "
                "mencerminkan harga pasar properti saat ini maupun kondisi fisik rumah secara individual. "
                "Hasil ini hanya digunakan sebagai gambaran awal, bukan penilaian properti resmi."
            )