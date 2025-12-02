
import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('health_linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Prediksi Health Score')
st.write('Masukkan parameter kesehatan Anda untuk memprediksi Health Score.')

# ----- USER INPUT (NUMBER INPUT, BUKAN SLIDER) -----

bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, step=0.1)
exercise = st.number_input('Frekuensi Olahraga (kali/minggu)', min_value=0, max_value=14, step=1)
diet = st.number_input('Diet Quality (0-100)', min_value=0, max_value=100, step=1)
sleep = st.number_input('Jam Tidur per Hari', min_value=0.0, max_value=15.0, step=0.1)

# Prepare data for prediction
input_data = pd.DataFrame({
    'BMI': [bmi],
    'Exercise_Frequency': [exercise],
    'Diet_Quality': [diet],
    'Sleep_Hours': [sleep]
})

st.subheader('Data yang Anda Masukkan:')
st.write(input_data)

if st.button('Prediksi Health Score'):
    try:
        prediction = model.predict(input_data)
        st.subheader('Hasil Prediksi:')
        st.write(f"Health Score Anda diprediksi: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
