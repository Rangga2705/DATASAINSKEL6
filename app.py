
import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('health_linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Prediksi Health Score')
st.write('Masukkan parameter kesehatan untuk memprediksi Health Score.')

# Sidebar input form
st.sidebar.header('Input Parameter')

def user_input_features():
    BMI = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    Exercise_Frequency = st.sidebar.slider('Frekuensi Olahraga (per minggu)', 0, 14, 3)
    Diet_Quality = st.sidebar.slider('Kualitas Diet (0 - 100)', 0, 100, 50)
    Sleep_Hours = st.sidebar.slider('Jam Tidur per Hari', 0.0, 12.0, 7.0)

    data = {
        'BMI': BMI,
        'Exercise_Frequency': Exercise_Frequency,
        'Diet_Quality': Diet_Quality,
        'Sleep_Hours': Sleep_Hours
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna')
st.write(df_input)

# Make prediction
if st.sidebar.button('Prediksi Health Score'):
    try:
        prediction = model.predict(df_input)
        st.subheader('Hasil Prediksi Health Score:')
        st.write(f"Health Score Diprediksi: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f'Terjadi kesalahan saat membuat prediksi: {e}')

