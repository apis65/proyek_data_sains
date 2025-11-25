import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Memuat model dan transformer --- #
try:
    with open('gradient_boosting_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('power_transformer.pkl', 'rb') as file:
        pt = pickle.load(file)
    with open('standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("File model atau transformer tidak ditemukan. Pastikan semua file .pkl berada dalam folder yang sama dengan app.py.")
    st.stop()

# --- Mapping kategori sesuai data training --- #
# Mapping ini digunakan untuk mengubah input kategori dari Streamlit ke nilai numerik
# sesuai dengan data yang digunakan saat pelatihan model.
fuel_mapping = {
    'Diesel': 0,
    'Petrol': 1,
    'CNG': 2,
    'LPG': 3,
    'Electric': 4
}
seller_type_mapping = {
    'Individual': 0,
    'Dealer': 1,
    'Trustmark Dealer': 2
}
transmission_mapping = {
    'Manual': 0,
    'Automatic': 1
}
owner_mapping = {
    'Test Drive Car': 0,
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
}

# --- Tampilan Aplikasi Streamlit --- #
st.title('Prediksi Harga Mobil Bekas')
st.write('Aplikasi ini memprediksi harga jual mobil bekas berdasarkan beberapa parameter.')

st.sidebar.header('Input Detail Mobil')

# --- Input dari pengguna --- #
year_input = st.sidebar.number_input('Tahun Mobil', min_value=1990, max_value=2024, value=2015, step=1)
km_driven_input = st.sidebar.number_input('Jarak Tempuh (km)', min_value=0, max_value=1000000, value=50000, step=1000)
fuel_type_input = st.sidebar.selectbox('Jenis Bahan Bakar', list(fuel_mapping.keys()))
seller_type_input = st.sidebar.selectbox('Tipe Penjual', list(seller_type_mapping.keys()))
transmission_type_input = st.sidebar.selectbox('Transmisi', list(transmission_mapping.keys()))
owner_status_input = st.sidebar.selectbox('Jumlah Pemilik Sebelumnya', list(owner_mapping.keys()))

# --- Logika Prediksi --- #
if st.sidebar.button('Prediksi Harga Mobil'):
    st.subheader('Detail Input Anda:')
    input_data = {
        'Tahun': year_input,
        'Jarak Tempuh (km)': km_driven_input,
        'Jenis Bahan Bakar': fuel_type_input,
        'Tipe Penjual': seller_type_input,
        'Transmisi': transmission_type_input,
        'Jumlah Pemilik': owner_status_input
    }
    st.write(pd.DataFrame([input_data]))

    # Mengonversi input kategori menjadi angka sesuai mapping
    age = 2025 - year_input
    fuel_encoded = fuel_mapping[fuel_type_input]
    seller_type_encoded = seller_type_mapping[seller_type_input]
    transmission_encoded = transmission_mapping[transmission_type_input]
    owner_encoded = owner_mapping[owner_status_input]

    # --- Transformasi PowerTransformer (selling_price_yj dan km_driven_yj) --- #
    # PowerTransformer dilatih pada dua kolom: 'selling_price' dan 'km_driven'.
    # Untuk mentransformasikan hanya 'km_driven', kita mengisi kolom pertama (selling_price)
    # dengan nilai dummy (0), karena tidak memengaruhi hasil transformasi km_driven.
    data_for_pt = np.array([[0, km_driven_input]])  # 0 sebagai dummy untuk selling_price
    transformed_data_for_pt = pt.transform(data_for_pt)
    km_driven_yj = transformed_data_for_pt[0, 1]  # mengambil nilai km_driven hasil transformasi

    # --- Transformasi StandardScaler (km_driven_yj dan age) --- #
    # StandardScaler dilatih menggunakan dua kolom: ['km_driven_yj', 'age']
    # Maka kita membuat DataFrame dengan nilai tersebut.
    data_for_scaler = pd.DataFrame([[km_driven_yj, age]], columns=['km_driven_yj', 'age'])
    scaled_features = scaler.transform(data_for_scaler)

    # Menyusun DataFrame untuk prediksi, sesuai urutan kolom pada X_train
    # Urutan kolom: ['fuel', 'seller_type', 'transmission', 'owner', 'km_driven_yj', 'age']
    prediction_df = pd.DataFrame([[fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded,
                                   scaled_features[0, 0], scaled_features[0, 1]]],
                                 columns=['fuel', 'seller_type', 'transmission', 'owner', 'km_driven_yj', 'age'])

    # Model memprediksi nilai selling_price_yj (hasil transformasi Yeo-Johnson)
    predicted_price_yj = model.predict(prediction_df)[0]

    # --- Inverse PowerTransformer untuk mengembalikan prediksi ke skala asli --- #
    # Untuk inverse transform, kita tetap memasukkan dua kolom, sehingga kolom kedua (km_driven_yj)
    # diisi nilai dummy karena tidak mempengaruhi hasil inverse selling_price.
    data_for_inverse_pt = np.array([[predicted_price_yj, 0]])  # 0 sebagai dummy
    original_scale_prediction = pt.inverse_transform(data_for_inverse_pt)
    final_predicted_selling_price = original_scale_prediction[0, 0]

    st.subheader('Hasil Prediksi Harga:')
    st.success(f'Harga Mobil Diprediksi: Rp {final_predicted_selling_price:,.2f}')

    st.markdown("""
    **Catatan Penting:**
    * Prediksi ini merupakan estimasi dan tidak sepenuhnya akurat.
    * Kondisi mobil, lokasi penjualan, dan fitur tambahan dapat mempengaruhi harga sebenarnya.
    * Model dilatih berdasarkan data yang tersedia sehingga kualitas prediksi bergantung pada representasi data tersebut.
    """)
