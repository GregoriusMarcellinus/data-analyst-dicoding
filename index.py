import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Pengaturan halaman
st.set_page_config(page_title="Analisis Kualitas Udara Stasiun Guanyuan oleh Gregorius")

# Memuat dataset
try:
    data = pd.read_csv('data/PRSA_Data_Guanyuan_20130301-20170228.csv')
except FileNotFoundError:
    st.error("File data tidak ditemukan. Silakan periksa jalur file.")
    st.stop()

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara: Stasiun Guanyuan')

# Deskripsi
st.write('Dashboard ini menyediakan cara interaktif untuk menjelajahi data kualitas udara dan hubungannya dengan berbagai kondisi cuaca.')

# Tentang pembuat
st.markdown("""
### Tentang Saya
- **Nama**: Gregorius Marcellinus Ongkosianbhadra
- **Alamat Email**: marcellongkosianbhadra@gmail.com
- **ID Dicoding**: gregorius1414""")

### Gambaran Proyek
html_gambaran_proyek = """<div style="text-align: justify;"> <h3>Gambaran Proyek</h3>
Proyek ini bertujuan untuk menganalisis data kualitas udara dari Stasiun Guanyuan guna mengidentifikasi tren musiman dan tahunan serta memahami hubungan antara level PM2.5 dan kondisi cuaca. Melalui visualisasi data, analisis korelasi, dan dekomposisi deret waktu, proyek ini mengungkap pola dan faktor-faktor yang mempengaruhi kualitas udara. Hasilnya menunjukkan bahwa kualitas udara di Guanyuan tidak memenuhi standar kesehatan WHO, dengan level PM2.5 melebihi batas tahunan dan harian yang direkomendasikan. Proyek ini memberikan wawasan untuk implementasi kebijakan pengurangan emisi, pemantauan berkelanjutan, dan peningkatan kesadaran masyarakat untuk memperbaiki kualitas udara dan kesehatan publik.
</div>
"""
st.markdown(html_gambaran_proyek, unsafe_allow_html=True)

# Sidebar untuk input pengguna
st.sidebar.header('Fitur Filter Pengguna')

# Pilihan tahun dan bulan
selected_year = st.sidebar.selectbox('Pilih Tahun', list(data['year'].unique()))
selected_month = st.sidebar.selectbox('Pilih Bulan', list(data['month'].unique()))

# Menyaring data berdasarkan input pengguna
data_filtered = data[(data['year'] == selected_year) & (data['month'] == selected_month)].copy()

# Tampilan statistik data
st.subheader('Gambaran Data untuk Periode yang Dipilih')
st.write(data_filtered.describe())

# Grafik garis PM2.5 harian
st.subheader('Level PM2.5 Harian')
fig, ax = plt.subplots()
ax.plot(data_filtered['day'], data_filtered['PM2.5'])
plt.xlabel('Hari dalam Bulan')
plt.ylabel('Konsentrasi PM2.5')
st.pyplot(fig)

# Heatmap korelasi
st.subheader('Heatmap Korelasi Indikator Kualitas Udara')
corr = data_filtered[['PM2.5', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
plt.title('Heatmap Korelasi')
st.pyplot(fig)

# Analisis tren musiman
st.subheader('Analisis Tren Musiman')
seasonal_trends = data.groupby('month')['PM2.5'].mean()
fig, ax = plt.subplots()
seasonal_trends.plot(kind='bar', color='skyblue', ax=ax)
plt.title('Rata-rata Level PM2.5 Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Rata-rata PM2.5')
st.pyplot(fig)

# Perbandingan dengan standar kualitas udara
st.subheader('Perbandingan dengan Standar Kualitas Udara')
who_annual_standard = 10
who_24hr_standard = 25

# Menghitung rata-rata tahunan dan harian
annual_avg_pm25 = data.groupby('year')['PM2.5'].mean().mean()
daily_avg_pm25 = data['PM2.5'].mean()

st.write(f"Rata-rata tahunan PM2.5: {annual_avg_pm25:.2f} µg/m³")
st.write(f"Standar tahunan WHO: {who_annual_standard} µg/m³")
st.write(f"Rata-rata harian PM2.5: {daily_avg_pm25:.2f} µg/m³")
st.write(f"Standar 24 jam WHO: {who_24hr_standard} µg/m³")

if annual_avg_pm25 > who_annual_standard:
    st.write("Kualitas udara melebihi standar tahunan WHO.")
else:
    st.write("Kualitas udara sesuai dengan standar tahunan WHO.")

if daily_avg_pm25 > who_24hr_standard:
    st.write("Kualitas udara melebihi standar 24 jam WHO.")
else:
    st.write("Kualitas udara sesuai dengan standar 24 jam WHO.")

# Dekomposisi deret waktu
st.subheader('Dekomposisi Deret Waktu PM2.5')
try:
    data_filtered['PM2.5'].ffill(inplace=True)
    decomposed = seasonal_decompose(data_filtered['PM2.5'], model='additive', period=24)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    decomposed.trend.plot(ax=ax1, title='Tren')
    decomposed.seasonal.plot(ax=ax2, title='Musiman')
    decomposed.resid.plot(ax=ax3, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)
except ValueError as e:
    st.error("Tidak dapat melakukan dekomposisi deret waktu: " + str(e))

# Rata-rata Jam Heatmap
st.subheader('Rata-rata Jam per Jam PM2.5')
try:
    # Pastikan tipe data benar dan tangani nilai yang hilang
    data['hour'] = data['hour'].astype(int)
    data['PM2.5'] = pd.to_numeric(data['PM2.5'], errors='coerce')
    data['PM2.5'].ffill(inplace=True)

    # Hitung rata-rata per jam
    hourly_avg = data.groupby('hour')['PM2.5'].mean()

    # Plotting
    fig, ax = plt.subplots()
    sns.heatmap([hourly_avg.values], ax=ax, cmap='coolwarm')
    plt.title('Rata-rata Jam per Jam PM2.5')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Kesalahan dalam plotting rata-rata jam per jam: {e}")

# Analisis Arah Angin
st.subheader('Analisis Arah Angin')
wind_data = data_filtered.groupby('wd')['PM2.5'].mean()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
theta = np.linspace(0, 2 * np.pi, len(wind_data))
bars = ax.bar(theta, wind_data.values, align='center', alpha=0.5)
plt.title('Level PM2.5 Berdasarkan Arah Angin')
st.pyplot(fig)

# Curah Hujan vs. Kualitas Udara
st.subheader('Curah Hujan vs. Level PM2.5')
fig, ax = plt.subplots()
sns.scatterplot(x='RAIN', y='PM2.5', data=data_filtered, ax=ax)
plt.title('Curah Hujan vs. Level PM2.5')
st.pyplot(fig)

# Heatmap Korelasi - Interaktif
st.subheader('Heatmap Korelasi Interaktif')
selected_columns = st.multiselect('Pilih Kolom untuk Korelasi', data.columns, default=['PM2.5', 'NO2', 'TEMP', 'PRES', 'DEWP'])
corr = data[selected_columns].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
st.pyplot(fig)


# Kesimpulan
st.write("""
- Dasbor menyediakan analisis mendalam dan interaktif mengenai data kualitas udara.
- Berbagai visualisasi menawarkan wawasan mengenai tingkat PM2.5, distribusinya, dan faktor-faktor yang memengaruhinya.
- Tren musiman dan dampak berbagai kondisi cuaca dan polutan terhadap kualitas udara digambarkan dengan jelas.
- Pengguna dapat menjelajahi data secara dinamis untuk memperoleh pemahaman yang lebih mendalam mengenai tren kualitas udara.
""")
