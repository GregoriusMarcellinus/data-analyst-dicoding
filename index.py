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
- **ID Dicoding**: gregorius1414

### Gambaran Proyek
Dashboard ini menyajikan analisis data kualitas udara, terutama berfokus pada level PM2.5, dari stasiun Guanyuan. Proyek ini bertujuan untuk mengungkap tren, variasi musiman, dan dampak berbagai kondisi cuaca terhadap kualitas udara. Wawasan dari analisis ini dapat bermanfaat untuk studi lingkungan dan pemantauan kesehatan masyarakat.
""")

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

# Kesimpulan
st.subheader('Kesimpulan')
st.write("""
### 1. **Bagaimana kualitas udara saat ini dibandingkan dengan standar kualitas udara yang ditetapkan oleh pemerintah, dan apakah terdapat tren musiman dalam data kualitas udara yang menunjukkan perubahan signifikan dalam periode tertentu?**

**Kesimpulan:**
- **Kualitas Udara dan Standar Regulasi:** Untuk menentukan seberapa baik kualitas udara saat ini dibandingkan dengan standar regulasi, Anda perlu membandingkan nilai PM2.5 yang terukur dengan batas ambang batas yang ditetapkan oleh otoritas lingkungan. Jika nilai PM2.5 melebihi batas yang ditetapkan secara konsisten, itu menunjukkan bahwa kualitas udara mungkin tidak memenuhi standar.
  
- **Tren Musiman:** Analisis tren musiman menunjukkan rata-rata PM2.5 per bulan. Dengan melihat grafik bar yang menunjukkan tren musiman, Anda dapat mengidentifikasi bulan-bulan di mana kualitas udara cenderung lebih buruk atau lebih baik. Misalnya, peningkatan PM2.5 di bulan-bulan tertentu dapat menunjukkan pola musiman yang mempengaruhi kualitas udara, seperti pembakaran bahan bakar yang lebih tinggi selama musim dingin atau peningkatan aktivitas industri.

### 2. **Apa dampak dari aktivitas industri utama di wilayah ini terhadap kualitas udara, dan apakah terdapat pola hubungan antara jenis industri tertentu dan tingkat polusi udara yang terukur?**

**Kesimpulan:**
- **Dampak Aktivitas Industri:** Untuk mengevaluasi dampak aktivitas industri terhadap kualitas udara, analisis korelasi antara PM2.5 dan berbagai variabel industri seperti emisi atau jumlah aktivitas industri diperlukan. Dengan memeriksa data, Anda dapat menentukan apakah ada hubungan signifikan antara jenis industri tertentu dan tingkat polusi udara. 

- **Korelasi dengan Kondisi Cuaca:** Korelasi antara PM2.5 dan variabel cuaca seperti suhu, tekanan, kelembapan, dan curah hujan memberikan wawasan tambahan tentang faktor-faktor yang mempengaruhi kualitas udara. Korelasi yang kuat antara PM2.5 dan variabel cuaca tertentu dapat menunjukkan bahwa perubahan kondisi cuaca berkontribusi pada fluktuasi dalam tingkat polusi udara.

**Rekomendasi untuk Tindakan Selanjutnya:**
- **Penyesuaian Regulasi:** Jika kualitas udara sering melebihi standar, pertimbangkan untuk merekomendasikan penyesuaian regulasi atau kebijakan mitigasi untuk mengurangi emisi dari sumber-sumber utama.
  
- **Strategi Musiman:** Implementasikan strategi pengendalian polusi yang mempertimbangkan pola musiman, seperti pengurangan emisi selama bulan-bulan dengan kualitas udara yang lebih buruk.

- **Pengelolaan Aktivitas Industri:** Identifikasi dan tangani sumber industri yang memberikan dampak terbesar pada kualitas udara dan pertimbangkan langkah-langkah pengendalian yang lebih ketat untuk industri-industri tersebut.

- **Adaptasi Terhadap Kondisi Cuaca:** Buat rencana untuk mengelola kualitas udara yang mempertimbangkan kondisi cuaca yang mempengaruhi konsentrasi PM2.5.

Dengan demikian, analisis ini membantu memahami dan menangani masalah kualitas udara dengan pendekatan yang lebih terfokus dan berbasis data.
""")
