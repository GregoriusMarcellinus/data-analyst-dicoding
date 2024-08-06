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

# Kesimpulan
st.write("""
### Pertanyaan Bisnis
1. **Bagaimana tren musiman dan tahunan dari level PM2.5 di Stasiun Guanyuan?**
   - Dari grafik rata-rata level PM2.5 bulanan dan dekomposisi deret waktu, kita dapat mengidentifikasi pola musiman dan tahunan dalam konsentrasi PM2.5. Apakah terdapat bulan-bulan tertentu dengan level PM2.5 yang lebih tinggi atau lebih rendah secara konsisten?

2. **Bagaimana hubungan antara kondisi cuaca dan level PM2.5 di Stasiun Guanyuan?**
   - Dengan menggunakan heatmap korelasi, kita dapat menganalisis hubungan antara indikator kualitas udara (seperti PM2.5) dan berbagai kondisi cuaca (seperti suhu, tekanan, dan kelembapan). Apakah terdapat korelasi signifikan yang dapat membantu memprediksi level PM2.5 berdasarkan kondisi cuaca tertentu?

### Kesimpulan dari Visualisasi Data

1. **Tren Musiman dan Tahunan:**
   - Dari analisis tren musiman, kita melihat bahwa rata-rata level PM2.5 bervariasi sepanjang tahun dengan beberapa bulan menunjukkan level yang lebih tinggi. Grafik batang rata-rata level PM2.5 bulanan menunjukkan adanya puncak pada bulan tertentu yang mungkin terkait dengan perubahan musim atau kegiatan manusia.
   - Dekomposisi deret waktu menunjukkan komponen musiman yang berulang setiap tahun serta tren jangka panjang dari level PM2.5. Komponen musiman mengindikasikan fluktuasi periodik yang konsisten, sementara komponen tren memberikan gambaran mengenai arah jangka panjang dari perubahan level PM2.5.

2. **Hubungan antara Kondisi Cuaca dan Level PM2.5:**
   - Heatmap korelasi menunjukkan adanya hubungan antara PM2.5 dengan indikator kualitas udara lainnya serta kondisi cuaca seperti suhu, tekanan, dan kelembapan. Misalnya, mungkin terdapat korelasi negatif antara PM2.5 dan suhu, yang menunjukkan bahwa level PM2.5 cenderung lebih rendah pada suhu yang lebih tinggi.
   - Korelasi ini memberikan wawasan penting mengenai faktor-faktor cuaca yang mungkin mempengaruhi kualitas udara, dan dapat digunakan untuk model prediksi serta kebijakan pengendalian kualitas udara.

3. **Perbandingan dengan Standar Kualitas Udara:**
   - Dari perbandingan dengan standar kualitas udara WHO, kita mengetahui bahwa rata-rata tahunan PM2.5 di Stasiun Guanyuan melebihi standar tahunan WHO sebesar 10 µg/m³, dan rata-rata harian PM2.5 juga melebihi standar 24 jam WHO sebesar 25 µg/m³. Ini menunjukkan bahwa kualitas udara di Stasiun Guanyuan tidak memenuhi standar kesehatan yang ditetapkan oleh WHO, yang dapat memiliki implikasi serius bagi kesehatan masyarakat.
""")
