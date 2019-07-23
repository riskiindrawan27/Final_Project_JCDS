# Final_Project_JCDS
# Liver Prediction
Project ini bertujuan untuk memudahkan para spesialis/dokter untuk mengambil keputusan terkait "Apakah seseorang terindikasi penyakit Liver atau tidak?"

DESKRIPSI PROJECT
- Tema : Prediksi Liver 
- Project ini bersumber dari "North East of Andhra Pradesh, India.
- Semua data adalah pasien Liver
- [Sumber Dataset](https://www.kaggle.com/uciml/indian-liver-patient-records)

PENGOLAHAN DATA
- Baca Data :

> df = pd.read_csv('indian_liver_patient.csv')

- Cek Nilai NaN:

> print(df.isnull().sum())

- Cek nilai Albumin_and_Globulin_Ratio yang kosong
> print(df[df['Albumin_and_Globulin_Ratio'].isnull()])

- Hapus Nilai NaN pada Albumin_and_Globulin_Ratio
> df = df.dropna(subset=['Albumin_and_Globulin_Ratio'])

- Hapus Kolom Gender 
df = df.drop(['Gender'], axis='columns')

- get dummies
> dfNew = pd.get_dummies(df['Dataset'])
> dfKomplit = pd.concat([df, dfNew], axis='columns')
> dfKomplit = dfKomplit.drop(['Dataset'], axis='columns')

-  Label Encoder
> from sklearn.preprocessing import LabelEncoder
> label = LabelEncoder()
> dfLabel = label.fit_transform(df['Dataset'])
> df['Dataset'] = label.fit_transform(df['Dataset'])


