import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv(
    'indian_liver_patient.csv',
    na_values={
    'Albumin_and_Globulin_Ratio': 0
    }
)
# print(df[df['Albumin_and_Globulin_Ratio'].isnull()])

# print('=====================================================================')
# print('Cek Nilai NaN (kosong) pada DataFrame')
# print('=====================================================================')
# print(df.isnull().sum())

# print('\n=====================================================================')
# print('Cek Tipe data dari setiap kolom')
# print('=====================================================================')
# print(df.dtypes)

# print('\n=====================================================================')
# print('Cek 10 Nilai urutan teratas')
# print('=====================================================================')
# print(df.head(10))

# print('\n=====================================================================')
# print('Cek 10 Nilai urutan terbawah')
# print('=====================================================================')
# print(df.tail(10))

# print('\n=====================================================================')
# print('Cek nilai Albumin_and_Globulin_Ratio')
# print('=====================================================================')
# print(df[df['Albumin_and_Globulin_Ratio'].isnull()])

# ==============================================================================
'''
1.  Dari data terlihat angka 0 pda setiap kolom mengindikasikan bahwa
    nilai dari kolom tersebut tidak diisi/ bernilai NaN
2.  Ubah nilai dari setiap nilai 0 menjadi NaN,
    Kecuali kolom outcome. Karena nilali 0 pada kolom Dataset mengindikasikan bahwa
    orang tersebut tidak terjangkit penyakit liver
3.  Menambahkan fungsi "na_values()" pada bagian fungsi "read_csv()"
'''

print('\n=====================================================================')
print('Nama-nama Kolom:')
print('=====================================================================')
print(df.columns.values)

print('\n=====================================================================')
print('Kolom yang memiliki nilai kosong (NaN):')
print('=====================================================================')
ksg = []
jml_k =[]
for i in df.columns.values:
    a = df[i].isna().sum()
    if a > 0:
        jml_k.append(a)
        ksg.append(i)
print(ksg)
print(jml_k)

# ==============================================================================
'''
Untuk mengetahui cara/ metode yang paling baik untuk mengisi kolom/nilai yang kosong(NaN)
yaitu dengan melihat dari tabel korelasi antar "feature"
'''
# ==============================================================================
# print('\n=====================================================================')
# print('Tabel korelasi antar "Features:"')
# print('=====================================================================')
# print(df.corr(method='spearman'))
# ==============================================================================
# corr = df.corr()
# corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# 'RdBu_r' & 'BrBG' are other good diverging colormaps

# ==============================================================================
# Korelasi antar Features (Heatmap)
# ==============================================================================

# corrmat = df.corr(method='spearman') 
# plt.subplots(figsize=(9,8)) 
# plt.subplots_adjust(bottom=0.23)
# sns.heatmap(corrmat, cmap ="YlGnBu", linewidths = 0.1) 
# plt.title('Heatmap Korelasi')
# plt.xticks(rotation = 45)               # atur rotasi dari value x dan y
# plt.yticks(rotation = 45)
# plt.tight_layout()
# plt.show()

# ==============================================================================
# from pandas.plotting import scatter_matrix
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(25, 25), diagonal='hist')
# plt.show()

# ==============================================================================
'''
1.  Kolom yang terdapat nilai NaN adalah:
    ['Albumin_and_Globulin_Ratio']
2.  Dari tabel Korelasi antar "Features" dan Heatmap diagram, 
    kita dapat menentukan metode korelasi yang tepat untuk mengisi nilai yang kosong
4.  Membuat functions untuk mengisi kolom yang kosong berdasarkan data diatas 
5.  Melakukan pembualatan nilai dari beberapa feature yang memiliki tipe data float:
    ['Total_Bilirubin' 'Direct_Bilirubin' 'Total_Protiens' 'Albumin' 'Albumin_and_Globulin_Ratio']
'''
# ==============================================================================
# Functions Pembulatan nilai
def pembulatan(x):
    br = []
    for i in df[x].values:
        a = round(i,2)
        br.append(a)
    df[x] = br

pembulatan('Total_Bilirubin')
pembulatan('Direct_Bilirubin')
pembulatan('Total_Protiens')
pembulatan('Albumin')
pembulatan('Albumin_and_Globulin_Ratio')

# Penghapusan nilai NaN
df = df.dropna(subset=['Albumin_and_Globulin_Ratio'])

# Penghapusan Kolom Gender
df = df.drop(['Gender'], axis='columns')

# get dummies
dfNew = pd.get_dummies(df['Dataset'])
dfKomplit = pd.concat([df, dfNew], axis='columns')
dfKomplit = dfKomplit.drop(['Dataset'], axis='columns')

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dfLabel = label.fit_transform(df['Dataset'])
df['Dataset'] = label.fit_transform(df['Dataset'])
print(df)

# Convert Dataframe to CSV File
df.to_csv('liver_new.csv', index=False)