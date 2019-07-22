import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('liver_news.csv')
# print(df.isnull().sum())
# print(df.describe())

# Sumbu X & sumbu y
x = df.drop(['Dataset'], axis=1)
# print(x)
y = df['Dataset']
# print(y)

# ==============================================================================
# Splitting data
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    x,
    y,
    test_size=.1
)

from sklearn.svm import SVC
modelS = SVC(gamma='auto')
modelS.fit(xtr, ytr)
print(modelS.predict([[ 29,0.7,0.1,162,52,41,5.2,2.5,0.9]]))
print('SVM:',round(modelS.score(xts, yts)*100,2),'%')

from sklearn.svm import SVC
modelS = SVC(gamma='auto')
modelS.fit(x, y)
from sklearn.model_selection import cross_val_score
crosS = cross_val_score(modelS, x, y, cv=5 )
print('SVM:',round(crosS.mean()*100,2),'%')
print(modelS.predict([[ 29,0.7,0.1,162,52,41,5.2,2.5,0.9 ]]))

# ========================================================
# from sklearn.ensemble import RandomForestClassifier
# modelT = RandomForestClassifier(n_estimators=50)
# modelT.fit(xtr, ytr)
# print('RandomForest:',round(modelT.score(xts, yts)*100,2),'%')

# from sklearn.ensemble import RandomForestClassifier
# modelT = RandomForestClassifier(n_estimators=50)
# modelT.fit(x, y)
# from sklearn.model_selection import cross_val_score
# crosT = cross_val_score(modelT, x, y, cv=5 )
# print('RandomForest:',round(crosT.mean()*100,2),'%')
# ============================================================
# from sklearn.ensemble import ExtraTreesClassifier
# modelE = ExtraTreesClassifier(n_estimators=50)
# modelE.fit(xtr, ytr)
# print('ExtraTree:',round(modelE.score(xts, yts)*100,2),'%')

# from sklearn.ensemble import RandomForestClassifier
# modelE = RandomForestClassifier(n_estimators=50)
# modelE.fit(x, y)
# from sklearn.model_selection import cross_val_score
# crosE = cross_val_score(modelE, x, y, cv=5 )
# print('ExtraTree:',round(crosE.mean()*100,2),'%')
# ==============================================================================
# ML Metode

# from sklearn.linear_model import LogisticRegression
# # {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
# modelLog = LogisticRegression(solver='liblinear', max_iter=500)
# modelLog.fit(xtr, ytr)

# from sklearn.ensemble import RandomForestClassifier
# modelF = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)
# modelF.fit(xtr, ytr)

# from sklearn import tree
# modelT = tree.DecisionTreeClassifier()
# modelT.fit(xtr,ytr)

# from sklearn.svm import SVC
# modelS = SVC(gamma='auto')
# modelS.fit(xtr, ytr)

# def nNeigbors():
#     x = round(len(df)**.5)
#     if x % 2 == 0:
#         return x + 1
#     else:
#         return x

# from sklearn.neighbors import KNeighborsClassifier
# modelK = KNeighborsClassifier(n_neighbors=nNeigbors())
# modelK.fit(xtr, ytr)

# ==============================================================================
# print('=====================================================================')
# print('Score dari setiap metode ML:')
# print('=====================================================================')
# print('Logistic Regression:',round(modelLog.score(xts, yts)*100,2),'%')
# print('Random Forest Classifier:',round(modelF.score(xts, yts)*100,2),'%')
# print('Decision Tree:',round(modelT.score(xts, yts)*100,2),'%')
# print('SVM:',round(modelS.score(xts, yts)*100,2),'%')
# print('K-Neighbors Classifier:',round(modelK.score(xts, yts)*100,2),'%')

# # ==============================================================================
# from sklearn.linear_model import LogisticRegression
# # {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
# modelLog = LogisticRegression(solver='liblinear', max_iter=500)
# modelLog.fit(x, y)

# from sklearn.ensemble import RandomForestClassifier
# modelF = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)
# modelF.fit(x, y)

# from sklearn import tree
# modelT = tree.DecisionTreeClassifier()
# modelT.fit(x, y)

# from sklearn.svm import SVC
# modelS = SVC(gamma='auto')
# modelS.fit(x, y)

# from sklearn.neighbors import KNeighborsClassifier
# modelK = KNeighborsClassifier(n_neighbors=27)
# modelK.fit(x, y)

# from sklearn.model_selection import cross_val_score
# crosL = cross_val_score(modelLog, x, y, cv=5 )
# crosF = cross_val_score(modelF, x, y, cv=5 )
# crosT = cross_val_score(modelT, x, y, cv=5 )
# crosS = cross_val_score(modelS, x, y, cv=5 )
# crosK = cross_val_score(modelK, x, y, cv=5 )

# # ==============================================================================
# print('\n=====================================================================')
# print('Validation Score dari setiap metode ML:')
# print('=====================================================================')
# print('Logistic Regression:',round(crosL.mean()*100,2),'%')
# print('Random Forest Classifier:',round(crosF.mean()*100,2),'%')
# print('Decision Tree:',round(crosT.mean()*100,2),'%')
# print('SVM:',round(crosS.mean()*100,2),'%')
# print('K-Neighbors Classifier:',round(crosK.mean()*100,2),'%')

# print('\n=====================================================================')

# ==============================================================================
'''
Dari score yang ditampilkan terlihat bahwa metode SVM,
memiliki persentase yang tinggi dibangding dengan metode yang lain.
Maka diigunakan model SVM sebagai model/metode Machine Learning
'''
# ==============================================================================
