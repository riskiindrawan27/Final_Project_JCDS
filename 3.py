import pandas as pd 
import joblib
df = pd.read_csv('liver_new.csv')
x = df.drop(['Dataset'], axis=1)
y = df['Dataset']
from sklearn.svm import SVC
model = SVC(gamma='auto')
model.fit(x, y)
joblib.dump(model, 'modelML1')
