import pandas as pd 
import joblib
df = pd.read_csv('liver_new.csv')
x = df.drop(['Dataset'], axis=1)
y = df['Dataset']
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', max_iter=500)
model.fit(x,y)
joblib.dump(model, 'modelML')