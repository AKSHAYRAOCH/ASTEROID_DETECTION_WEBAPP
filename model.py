import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('nasa.csv')


# Dropping completely correlated features and datetime features
df = df.drop(['Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Est Dia in KM(max)',
              'Relative Velocity km per hr', 'Miles per hour',
              'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)',
              'Semi Major Axis',
              'Neo Reference ID', 'Name',
              'Close Approach Date', 'Epoch Date Close Approach', 'Orbit Determination Date','Equinox','Orbiting Body'],axis=1)

#encoding the y variable
from sklearn.preprocessing import LabelEncoder
l_enc = LabelEncoder()
df['hazardous'] = l_enc.fit_transform(df.Hazardous)

X = df.iloc[:,:-2].values
Y = df.iloc[:,-1:].values
print(X)
print(Y)
print(len(X[0]))



#splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1,shuffle=True)

from sklearn.preprocessing import StandardScaler
sst = StandardScaler()
x_train = sst.fit_transform(x_train)


from sklearn.ensemble import RandomForestClassifier
model5 = RandomForestClassifier()
model5.fit(x_train,y_train)

y1 = [21.6, 0.127, 6.115, 62753692.0, 17.0, 5.0, 0.025, 4.634, 2458000.0, 0.4255, 6.025, 314.373, 609.59, 0.808, 57.25, 2.005, 2.459, 264.83, 0.6]
y_pred5 = model5.predict(sst.transform([y1]))
print(y_pred5)

pickle.dump(model5,open('model5.pkl','wb'))
model5 = pickle.load(open('model5.pkl','rb'))
print("sucess loaded")