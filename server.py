from flask import Flask,request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_csv("nasa.csv")

df = df.drop(['Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
                  'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Est Dia in KM(max)',
                  'Relative Velocity km per hr', 'Miles per hour',
                  'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)',
                  'Semi Major Axis',
                  'Neo Reference ID', 'Name',
                  'Close Approach Date', 'Epoch Date Close Approach', 'Orbit Determination Date', 'Equinox',
                  'Orbiting Body'], axis=1)

app = Flask(__name__)

model5 = pickle.load(open('model5.pkl','rb'))




@app.route('/')
def index():
    return render_template("index.html") #due to this function we are able to send our webpage to client(browser) - GET

@app.route('/predict',methods=['POST','GET'])  #gets inputs data from client(browser) to Flask Server - to give to ml model
def predict():
    features = [float(x) for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    #our model was trained on Normalized(scaled) data


    X = df.iloc[:, :-1].values
    sst=StandardScaler().fit(X)
    output = model5.predict(sst.transform(final))
    print(output)

    if output[0] == 0:
        return render_template('index.html', pred = f'NOT HAZARDOUS TO EARTH')
    else:
        return render_template('index.html', pred = f'HAZARDOUS TO EARTH')

if __name__ == '__main__':
    app.run(debug = True)