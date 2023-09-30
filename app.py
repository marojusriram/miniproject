from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle


Knr = pickle.load(open('Knr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

app = Flask(__name__)

df=pd.read_csv('complete_dataset - Copy.csv')
state_values=[]
crop_values = []
season_values=[]
soil_values=[]
district_values=[]

state_values.append(df['state_names'].unique().tolist())
district_values.append(df['district_names'].unique().tolist())
crop_values.append(df['crop_names'].unique().tolist())
season_values.append(df['season_names'].unique().tolist())
soil_values.append(df['soil_type'].unique().tolist())

@app.route('/')
def index():
    return render_template('index.html', state_values=state_values, crop_values=crop_values, season_values=season_values, soil_values=soil_values, district_values=district_values)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        season_names = request.form.get('season_names')
        crop_names = request.form.get('crop_names')
        area = request.form['area']
        temperature = request.form['temperature']
        precipitation = request.form['precipitation']
        humidity = request.form['humidity']
        soil_type = request.form.get('soil_type')
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']




        features = np.array([[season_names,crop_names,area,temperature,precipitation,humidity,soil_type,N,P,K]])

        transformed_features = preprocessor.transform(features)
        predicted_value = Knr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html',predicted_value=predicted_value, state_values=state_values, crop_values=crop_values, season_values=season_values, soil_values=soil_values, district_values=district_values)

if __name__ == '__main__':
    app.run(debug=True)