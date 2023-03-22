from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("my_scaler.save")

import pyrebase

config = {
    "apiKey": "AIzaSyAZj0MX19-WeHexcoJICUGbaQDO6POs5cA",
    "authDomain": "nodemcu-e945b.firebaseapp.com",
    "databaseURL": "https://nodemcu-e945b-default-rtdb.firebaseio.com",
    "projectId": "nodemcu-e945b",
    "storageBucket": "nodemcu-e945b.appspot.com",
    "messagingSenderId": "1092087552226",
    "appId": "1:1092087552226:web:2217b4b9aed2e772d5e6bb",
    "measurementId": "G-BDW1T5JV4D"
}

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/home", methods=["GET"])
@app.route("/")
def hello():
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    ph_val = db.child("sensors_data").child("15:51:43").child("Ph").get()
    tds_val = db.child("sensors_data").child("15:51:43").child("Tds").get()
    temp_val = db.child("sensors_data").child("15:51:43").child("Temperature").get()
    ph = ph_val.val()
    tds = tds_val.val()
    temp = temp_val.val()


    input_features = [ph, 290, 10000, 8, 481, 4, 18, 124, 6]
    features_value = [np.array(input_features)]

    feature_names = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                     "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

    df = pd.DataFrame(features_value, columns=feature_names)
    df = scaler.transform(df)
    output = model.predict(df)

    if output[0] == 1:
        prediction = "safe"
    else:
        prediction = "not safe"

    return render_template('predict.html', prediction_text="water is {} for human consumption ".format(prediction), ph_val_use=ph, tds_val_use=tds)

if __name__ == "__main__":
    app.run(debug=True)
