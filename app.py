from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# load model + columns
model, columns = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])

    # full feature structure
    input_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 0,
        "smoker_yes": 0,
        "region_northwest": 0,
        "region_southeast": 0,
        "region_southwest": 0
    }

    input_df = pd.DataFrame([input_dict])

    # match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction_text=f"Prediction: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)