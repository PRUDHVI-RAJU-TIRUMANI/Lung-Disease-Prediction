from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("decisionmodel.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    age = int(request.form["age"])
    smoking = int(request.form["smoking"])
    yellow_fingers = int(request.form["yellow_fingers"])
    anxiety = int(request.form["anxiety"])
    peer_pressure = int(request.form["peer_pressure"])
    chronic_disease = int(request.form["chronic_disease"])
    fatigue = int(request.form["fatigue"])
    allergy = int(request.form["allergy"])
    wheezing = int(request.form["wheezing"])
    alcohol_consuming = int(request.form["alcohol_consuming"])
    coughing = int(request.form["coughing"])
    shortness_of_breath = int(request.form["shortness_of_breath"])
    swallowing_difficulty = int(request.form["swallowing_difficulty"])
    chest_pain = int(request.form["chest_pain"])

    features = np.array([[
        gender,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol_consuming,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain
    ]])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    # Probability of Lung Cancer (class 2)
    cancer_probability = round(probabilities[1] * 100, 2)

    if prediction == 2:
        result = "High Risk of Lung Cancer"
    else:
        result = "Low Risk of Lung Cancer"

    return render_template(
        "index.html",
        prediction=result,
        cancer_probability=cancer_probability
    )


if __name__ == "__main__":
    app.run(debug=True)