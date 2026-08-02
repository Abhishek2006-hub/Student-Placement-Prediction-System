from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ==========================
    # Numerical Inputs
    # ==========================

    ssc_p = float(request.form["ssc_p"])
    hsc_p = float(request.form["hsc_p"])
    degree_p = float(request.form["degree_p"])
    etest_p = float(request.form["etest_p"])
    mba_p = float(request.form["mba_p"])

    # ==========================
    # Categorical Inputs
    # ==========================

    gender = request.form["gender"]
    ssc_b = request.form["ssc_b"]
    hsc_b = request.form["hsc_b"]
    hsc_s = request.form["hsc_s"]
    degree_t = request.form["degree_t"]
    workex = request.form["workex"]
    specialisation = request.form["specialisation"]

    # ==========================
    # One Hot Encoding
    # ==========================

    sample = pd.DataFrame([[
        ssc_p,
        hsc_p,
        degree_p,
        etest_p,
        mba_p,

        1 if gender == "Male" else 0,

        1 if ssc_b == "Others" else 0,

        1 if hsc_b == "Others" else 0,

        1 if hsc_s == "Commerce" else 0,
        1 if hsc_s == "Science" else 0,

        1 if degree_t == "Others" else 0,
        1 if degree_t == "Sci&Tech" else 0,

        1 if workex == "Yes" else 0,

        1 if specialisation == "Mkt&HR" else 0

    ]], columns=[
        'ssc_p',
        'hsc_p',
        'degree_p',
        'etest_p',
        'mba_p',
        'gender_M',
        'ssc_b_Others',
        'hsc_b_Others',
        'hsc_s_Commerce',
        'hsc_s_Science',
        'degree_t_Others',
        'degree_t_Sci&Tech',
        'workex_Yes',
        'specialisation_Mkt&HR'
    ])

    # Scale
    sample_scaled = scaler.transform(sample)

    # Prediction
    prediction = model.predict(sample_scaled)

    # Probability
    probability = model.predict_proba(sample_scaled)

    confidence = round(max(probability[0]) * 100, 2)

    result = label_encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)