import joblib
import pandas as pd

# Load the saved model
model = joblib.load("../models/model.pkl")

# Load the saved scaler
scaler = joblib.load("../models/scaler.pkl")

# Load the saved label encoder
label_encoder = joblib.load("../models/label_encoder.pkl")


# Sample input
sample = pd.DataFrame([[
    75,
    80,
    70,
    85,
    65,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0
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

# Scale the input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

# Convert prediction back to label
result = label_encoder.inverse_transform(prediction)

print("Prediction:", result[0])