import streamlit as st
from joblib import load
import numpy as np
import joblib
from pathlib import Path

# 1. Get the directory of the current script
script_dir = Path(__file__).parent

# 2. Build the absolute path to your model file
model_path = script_dir / "linear_regression_model.joblib"

# 3. Load the model safely
model = joblib.load(model_path)


# Create a simple user input
user_input = st.number_input('Enter house size:', min_value=100, max_value=10000, step=50)

# Reshape the input for the model
input_array = np.array([user_input]).reshape(-1, 1)

# Predict the house price
if st.button('Predict Price'):
    predicted_price = model.predict(input_array)
    st.write(f"The predicted house price is: ${predicted_price[0]:.2f}")
