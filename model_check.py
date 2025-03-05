import dill  # Use dill instead of pickle
import pandas as pd
import numpy as np

# Load the saved model
with open("flight_price_rf_dill.pkl", "rb") as file:
    loaded_model = dill.load(file)  # Using dill to load the model

print("Model loaded successfully.")

# Sample test input
sample_input = pd.DataFrame([[3, 12, 3, 17, 25, 18, 5, 24, 40, 
                          0, 0, 1, 0, 0, 0, 0, 0, 0, 
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

# Predict using loaded model
predicted_price = loaded_model.predict(sample_input)
print("Predicted Price:", predicted_price)
