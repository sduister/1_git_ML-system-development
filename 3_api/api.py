from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load trained model and feature boundaries
model = joblib.load("model.pkl")
feature_bounds = joblib.load("feature_bounds.pkl")

# Load actual vs. predicted results
results_df = pd.read_csv("results.csv")

# Initialize FastAPI
app = FastAPI()

# Define input schema for predictions
class InputData(BaseModel):
    Vs_kn: float
    Lwl_m: float
    Bmld_m: float
    T_m: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        features = [[data.Vs_kn, data.Lwl_m, data.Bmld_m, data.T_m]]
        prediction = model.predict(features)
        return {"predicted_resistance": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Endpoint to get training feature boundaries
@app.get("/feature_bounds")
def get_feature_bounds():
    return feature_bounds

# Endpoint to get actual vs. predicted results
@app.get("/results")
def get_results():
    return results_df.to_dict(orient="records")
