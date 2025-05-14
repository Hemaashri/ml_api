from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize the FastAPI app
app = FastAPI()

# Define the data model for the input
class InputData(BaseModel):
    features: list[float]

# Root route
@app.get("/")
def read_root():
    return {"message": "API is live!"}

# Predict route
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert the features to a NumPy array and reshape it for prediction
        X = np.array(data.features).reshape(1, -1)
        
        # Scale the data using the pre-loaded scaler
        X_scaled = scaler.transform(X)
        
        # Get the prediction from the model
        prediction = model.predict(X_scaled)
        
        # Return the prediction result
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
