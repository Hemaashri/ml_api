from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
