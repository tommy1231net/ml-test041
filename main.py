import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the trained XGBoost model
# Assuming you saved the model using joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')

# Define the request schema based on dataset features (Time, V1-V28, Amount)
class PredictionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    # ... (In a real app, you would define all V1-V28)
    # For simplicity in this example, we use a dynamic dict or list
    V_features: list[float] # Expecting a list of 28 floats
    Amount: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # 2. Prepare the input data for the model
    # Combine all features into a single row (2D array)
    # Order: Time, V1, V2, ..., V28, Amount
    features = [request.Time] + request.V_features + [request.Amount]
    df_input = pd.DataFrame([features])

    # 3. Execute Inference
    # Get probability for Class 1 (Fraud)
    fraud_probability = model.predict_proba(df_input)[0][1]
    
    # Simple threshold logic for classification
    # In fraud detection, thresholds are often adjusted (e.g., 0.5 or 0.2)
    prediction = 1 if fraud_probability > 0.5 else 0
    
    # 4. Return results
    return {
        "is_fraud": bool(prediction),
        "fraud_score": round(float(fraud_probability), 4),
        "status": "High Risk" if prediction == 1 else "Normal",
        "details": {
            "threshold": 0.5,
            "algorithm": "XGBoost (Cost-sensitive learning)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Cloud Run default port is 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)