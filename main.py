import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# --- CORS Settings ---
# Allow all origins for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the trained XGBoost model
# MLOps Tip: Use absolute path to ensure the model is found in the container environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

print(f"DEBUG: Loading model from {model_path}")

try:
    # Check if file exists before loading to provide clearer error logs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = joblib.load(model_path)
    print("DEBUG: Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    # Force failure during startup so Cloud Run logs show the error immediately
    raise e

# 2. Define the request schema for Credit Card Fraud dataset
# Total 30 features: Time(1) + V1~V28(28) + Amount(1)
class PredictionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    # V_features expects a list of 26 floats (V3 to V28)
    V_features: list[float] = Field(..., min_items=26, max_items=26)
    Amount: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # 3. Data Preparation
    # Ensure features are in the exact order: Time, V1, V2, V3...V28, Amount
    feature_values = [request.Time, request.V1, request.V2] + request.V_features + [request.Amount]
    
    # Define EXACT column names used during training to prevent feature_names mismatch
    column_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Create DataFrame with proper column names
    df_input = pd.DataFrame([feature_values], columns=column_names)

    # 4. Inference
    # Get probability for Class 1 (Fraud)
    fraud_probability = model.predict_proba(df_input)[0][1]
    
    # Thresholding logic: Default 0.5
    threshold = 0.5
    prediction = 1 if fraud_probability > threshold else 0
    
    # 5. Return JSON Response
    return {
        "is_fraud": bool(prediction),
        "fraud_score": round(float(fraud_probability), 4),
        "status": "High Risk" if prediction == 1 else "Normal",
        "metadata": {
            "model_type": "XGBoost",
            "threshold": threshold,
            "feature_count": len(feature_values)
        }
    }

@app.get("/health")
def health_check():
    # Health check endpoint for Cloud Run/Monitoring
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable provided by Cloud Run
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)