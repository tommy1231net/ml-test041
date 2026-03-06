import os
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# --- CORS Settings ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the trained XGBoost model (.bst format)
# MLOps Insight: .bst files require xgb.Booster or XGBClassifier.load_model()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.bst')

print(f"DEBUG: Loading model from {model_path}")

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize the classifier and load the .bst file
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("DEBUG: Model (.bst) loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    raise e

# 2. Define the request schema
class PredictionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    # Expecting 26 features (V3 to V28)
    V_features: list[float] = Field(..., min_items=26, max_items=26)
    Amount: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # 3. Data Preparation
    # Constructing features in order: Time, V1, V2, V3...V28, Amount (Total 30)
    feature_values = [request.Time, request.V1, request.V2] + request.V_features + [request.Amount]
    
    # Define column names to match the training phase
    column_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Create DataFrame
    df_input = pd.DataFrame([feature_values], columns=column_names)

    # 4. Inference
    # predict_proba returns [prob_class_0, prob_class_1]
    fraud_probability = model.predict_proba(df_input)[0][1]
    
    threshold = 0.5
    prediction = 1 if fraud_probability > threshold else 0
    
    return {
        "is_fraud": bool(prediction),
        "fraud_score": round(float(fraud_probability), 4),
        "status": "High Risk" if prediction == 1 else "Normal",
        "metadata": {
            "model_format": "XGBoost Native (.bst)",
            "feature_count": len(feature_values)
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)