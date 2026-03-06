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
# すべてのオリジンからのアクセスを許可し、フロントエンドからのリクエストに対応
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

print(f"Loading model from: {model_path}")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # 初期化時にエラーを出すことで、デプロイ失敗としてログに残るようにする
    raise e

# 2. Define the request schema for Credit Card Fraud dataset
# Total 30 features: Time(1) + V1~V28(28) + Amount(1)
class PredictionRequest(BaseModel):
    Time: float
    # V1, V2を個別に定義し、残りのV3-V28をリストで受け取る構成
    V1: float
    V2: float
    # V_features expects a list of 26 floats (V3 to V28)
    V_features: list[float] = Field(..., min_items=26, max_items=26)
    Amount: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # 3. Data Preparation
    # XGBoost requires columns in the EXACT same order and names as training
    # Constructing features in order: Time, V1, V2, V3...V28, Amount
    feature_values = [request.Time, request.V1, request.V2] + request.V_features + [request.Amount]
    
    # Define EXACT column names used during training
    # This prevents 'feature_names mismatch' error in XGBoost
    column_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Create DataFrame with proper column names
    df_input = pd.DataFrame([feature_values], columns=column_names)

    # 4. Inference
    # Get probability for Class 1 (Fraud)
    # probabilities is an array like [[prob_0, prob_1]]
    fraud_probability = model.predict_proba(df_input)[0][1]
    
    # Thresholding logic: 0.5 is default, but can be lowered for higher Recall
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
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    # Cloud Run uses the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)