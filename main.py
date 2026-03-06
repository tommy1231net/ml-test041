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

# 1. Load the trained XGBoost Regression model (.bst format)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.bst')

print(f"DEBUG: Loading Taxi Fare model from {model_path}")

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize Regressor and load the .bst file
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print("DEBUG: Taxi Fare model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    raise e

# 2. Define the request schema based on BigQuery features
class FarePredictionRequest(BaseModel):
    trip_distance: float = Field(..., example=2.5)
    pickup_hour: int = Field(..., ge=0, le=23, example=14)
    pickup_day: int = Field(..., ge=1, le=7, example=3) # 1=Sun, 7=Sat
    passenger_count: int = Field(..., ge=1, example=1)
    pickup_location_id: int = Field(..., example=161)
    dropoff_location_id: int = Field(..., example=237)
    rate_code: int = Field(..., example=1)
    payment_type: int = Field(..., example=1)
    # The actual fare paid by the user to compare with AI prediction
    actual_total_amount: float = Field(..., example=15.0)

@app.post("/predict")
def predict(request: FarePredictionRequest):
    # 3. Data Preparation
    # Ensure features are in the same order as training
    feature_names = [
        'trip_distance', 'pickup_hour', 'pickup_day', 
        'passenger_count', 'pickup_location_id', 'dropoff_location_id', 
        'rate_code', 'payment_type'
    ]
    
    feature_values = [
        request.trip_distance, request.pickup_hour, request.pickup_day,
        request.passenger_count, request.pickup_location_id, request.dropoff_location_id,
        request.rate_code, request.payment_type
    ]
    
    df_input = pd.DataFrame([feature_values], columns=feature_names)

    # 4. Inference: Predict the "Fair Price"
    predicted_fare = float(model.predict(df_input)[0])
    
    # 5. Anomaly Detection Logic
    # Calculate the absolute difference (Residual)
    diff = request.actual_total_amount - predicted_fare
    
    # Define Threshold: If the difference is more than $10 or 50% of predicted fare
    # This logic can be adjusted based on business requirements
    abs_diff = abs(diff)
    threshold = max(10.0, predicted_fare * 0.5) 
    
    is_anomaly = abs_diff > threshold
    
    # Determine the type of anomaly
    status = "Normal"
    if is_anomaly:
        status = "Overcharged" if diff > 0 else "Undercharged"

    # 6. Return JSON Response
    return {
        "is_anomaly": bool(is_anomaly),
        "status": status,
        "predicted_fare": round(predicted_fare, 2),
        "actual_fare": round(request.actual_total_amount, 2),
        "difference": round(diff, 2),
        "anomaly_score": round(abs_diff, 2),
        "metadata": {
            "model_format": "XGBoost Regressor (.bst)",
            "threshold_used": round(threshold, 2)
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)