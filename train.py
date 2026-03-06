import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# 1. Load the dataset
# Make sure your BigQuery export is named 'taxi_data.csv' or adjust the name
data_path = 'taxi_data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Ensure {data_path} exists in the same directory.")

df = pd.read_csv(data_path)

# 2. Preprocessing
# We only use numerical features for this demo to keep the model lightweight
features = [
    'trip_distance', 
    'pickup_hour', 
    'pickup_day', 
    'passenger_count', 
    'pickup_location_id', 
    'dropoff_location_id', 
    'rate_code', 
    'payment_type'
]
target = 'total_amount'

X = df[features]
y = df[target]

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Definition: XGBRegressor
# MLOps Tip: We use Regression because we want to predict a continuous price value
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='reg:squarederror', # Required for regression
    n_jobs=-1
)

# 4. Training
print("Starting model training...")
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Training Complete.")
print(f"Mean Absolute Error: ${mae:.2f}") # Average error in dollars
print(f"R2 Score: {r2:.4f}")

# 6. Save the Model as .bst
# Native XGBoost format for efficient deployment
model.save_model('model.bst')
print("Model saved as 'model.bst'")