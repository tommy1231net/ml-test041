import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

# 1. Load the dataset from the exported CSV
# Note: Ensure 'fraud_detection_data.csv' is in the same directory
df = pd.read_csv("fraud_detection_data.csv")

# 2. Separate features and target label
# Drop 'Class' (target) and 'row_id'/'split_key' if you added them in SQL
X = df.drop(['Class'], axis=1)
y = df['Class']

# 3. Split the data into Training and Testing sets
# Use shuffle=False to respect the chronological order (Time-series split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Initialize and train the XGBoost model
# scale_pos_weight is used to handle the extreme class imbalance (approx. 0.17% fraud)
# Calculation: count(negative) / count(positive)
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=580,  # Balanced weight for fraud detection
    use_label_encoder=False,
    eval_metric='logloss'
)

print("Starting model training...")
model.fit(X_train, y_train)
print("Training completed.")

# 5. Perform inference on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Evaluate the model performance
# Focus on Recall and Precision-Recall AUC due to data imbalance
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# AUPRC (Area Under the Precision-Recall Curve) is the key metric for anomaly detection
auprc = average_precision_score(y_test, y_prob)
print(f"\nArea Under the Precision-Recall Curve (AUPRC): {auprc:.4f}")

# 7. Save the model for future deployment (Vertex AI / Cloud Run)
model.save_model("model.bst")
print("\nModel saved as model.bst")