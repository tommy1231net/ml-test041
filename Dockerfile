# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory
WORKDIR /app

# --- MLOps Insight: Install OS-level dependencies for XGBoost ---
# libgomp1 is required for OpenMP (multi-threading) in XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application files and the trained model
# Ensure main.py and model.pkl (or model.bst) are in the same directory
COPY . .

# 5. Run the application
# Cloud Run expects the container to listen on the port defined by $PORT (default 8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]