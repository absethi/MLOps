import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import pickle
from huggingface_hub import HfApi, HfFolder
import requests  # Correct way to catch HTTP errors

# ---------------------------
# CONFIG
# ---------------------------
dataset_path = "data/tourism.csv"
model_repo = "absethi1894/MLOps"
dataset_repo = "absethi1894/Visit_with_Us"
model_save_path = "artifacts/tourism_xgb_model.pkl"

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# Target column
target_col = "ProdTaken"
if target_col not in df.columns:
    target_col = df.columns[0]  # fallback

# ---------------------------
# PREPROCESSING
# ---------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Train-test split
if y.value_counts().min() < 2:
    print(f"Warning: smallest class has {y.value_counts().min()} sample(s). Using simple train/test split instead of stratified CV.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------
# MODEL TRAINING
# ---------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True  # enable categorical support
)

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training completed.")

# Save model locally
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
with open(model_save_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_save_path}")

# ---------------------------
# LOG MODEL WITH MLFLOW
# ---------------------------
mlflow.set_experiment("tourism-mlops-training-experiment")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.sklearn.log_model(model, "tourism_xgb_model")

# ---------------------------
# HUGGING FACE HUB UPLOAD
# ---------------------------
api = HfApi()
# Model repo
try:
    api.repo_info(repo_id=model_repo)
    print(f"Model repo '{model_repo}' exists. Uploading model...")
except requests.exceptions.HTTPError:
    print(f"Model repo '{model_repo}' not found. Creating repo...")
    api.create_repo(repo_id=model_repo, private=False, repo_type="model")

# Dataset repo
try:
    api.repo_info(repo_id=dataset_repo)
    print(f"Dataset repo '{dataset_repo}' exists. Skipping creation.")
except requests.exceptions.HTTPError:
    print(f"Dataset repo '{dataset_repo}' not found. Creating repo...")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)

print("Pipeline execution completed successfully.")
