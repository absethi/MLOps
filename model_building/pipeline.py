# model_building/pipeline.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
from huggingface_hub import HfApi

# -----------------------------
# 1. Load dataset
# -----------------------------
dataset_path = "data/tourism.csv"
os.makedirs("data", exist_ok=True)

if not os.path.isfile(dataset_path):
    # Replace with actual dataset URL if needed
    import urllib.request
    url = "https://your-dataset-url.com/tourism.csv"
    urllib.request.urlretrieve(url, dataset_path)

df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# -----------------------------
# 2. Choose target
# -----------------------------
if "Failure" in df.columns:
    target_col = "Failure"
else:
    # Pick a valid column for demo (binary/multi-class)
    target_col = "ProdTaken"
    print(f"Target column 'Failure' not found. Using '{target_col}' instead.")

y = df[target_col]
X = df.drop(columns=[target_col, 'CustomerID', 'Unnamed: 0'])  # drop IDs

# -----------------------------
# 3. Train-test split
# -----------------------------
# If the smallest class has only 1 sample, use simple split
min_class_count = y.value_counts().min()
if min_class_count < 2:
    print(f"Warning: smallest class has {min_class_count} sample(s). Using simple train/test split instead of stratified CV.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# -----------------------------
# 4. Start MLflow experiment
# -----------------------------
mlflow.set_experiment("tourism-mlops-training-experiment")
mlflow.start_run()
print("Starting model training...")

# -----------------------------
# 5. Train model
# -----------------------------
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("Model trained successfully.")

# Save model locally
os.makedirs("artifacts", exist_ok=True)
model_file = "artifacts/tourism_xgb_model.pkl"
import pickle
with open(model_file, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}")

mlflow.end_run()

# -----------------------------
# 6. Upload dataset & model to Hugging Face
# -----------------------------
hf_token = os.environ.get("HF_TOKEN")
api = HfApi()

# Dataset repo
dataset_repo = "absethi1894/Visit_with_Us"
try:
    api.repo_info(repo_id=dataset_repo, repo_type="dataset")
except:
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)

# Upload dataset
api.upload_file(
    path_or_fileobj=dataset_path,
    path_in_repo="tourism.csv",
    repo_id=dataset_repo,
    token=hf_token
)
print("Dataset uploaded to Hugging Face Hub.")

# Model repo
model_repo = "absethi1894/MLOps"
try:
    api.repo_info(repo_id=model_repo, repo_type="model")
except:
    api.create_repo(repo_id=model_repo, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="tourism_xgb_model.pkl",
    repo_id=model_repo,
    token=hf_token
)
print("Model uploaded to Hugging Face Hub.")
