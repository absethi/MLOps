import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
import joblib
from huggingface_hub import HfApi, HfFolder, HfHubHTTPError

# Paths & repo names
dataset_path = "data/tourism.csv"
model_output_path = "artifacts/tourism_xgb_model.pkl"
model_repo = "absethi1894/MLOps"
dataset_repo = "absethi1894/Visit_with_Us"

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# Target selection
target_col = "ProdTaken"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Features & preprocessing
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical columns to 'category' dtype for XGBoost
categorical_cols = X.select_dtypes(include="object").columns
for col in categorical_cols:
    X[col] = X[col].astype("category")

# Train-test split
if y.value_counts().min() < 2:
    print(f"Warning: smallest class has {y.value_counts().min()} sample(s). Using simple split.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Starting model training...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True  # Important for categorical columns
)
model.fit(X_train, y_train)
print("Model training completed.")

# Save locally
joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")

# Log with MLflow
mlflow.set_experiment("tourism-mlops-training-experiment")
with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.sklearn.log_model(model, "model")
print("MLflow logging done.")

# Upload to Hugging Face Hub
api = HfApi()
token = HfFolder.get_token()
if token is None:
    print("No HF token found. Skipping HF upload.")
else:
    try:
        api.repo_info(repo_id=model_repo)
        print(f"Model repo '{model_repo}' exists. Uploading model...")
    except HfHubHTTPError:
        print(f"Model repo '{model_repo}' not found. Creating repo...")
        api.create_repo(repo_id=model_repo, private=False, repo_type="model")

    api.upload_file(
        path_or_fileobj=model_output_path,
        path_in_repo=os.path.basename(model_output_path),
        repo_id=model_repo,
        repo_type="model",
        token=token
    )
    print(f"Model uploaded to HF: {model_repo}")

# Upload dataset to HF
try:
    api.repo_info(repo_id=dataset_repo)
    print(f"Dataset repo '{dataset_repo}' exists. Skipping creation.")
except HfHubHTTPError:
    print(f"Dataset repo '{dataset_repo}' not found. Creating repo...")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)

api.upload_file(
    path_or_fileobj=dataset_path,
    path_in_repo=os.path.basename(dataset_path),
    repo_id=dataset_repo,
    repo_type="dataset",
    token=token
)
print(f"Dataset uploaded to HF: {dataset_repo}")
