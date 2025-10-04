import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
from huggingface_hub import HfApi, HfFolder
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------
# Paths & Repos
# -------------------
dataset_path = "data/tourism.csv"
model_path = "artifacts/tourism_xgb_model.pkl"
dataset_repo = "absethi1894/Visit_with_Us"
model_repo = "absethi1894/MLOps"

# -------------------
# Load Dataset
# -------------------
df = pd.read_csv(dataset_path)
print(f"Dataset loaded successfully.\nColumns: {list(df.columns)}")

# Select target
target_column = "ProdTaken"
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert object columns to category for XGBoost
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# -------------------
# Model Training
# -------------------
mlflow.set_experiment("tourism-mlops-training-experiment")
print("Starting model training...")

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

print("Model training completed.")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# -------------------
# Hugging Face Upload
# -------------------
api = HfApi()
token = HfFolder.get_token()

# Model repo
try:
    api.repo_info(repo_id=model_repo, repo_type="model", token=token)
    print(f"Model repo '{model_repo}' exists. Uploading model...")
except Exception:
    print(f"Model repo '{model_repo}' not found. Creating repo...")
    api.create_repo(repo_id=model_repo, repo_type="model", private=False, token=token)

# Dataset repo
try:
    api.repo_info(repo_id=dataset_repo, repo_type="dataset", token=token)
    print(f"Dataset repo '{dataset_repo}' exists. Skipping creation.")
except Exception as e:
    if "409" in str(e):
        print(f"Dataset repo '{dataset_repo}' already exists. Skipping creation.")
    else:
        print(f"Dataset repo '{dataset_repo}' not found. Creating repo...")
        api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False, token=token)

# Upload model file
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_xgb_model.pkl",
    repo_id=model_repo,
    repo_type="model",
    token=token,
)
print(f"Model uploaded to Hugging Face model repo '{model_repo}'")
