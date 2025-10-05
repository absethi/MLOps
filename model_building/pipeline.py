import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import joblib
from huggingface_hub import HfApi, HfFolder

# ----------------------------
# Configuration
# ----------------------------
dataset_path = "data/tourism.csv"
dataset_repo = "absethi1894/Visit_with_Us"
model_repo = "absethi1894/MLOps"
model_artifact_path = "artifacts/best_tourism_model_v1.joblib"   # 👈 updated name

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# ----------------------------
# Select target and features
# ----------------------------
target_col = "ProdTaken"  # Replace with actual target if different
y = df[target_col]
X = df.drop(columns=[target_col, "CustomerID", "Unnamed: 0"])  # Drop irrelevant columns

# Convert object columns to categorical
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# Encode target if it's categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

num_classes = len(np.unique(y))
print(f"Detected {num_classes} unique classes in target '{target_col}'.")

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if num_classes > 1 else None
)
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# ----------------------------
# Train XGBoost Model
# ----------------------------
model = xgb.XGBClassifier(
    objective="multi:softprob" if num_classes > 2 else "binary:logistic",
    eval_metric="mlogloss" if num_classes > 2 else "logloss",
    enable_categorical=True,
    use_label_encoder=False,
    num_class=num_classes if num_classes > 2 else None
)

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training completed.")

# ----------------------------
# Save Model (Joblib format)
# ----------------------------
os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
joblib.dump(model, model_artifact_path)   # 👈 joblib instead of xgb.save_model
print(f"Model saved to {model_artifact_path}")

# ----------------------------
# Hugging Face Upload
# ----------------------------
api = HfApi()
hf_token = HfFolder.get_token()

# Upload model
try:
    api.repo_info(model_repo)
except:
    api.create_repo(model_repo, repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=model_artifact_path,
    path_in_repo="best_tourism_model_v1.joblib",   # 👈 consistent naming
    repo_id=model_repo,
    repo_type="model",
    token=hf_token,
)
print(f"Model uploaded to Hugging Face repo '{model_repo}'.")

# Ensure dataset repo exists
try:
    api.repo_info(dataset_repo, repo_type="dataset")
except:
    api.create_repo(dataset_repo, repo_type="dataset", private=False)
print(f"Dataset repo '{dataset_repo}' verified.")
