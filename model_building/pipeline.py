import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from huggingface_hub import HfApi, HfFolder

# ----------------------------
# Configuration
# ----------------------------
dataset_path = "data/tourism.csv"
dataset_repo = "absethi1894/Visit_with_Us"
model_repo = "absethi1894/MLOps"
space_repo = "absethi1894/Visit_with_Us"   # same repo where app.py runs
model_artifact_path = "tourism_xgb_model.pkl"

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(dataset_path)
print("✅ Dataset loaded successfully")
print("Columns:", df.columns.tolist())

# ----------------------------
# Target and Features
# ----------------------------
target_col = "ProdTaken"
y = df[target_col]
X = df.drop(columns=[target_col, "CustomerID", "Unnamed: 0"], errors="ignore")

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# Encode target
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

num_classes = len(np.unique(y))
print(f"✅ Target '{target_col}' has {num_classes} unique classes")

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# ----------------------------
# Train Model
# ----------------------------
model = xgb.XGBClassifier(
    objective="multi:softprob" if num_classes > 2 else "binary:logistic",
    eval_metric="mlogloss" if num_classes > 2 else "logloss",
    enable_categorical=True,
    use_label_encoder=False,
    num_class=num_classes if num_classes > 2 else None
)

print("🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Model training completed")

# ----------------------------
# Save Model
# ----------------------------
model.save_model(model_artifact_path)
print(f"✅ Model saved as {model_artifact_path}")

# ----------------------------
# Upload to Hugging Face
# ----------------------------
api = HfApi()
hf_token = HfFolder.get_token()

# Ensure model repo exists
try:
    api.repo_info(model_repo, repo_type="model")
except:
    api.create_repo(model_repo, repo_type="model", private=False, token=hf_token)

# Upload to model repo
api.upload_file(
    path_or_fileobj=model_artifact_path,
    path_in_repo=model_artifact_path,
    repo_id=model_repo,
    repo_type="model",
    token=hf_token,
)
print(f"✅ Model uploaded to model repo: {model_repo}")

# Also upload to Space repo (so app.py can load it locally)
api.upload_file(
    path_or_fileobj=model_artifact_path,
    path_in_repo=model_artifact_path,
    repo_id=space_repo,
    repo_type="space",
    token=hf_token,
)
print(f"✅ Model uploaded to Space repo: {space_repo}")
