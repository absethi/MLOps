import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
from huggingface_hub import HfApi, HfFolder

# -------------------
# Paths and repos
# -------------------
dataset_path = "data/tourism.csv"   # updated dataset name
dataset_repo = "absethi1894/Visit_with_Us"
model_repo = "absethi1894/MLOps"
model_artifact_path = "artifacts/tourism_xgb_model.pkl"

# -------------------
# Load Dataset
# -------------------
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# -------------------
# Target selection
# -------------------
target_column = "ProdTaken"  # using ProdTaken as target
X = df.drop(columns=[target_column, "Unnamed: 0", "CustomerID"])
y = df[target_column]

# Convert object columns to category
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# -------------------
# Train-test split
# -------------------
if y.value_counts().min() < 2:
    # Too few samples for CV
    print(f"Warning: smallest class has {y.value_counts().min()} sample(s). Using simple train/test split.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # Stratified split if possible
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# -------------------
# Model Training
# -------------------
mlflow.set_experiment("tourism-mlops-training-experiment")
print("Starting model training...")

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True  # important for category columns
)
model.fit(X_train, y_train)

print("Model training completed.")

# -------------------
# Save model locally
# -------------------
os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
import joblib
joblib.dump(model, model_artifact_path)
print(f"Model saved to {model_artifact_path}")

# -------------------
# Hugging Face Hub Upload
# -------------------
api = HfApi()
hf_token = HfFolder.get_token()

# Upload model
try:
    api.repo_info(model_repo)
    print(f"Model repo '{model_repo}' exists. Uploading model...")
except Exception:
    api.create_repo(model_repo, repo_type="model", private=False)
    print(f"Model repo '{model_repo}' created.")

api.upload_file(
    path_or_fileobj=model_artifact_path,
    path_in_repo="tourism_xgb_model.pkl",
    repo_id=model_repo,
    token=hf_token,
)

# Upload dataset
try:
    api.repo_info(dataset_repo)
    print(f"Dataset repo '{dataset_repo}' exists.")
except Exception:
    api.create_repo(dataset_repo, repo_type="dataset", private=False)
    print(f"Dataset repo '{dataset_repo}' created.")

print("Pipeline completed successfully.")
