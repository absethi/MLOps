import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
from huggingface_hub import HfApi, HfFolder, RepositoryNotFoundError

# -----------------------------
# Configuration
# -----------------------------
dataset_path = "data/tourism.csv"
model_output_path = "artifacts/tourism_xgb_model.pkl"
dataset_repo = "absethi1894/Visit_with_Us"  # HF dataset
model_repo = "absethi1894/MLOps"            # HF model repo

# -----------------------------
# Load dataset
# -----------------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Dataset loaded successfully.\nColumns: {list(df.columns)}")

# -----------------------------
# Select target column
# -----------------------------
target_column = "ProdTaken"
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")
y = df[target_column]
X = df.drop(columns=[target_column])

# -----------------------------
# Encode categorical columns
# -----------------------------
cat_columns = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# -----------------------------
# Train-test split
# -----------------------------
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # fallback if stratify fails due to class imbalance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# -----------------------------
# Start MLflow experiment
# -----------------------------
mlflow.set_experiment("tourism-mlops-training-experiment")

# -----------------------------
# Train model
# -----------------------------
print("Starting model training...")
model = xgb.XGBClassifier(
    objective="multi:softprob" if len(y.unique()) > 2 else "binary:logistic",
    enable_categorical=True,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)
print("Model training completed.")

# -----------------------------
# Save model
# -----------------------------
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
import joblib
joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")

# -----------------------------
# Upload model to Hugging Face Hub
# -----------------------------
api = HfApi()
token = HfFolder.get_token()
if token is None:
    print("No HF token found. Skipping HF upload.")
else:
    try:
        api.repo_info(repo_id=model_repo)
        print(f"Model repo '{model_repo}' exists. Uploading model...")
    except RepositoryNotFoundError:
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
