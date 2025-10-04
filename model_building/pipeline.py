import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
from huggingface_hub import HfApi, HfFolder

# -------------------
# Paths & Repos
# -------------------
dataset_path = "data/tourism.csv"
model_artifact_path = "artifacts/tourism_xgb_model.pkl"

model_repo = "absethi1894/MLOps"
dataset_repo = "absethi1894/Visit_with_Us"

os.makedirs("artifacts", exist_ok=True)

# -------------------
# Load Dataset
# -------------------
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# Select target column
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------
# Encode categorical columns
# -------------------
categorical_cols = X.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")

# Encode target if it's categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# -------------------
# Train-Test Split
# -------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# -------------------
# Train Model
# -------------------
mlflow.set_experiment("tourism-mlops-training-experiment")
mlflow.start_run()

model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    enable_categorical=True,
    use_label_encoder=False
)

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training completed.")

# Save locally
model.save_model(model_artifact_path)
print(f"Model saved to {model_artifact_path}")

mlflow.xgboost.log_model(model, artifact_path="tourism_xgb_model")
mlflow.end_run()

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
    print(f"Dataset repo '{dataset_repo}' exists. Skipping creation.")
except Exception:
    api.create_repo(dataset_repo, repo_type="dataset", private=False)
    print(f"Dataset repo '{dataset_repo}' created.")
