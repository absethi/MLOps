import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
from huggingface_hub import HfApi

# ----------------------------
# Load Dataset
# ----------------------------
dataset_path = "data/tourism.csv"
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# ----------------------------
# Target column
# ----------------------------
target_col = "Failure" if "Failure" in df.columns else "ProdTaken"
print(f"Target column '{target_col}' selected.")

X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------
# Handle categorical features
# ----------------------------
categorical_cols = ["TypeofContact", "Occupation", "Gender", "ProductPitched", 
                    "MaritalStatus", "Designation"]

for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype("category")

# ----------------------------
# Train-test split
# ----------------------------
if y.value_counts().min() < 2:
    print(f"Warning: smallest class has {y.value_counts().min()} sample(s). Using simple train/test split.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    from sklearn.model_selection import StratifiedKFold
    cv_folds = min(3, y.value_counts().min())
    if cv_folds < 2:
        print("Not enough samples for stratified CV, using train_test_split instead.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        # fallback to simple split for now
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Train XGBoost Model
# ----------------------------
print("Starting model training...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    enable_categorical=True  # important for categorical columns
)
model.fit(X_train, y_train)
print("Model training completed.")

# ----------------------------
# Save Model
# ----------------------------
os.makedirs("artifacts", exist_ok=True)
model_file = "artifacts/tourism_xgb_model.pkl"
import joblib
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")

# ----------------------------
# MLflow Tracking
# ----------------------------
mlflow.set_experiment("tourism-mlops-training-experiment")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.sklearn.log_model(model, "xgb_model")

# ----------------------------
# Upload Dataset & Model to Hugging Face Hub
# ----------------------------
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi()
dataset_repo = "absethi1894/Visit_with_Us"
model_repo = "absethi1894/MLOps"

# Dataset upload
try:
    api.repo_info(repo_id=dataset_repo)
except:
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)

api.upload_file(
    path_or_fileobj=dataset_path,
    path_in_repo="data/tourism.csv",
    repo_id=dataset_repo,
    token=hf_token
)

# Model upload
try:
    api.repo_info(repo_id=model_repo)
except:
    api.create_repo(repo_id=model_repo, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="artifacts/tourism_xgb_model.pkl",
    repo_id=model_repo,
    token=hf_token
)

print("Dataset and model uploaded to Hugging Face Hub successfully.")
