import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from huggingface_hub import HfApi
import mlflow

# -----------------------
# Config
# -----------------------
dataset_path = "data/dataset.csv"  # Update if needed
dataset_repo = "absethi1894/Visit_with_Us"
model_repo = "absethi1894/MLOps"
model_local_path = "artifacts/tourism_xgb_model.pkl"
target_column = "ProdTaken"  # Default target

os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

if target_column not in df.columns:
    print(f"Warning: Target column '{target_column}' not found. Using first column instead.")
    target_column = df.columns[0]

X = df.drop(columns=[target_column])
y = df[target_column]

# -----------------------
# Handle categorical columns
# -----------------------
categorical_cols = X.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")

# -----------------------
# Train-test split
# -----------------------
if y.value_counts().min() < 2:
    print(f"Warning: smallest class has {y.value_counts().min()} sample(s). Using simple train/test split.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    from sklearn.model_selection import StratifiedKFold
    cv_folds = min(3, y.value_counts().min())
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X, y))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# -----------------------
# Train XGBoost model
# -----------------------
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True
)
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training completed.")

# -----------------------
# Save model locally
# -----------------------
pickle.dump(model, open(model_local_path, "wb"))
print(f"Model saved to {model_local_path}")

# -----------------------
# Upload dataset to HF
# -----------------------
api = HfApi()
try:
    api.repo_info(repo_id=dataset_repo, repo_type="dataset")
    print(f"Dataset repo '{dataset_repo}' already exists.")
except Exception:
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)
    print(f"Created dataset repo '{dataset_repo}'.")

# -----------------------
# Upload model to HF
# -----------------------
try:
    api.repo_info(repo_id=model_repo, repo_type="model")
    print(f"Model repo '{model_repo}' already exists.")
except Exception:
    api.create_repo(repo_id=model_repo, repo_type="model", private=False)
    print(f"Created model repo '{model_repo}'.")

api.upload_file(
    path_or_fileobj=model_local_path,
    path_in_repo=os.path.basename(model_local_path),
    repo_id=model_repo,
    repo_type="model"
)
print(f"Model uploaded to HF model repo '{model_repo}'.")
