# pipeline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import mlflow
from huggingface_hub import HfApi

# ---------------------------
# SETTINGS
# ---------------------------
DATA_PATH = "data/tourism.csv"
TARGET_COL = "Failure"  # fallback to first column if missing
TEST_SIZE = 0.2
CV_FOLDS = 3
HF_DATASET_REPO = "absethi1894/Visit_with_Us"
HF_MODEL_REPO = "absethi1894/MLOps"

# ---------------------------
# LOAD DATASET
# ---------------------------
if not os.path.exists("data"):
    os.makedirs("data")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

if TARGET_COL not in df.columns:
    print(f"Warning: Target column '{TARGET_COL}' not found. Using first column instead.")
    TARGET_COL = df.columns[0]

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ---------------------------
# TRAIN-TEST SPLIT WITH CLASS CHECK
# ---------------------------
class_counts = y.value_counts()
min_class_count = class_counts.min()

if min_class_count < 2:
    print(f"Warning: smallest class has {min_class_count} sample(s). Using simple train/test split instead of stratified CV.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=42
    )
    use_cv = False
else:
    if CV_FOLDS > min_class_count:
        CV_FOLDS = min_class_count
        print(f"Adjusted CV folds to {CV_FOLDS} based on class sizes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    use_cv = True

# ---------------------------
# START MLflow EXPERIMENT
# ---------------------------
mlflow.set_experiment("tourism-mlops-training-experiment")

with mlflow.start_run():
    print("Starting model training...")

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    if use_cv:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            print(f"Fold accuracy: {acc:.4f}")
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Test accuracy: {acc:.4f}")

    # Save model locally
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    model_file = "artifacts/tourism_xgb_model.pkl"
    pd.to_pickle(model, model_file)
    print(f"Model saved to {model_file}")

# ---------------------------
# UPLOAD TO HUGGING FACE HUB
# ---------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi()

# Upload dataset
try:
    api.repo_info(HF_DATASET_REPO, repo_type="dataset")
except Exception:
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=False)

api.upload_file(
    path_or_fileobj=DATA_PATH,
    path_in_repo="data/tourism.csv",
    repo_id=HF_DATASET_REPO,
    token=HF_TOKEN
)
print(f"Dataset uploaded to Hugging Face: {HF_DATASET_REPO}")

# Upload model
try:
    api.repo_info(HF_MODEL_REPO, repo_type="model")
except Exception:
    api.create_repo(HF_MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="artifacts/tourism_xgb_model.pkl",
    repo_id=HF_MODEL_REPO,
    token=HF_TOKEN
)
print(f"Model uploaded to Hugging Face: {HF_MODEL_REPO}")

print("Pipeline completed successfully.")
