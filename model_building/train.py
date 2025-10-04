# -----------------------------
# train.py - Model Training
# -----------------------------

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

# HF Token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi(token=HF_TOKEN)

# -----------------------------
# Load dataset splits
# -----------------------------
Xtrain_path = "data/Xtrain.csv"
Xtest_path = "data/Xtest.csv"
ytrain_path = "data/ytrain.csv"
ytest_path = "data/ytest.csv"

for p in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# -----------------------------
# Feature Groups
# -----------------------------
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'Gender', 'MaritalStatus', 'Designation', 'Occupation', 'ProductPitched'
]

# -----------------------------
# Class imbalance handling
# -----------------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__colsample_bytree': [0.5, 0.7],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Training with MLflow logging
# -----------------------------
with mlflow.start_run():
    # Determine min class count for CV splits
    min_class_count = min(ytrain.value_counts())
    cv_splits = min(5, min_class_count)
    if cv_splits < 2:
        raise ValueError("Not enough samples in the smallest class for cross-validation.")

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv_splits, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions
    classification_threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save model locally
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/tourism_xgb_model.pkl"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally and logged to MLflow: {model_path}")

# -----------------------------
# Upload trained model to Hugging Face
# -----------------------------
repo_id = "absethi1894/MLOps"
try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="artifacts/tourism_xgb_model.pkl",
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN
)
print("Trained model uploaded to Hugging Face successfully.")
