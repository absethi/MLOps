# model_building/train.py

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Load dataset splits (local fallback to HF)
# -----------------------------
local_files_exist = all(os.path.exists(f) for f in ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"])

if local_files_exist:
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    print("Loaded dataset from local CSVs.")
else:
    base_url = "https://huggingface.co/datasets/absethi1894/Visit_with_Us/resolve/main/"
    Xtrain = pd.read_csv(base_url + "Xtrain.csv")
    Xtest = pd.read_csv(base_url + "Xtest.csv")
    ytrain = pd.read_csv(base_url + "ytrain.csv").squeeze()
    ytest = pd.read_csv(base_url + "ytest.csv").squeeze()
    print("Loaded dataset from Hugging Face dataset repo.")

# -----------------------------
# Feature groups
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
# Handle class imbalance
# -----------------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# -----------------------------
# Preprocessing pipeline
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
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__colsample_bytree': [0.5, 0.7, 0.9],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Training with MLflow logging
# -----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Threshold tuning
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:,1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:,1] >= threshold).astype(int)

    # Classification reports
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

    # Save model locally & MLflow
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally & logged to MLflow: {model_path}")

    # -----------------------------
    # Upload to Hugging Face Hub
    # -----------------------------
    repo_id = "absethi1894/churn-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
