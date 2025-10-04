# model_building/train.py

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# Check HF_TOKEN
# -----------------------------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi(token=hf_token)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

# -----------------------------
# Load dataset splits
# -----------------------------
dataset_repo = "absethi1894/Visit_with_Us"
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    if not os.path.isfile(file):
        url = f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/{file}"
        pd.read_csv(url).to_csv(file, index=False)
        print(f"Downloaded {file} from HF Hub.")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# -----------------------------
# Features
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

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__colsample_bytree': [0.5, 0.7, 0.9],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5],
}

pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Training
# -----------------------------
with mlflow.start_run():
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_
    mlflow.log_params(grid.best_params_)

    # Threshold tuning
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

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

    # Save and upload model
    model_path = "tourism_project/best_tourism_model_v1.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally: {model_path}")

    # Upload to HF Hub
    repo_id = "absethi1894/churn-model"
    try:
        api.repo_info(repo_id=repo_id)
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type="model", private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        token=hf_token
    )
    print("Model uploaded to Hugging Face Hub successfully.")
