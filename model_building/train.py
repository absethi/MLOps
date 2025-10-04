# train.py
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi(token=HF_TOKEN)

# -----------------------------
# Load dataset splits from HF dataset
# -----------------------------
dataset_repo = "absethi1894/Visit_with_Us"
Xtrain_path = f"hf://datasets/{dataset_repo}/Xtrain.csv"
Xtest_path = f"hf://datasets/{dataset_repo}/Xtest.csv"
ytrain_path = f"hf://datasets/{dataset_repo}/ytrain.csv"
ytest_path = f"hf://datasets/{dataset_repo}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

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
# Class imbalance
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
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__colsample_bytree': [0.5, 0.7],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Training
# -----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    classification_threshold = 0.45
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)

    # Log metrics
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
    model_file = "tourism_xgb_model.joblib"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file, artifact_path="model")
    print(f"Model saved locally and logged to MLflow: {model_file}")

# -----------------------------
# Upload model to HF model repo
# -----------------------------
model_repo = "absethi1894/MLOps"
try:
    api.repo_info(repo_id=model_repo, repo_type="model")
    print(f"Model repo '{model_repo}' exists. Uploading model...")
except Exception:
    print(f"Model repo '{model_repo}' not found. Creating...")
    create_repo(repo_id=model_repo, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo=model_file,
    repo_id=model_repo,
    repo_type="model"
)
print(f"Model uploaded to Hugging Face repo '{model_repo}'.")
