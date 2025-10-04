# model_building/train.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo

# -----------------------------
# MLflow setup (local file-based tracking)
# -----------------------------
mlruns_path = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.set_experiment("tourism-mlops-training-experiment")

# -----------------------------
# Load dataset splits
# -----------------------------
data_dir = "data"
Xtrain_path = os.path.join(data_dir, "Xtrain.csv")
Xtest_path = os.path.join(data_dir, "Xtest.csv")
ytrain_path = os.path.join(data_dir, "ytrain.csv")
ytest_path = os.path.join(data_dir, "ytest.csv")

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
# Handle class imbalance
# -----------------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# -----------------------------
# Model + hyperparameter grid
# -----------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__colsample_bytree': [0.7, 0.9],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Training with MLflow logging
# -----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions with threshold
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    # Classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_params(grid_search.best_params_)
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

    # Save model artifact
    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    model_path = os.path.join(artifact_dir, "tourism_xgb_model.pkl")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally & logged to MLflow: {model_path}")

# -----------------------------
# Upload model to Hugging Face
# -----------------------------
api = HfApi()
repo_id = "absethi1894/MLOps"

try:
    api.repo_info(repo_id=repo_id)
    print(f"Repo '{repo_id}' exists, using it.")
except:
    create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"Repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="artifacts/tourism_xgb_model.pkl",
    repo_id=repo_id,
    token=os.environ.get("HF_TOKEN")
)
print("Model uploaded to Hugging Face Hub successfully!")
