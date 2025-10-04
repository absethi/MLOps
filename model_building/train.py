# model_building/train.py

import os
import pandas as pd
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
from huggingface_hub import HfApi, create_repo

# Ensure HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

# Paths
DATA_DIR = "data"
Xtrain_path = os.path.join(DATA_DIR, "Xtrain.csv")
Xtest_path = os.path.join(DATA_DIR, "Xtest.csv")
ytrain_path = os.path.join(DATA_DIR, "ytrain.csv")
ytest_path = os.path.join(DATA_DIR, "ytest.csv")

# Verify files exist
for p in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"File not found: {p}")

# Load data
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# Feature lists
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
categorical_features = [
    'Gender', 'MaritalStatus', 'Designation', 'Occupation', 'ProductPitched'
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
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
    'xgbclassifier__colsample_bytree': [0.7, 0.9],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Training with MLflow
with mlflow.start_run():
    n_splits = min(3, min(ytrain.value_counts()))
    if n_splits < 2:
        raise ValueError("Not enough data for cross-validation")
    
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=n_splits, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predict
    classification_threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:,1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:,1] >= classification_threshold).astype(int)

    # Reports
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
model_path = "artifacts/tourism_xgb_model.joblib"
joblib.dump(best_model, model_path)
print(f"Model saved locally: {model_path}")

# Upload model to Hugging Face Hub
api = HfApi(token=HF_TOKEN)
model_repo_id = "absethi1894/MLOps"

try:
    api.repo_info(repo_id=model_repo_id)
    print(f"Model repo '{model_repo_id}' exists.")
except:
    print(f"Creating model repo '{model_repo_id}'...")
    create_repo(repo_id=model_repo_id, repo_type="model", private=False)
    print(f"Repo '{model_repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=os.path.basename(model_path),
    repo_id=model_repo_id,
    repo_type="model",
    token=HF_TOKEN
)
print("Model uploaded to Hugging Face Hub.")
