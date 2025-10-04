# model_building/train.py

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
from huggingface_hub import HfApi, create_repo
import mlflow
import os

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

# -----------------------------
# Hugging Face API
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Load dataset splits
# -----------------------------
Xtrain_path = "Xtrain.csv"
Xtest_path = "Xtest.csv"
ytrain_path = "ytrain.csv"
ytest_path = "ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # convert to Series
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
# Preprocessing pipeline
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# -----------------------------
# Base model
# -----------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# -----------------------------
# Hyperparameter grid
# -----------------------------
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__colsample_bytree': [0.5, 0.7, 0.9],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5],
}

# -----------------------------
# Model pipeline
# -----------------------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Cross-validation setup
# -----------------------------
min_class_count = ytrain.value_counts().min()
n_splits = min(5, min_class_count)
if n_splits < 2:
    raise ValueError("Not enough samples in smallest class for CV. Add more data or reduce n_splits.")

if n_splits < 5:
    print(f"Warning: Number of CV splits reduced to {n_splits} due to small class size.")

cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# -----------------------------
# Training with MLflow
# -----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv_strategy, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log CV results
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Threshold tuning
    classification_threshold = 0.45
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Metrics
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

    # Save the model
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/tourism_xgb_model.pkl"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally & logged to MLflow: {model_path}")

# -----------------------------
# Upload to Hugging Face Model Hub
# -----------------------------
repo_id = "absethi1894/MLOps"  # Models repo
try:
    api.repo_info(repo_id=repo_id)
    print(f"Repo '{repo_id}' exists.")
except:
    print(f"Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"Repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_xgb_model.pkl",
    repo_id=repo_id,
    token=os.environ["HF_TOKEN"]
)
print("Model uploaded to Hugging Face Hub successfully.")
