import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from huggingface_hub import HfApi

# Paths
data_dir = "data"
Xtrain_path = os.path.join(data_dir, "Xtrain.csv")
ytrain_path = os.path.join(data_dir, "ytrain.csv")
model_file = "artifacts/tourism_xgb_model.pkl"

os.makedirs("artifacts", exist_ok=True)

# Load data
if not os.path.exists(Xtrain_path) or not os.path.exists(ytrain_path):
    raise FileNotFoundError(f"Missing training data: {Xtrain_path} or {ytrain_path}")

Xtrain = pd.read_csv(Xtrain_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # convert to Series if single column

# Determine CV strategy based on smallest class size
min_class_size = ytrain.value_counts().min()
if min_class_size < 2:
    print(f"Warning: smallest class has {min_class_size} sample(s). Using simple train/test split instead of CV.")
    X_train, X_val, y_train, y_val = train_test_split(
        Xtrain, ytrain, test_size=0.2, stratify=ytrain, random_state=42
    )
    use_cv = False
else:
    cv_folds = min(3, min_class_size)
    print(f"Using {cv_folds}-fold Stratified CV (smallest class has {min_class_size} samples)")
    cv_strategy = StratifiedKFold(n_splits=cv_folds)
    use_cv = True

# Define model and hyperparameters
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1]
}

# MLflow experiment
mlflow.set_experiment("tourism-mlops-training-experiment")

with mlflow.start_run() as run:
    if use_cv:
        grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)
        best_model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)
    else:
        model.fit(X_train, y_train)
        best_model = model

    # Save trained model
    import joblib
    joblib.dump(best_model, model_file)
    print(f"Model saved to {model_file}")
    mlflow.log_artifact(model_file)

# Upload to Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    api = HfApi()
    repo_id = "absethi1894/MLOps"
    if os.path.isfile(model_file):
        api.upload_file(
            path_or_fileobj=model_file,
            path_in_repo="artifacts/tourism_xgb_model.pkl",
            repo_id=repo_id,
            token=hf_token
        )
        print(f"Model uploaded to Hugging Face: {repo_id}")
else:
    print("HF_TOKEN not set. Skipping Hugging Face upload.")
