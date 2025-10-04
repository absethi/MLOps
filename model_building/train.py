import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

# Paths
Xtrain_path = "data/Xtrain.csv"
ytrain_path = "data/ytrain.csv"

# Check if files exist
for p in [Xtrain_path, ytrain_path]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"File not found: {p}")

# Load data
Xtrain = pd.read_csv(Xtrain_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # .squeeze() to get Series if CSV has single column

# Start MLflow experiment
mlflow.set_experiment("tourism-mlops-training-experiment")

with mlflow.start_run():
    # Define model
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }

    # Automatically adjust CV folds based on smallest class
    min_class_size = ytrain.value_counts().min()
    cv_folds = min(3, min_class_size)
    cv_strategy = StratifiedKFold(n_splits=cv_folds)

    print(f"Using {cv_folds}-fold Stratified CV (smallest class has {min_class_size} samples)")

    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Optionally evaluate on training set (or separate test set if available)
    train_preds = best_model.predict(Xtrain)
    train_acc = accuracy_score(ytrain, train_preds)
    mlflow.log_metric("train_accuracy", train_acc)

    # Save model locally
    os.makedirs("artifacts", exist_ok=True)
    model_file = "artifacts/tourism_xgb_model.pkl"
    import joblib
    joblib.dump(best_model, model_file)
    print(f"Model saved at {model_file}")
