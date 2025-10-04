# model_building/prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo

# Ensure HF_TOKEN is set
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

# Paths
DATASET_URL = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_URL)
print("Dataset loaded successfully.")
print("Columns in dataset:", list(df.columns))

# Drop unique identifier if exists
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode categorical 'Type' column if exists
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in dataset.")

# Set target column
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found. Using first column as target instead.")
    target_col = df.columns[0]

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits locally
Xtrain.to_csv(os.path.join(DATA_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(DATA_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(DATA_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(DATA_DIR, "ytest.csv"), index=False)
print("Train-test splits saved locally.")

# -----------------------------
# Upload splits to Hugging Face Dataset
# -----------------------------
api = HfApi(token=HF_TOKEN)
dataset_repo_id = "absethi1894/Visit_with_Us"

# Ensure repo exists
try:
    api.repo_info(repo_id=dataset_repo_id, repo_type="dataset")
    print(f"Dataset repo '{dataset_repo_id}' already exists.")
except:
    print(f"Dataset repo '{dataset_repo_id}' not found. Creating...")
    create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=False)
    print(f"Dataset repo '{dataset_repo_id}' created.")

# Upload files
for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    local_path = os.path.join(DATA_DIR, file_name)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=file_name,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
print("All dataset files uploaded to Hugging Face Hub.")
