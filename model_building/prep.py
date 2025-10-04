# model_building/prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
# Dataset path
# -----------------------------
DATASET_PATH = "data/tourism.csv"

# Download dataset if missing
if not os.path.isfile(DATASET_PATH):
    import urllib.request
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    url = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"
    urllib.request.urlretrieve(url, DATASET_PATH)

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Columns in dataset:", df.columns.tolist())

# Drop unique identifier
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode categorical 'Type' column if exists
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in dataset.")

# Target column
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found. Using first column as target.")
    target_col = df.columns[0]

# Split features and target
X = df.drop(columns=[target_col], errors='ignore')
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train-test splits saved locally.")

# Upload splits to Hugging Face Hub
repo_id = "absethi1894/Visit_with_Us"
try:
    api.repo_info(repo_id=repo_id)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"Repo '{repo_id}' created.")

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        token=hf_token
    )
print("Train-test splits uploaded to Hugging Face Hub.")
