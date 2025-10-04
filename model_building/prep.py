# prep.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo

# Hugging Face API
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi(token=HF_TOKEN)

# Load dataset
DATASET_PATH = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Columns in dataset: {list(df.columns)}")

# Drop unique identifier if present
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode 'Type' column if present
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in dataset.")

# Target column
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found. Using first column as target instead.")
    target_col = df.columns[0]

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train-test splits saved locally.")

# Upload to Hugging Face dataset Space
dataset_repo = "absethi1894/Visit_with_Us"
try:
    api.repo_info(repo_id=dataset_repo, repo_type="dataset")
    print(f"Dataset repo '{dataset_repo}' exists. Uploading files...")
except Exception:
    print(f"Dataset repo '{dataset_repo}' not found. Creating...")
    create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)
    print("Repo created.")

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=dataset_repo,
        repo_type="dataset"
    )
    print(f"Uploaded {file} to {dataset_repo}")
