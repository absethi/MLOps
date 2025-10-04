# -----------------------------
# prep.py - Data Preparation
# -----------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo

# Ensure HF_TOKEN is set
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi(token=HF_TOKEN)

# Dataset path
DATASET_PATH = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Columns in dataset:", list(df.columns))

# Drop unique identifier
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode 'Type' column if exists
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in dataset.")

# Define target column
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found. Using first column as target instead.")
    target_col = df.columns[0]

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Create local folder for splits
os.makedirs("data", exist_ok=True)

# Save splits
Xtrain.to_csv("data/Xtrain.csv", index=False)
Xtest.to_csv("data/Xtest.csv", index=False)
ytrain.to_csv("data/ytrain.csv", index=False)
ytest.to_csv("data/ytest.csv", index=False)

print("Train-test splits saved locally in 'data/' folder.")

# -----------------------------
# Upload dataset to Hugging Face
# -----------------------------
repo_id = "absethi1894/Visit_with_Us"
try:
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"Dataset repo '{repo_id}' exists. Using it.")
except:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"Dataset repo '{repo_id}' created.")

for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    path = f"data/{file_name}"
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"data/{file_name}",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
print("All dataset files uploaded to Hugging Face successfully.")
