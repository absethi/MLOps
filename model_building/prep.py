# model_building/prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo, RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

# -----------------------------
# Hugging Face authentication
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi(token=HF_TOKEN)

# -----------------------------
# Load dataset
# -----------------------------
DATASET_PATH = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Columns in dataset:", list(df.columns))

# -----------------------------
# Drop unique identifier
# -----------------------------
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# -----------------------------
# Encode categorical column 'Type' if exists
# -----------------------------
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in dataset.")

# -----------------------------
# Handle target column
# -----------------------------
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found. Using first column as target instead.")
    target_col = df.columns[0]

# -----------------------------
# Split into features and target
# -----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Train-test split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save locally
Xtrain_path = "Xtrain.csv"
Xtest_path = "Xtest.csv"
ytrain_path = "ytrain.csv"
ytest_path = "ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)
print("Train-test splits saved locally.")

# -----------------------------
# Upload splits to Hugging Face
# -----------------------------
repo_id = "absethi1894/Visit_with_Us"

# Ensure dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"Dataset repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        print(f"Dataset repo '{repo_id}' created.")
    except HfHubHTTPError as e:
        if "409" in str(e):
            print(f"Dataset repo '{repo_id}' already exists.")
        else:
            raise e

# Upload each file
for file_path in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        token=HF_TOKEN
    )
    print(f"Uploaded {file_path} to HF repo '{repo_id}'.")

print("All files uploaded successfully.")
