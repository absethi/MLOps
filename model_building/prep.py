# -----------------------------
# prep.py
# -----------------------------

# for data manipulation
import pandas as pd
import os
# for data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# Constants
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

api = HfApi(token=HF_TOKEN)

DATASET_PATH = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Columns in dataset: {list(df.columns)}")

# Drop unique identifier if exists
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode 'Type' column if exists
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in the dataset.")

# Set target column
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found in dataset. Using first column as target instead.")
    target_col = df.columns[0]

# Split features and target
X = df.drop(columns=[target_col], errors='ignore')
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train-test splits saved locally.")

# -----------------------------
# Hugging Face Dataset Upload
# -----------------------------
repo_id = "absethi1894/Visit_with_Us"
repo_type = "dataset"

# Ensure repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Upload files
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {file_path} to Hugging Face dataset repo '{repo_id}'.")
