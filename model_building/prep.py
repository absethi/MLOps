# model_building/prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path
DATASET_PATH = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Columns in dataset: {list(df.columns)}")

# Drop unique identifier
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode 'Type' column if exists
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in the dataset.")

# Target column
target_col = 'Failure'
if target_col not in df.columns:
    print(f"Warning: Target column '{target_col}' not found in dataset. Using first column as target instead.")
    target_col = df.columns[0]  # fallback to first column

# Split into features and target
X = df.drop(columns=[target_col], errors='ignore')
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train-test splits saved locally.")

# Upload to Hugging Face dataset repo
repo_id = "absethi1894/Visit_with_Us"
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
    )
print("All files uploaded to Hugging Face dataset repo.")
