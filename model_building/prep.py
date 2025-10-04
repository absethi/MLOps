# for data manipulation
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path
DATASET_PATH = "https://raw.githubusercontent.com/absethi/MLOps/main/data/tourism.csv"

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset from {DATASET_PATH}: {e}")

# Drop the unique identifier column if it exists
df.drop(columns=['UDI'], inplace=True, errors='ignore')

# Encode categorical columns if they exist
label_encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = label_encoder.fit_transform(df['Type'])
else:
    print("Warning: 'Type' column not found in the dataset.")

# Define target column
target_col = 'Failure'

# Split into X (features) and y (target)
if target_col in df.columns:
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]
else:
    print(f"Warning: target column '{target_col}' not found. Using all columns as features.")
    X = df.copy()
    y = pd.Series()  # empty series

# Perform train-test split only if y is not empty
if not y.empty:
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save split files
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
    
    # Upload to Hugging Face dataset repo
    for file_path in files:
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path.split("/")[-1],  # just the filename
                repo_id="absethi1894/Visit_with_Us",
                repo_type="dataset",
            )
            print(f"Uploaded {file_path} successfully.")
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")
else:
    print("No target column to split. Only features are available.")
