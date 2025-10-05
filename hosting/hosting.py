from huggingface_hub import HfApi
import os

# Authenticate with your HF token from GitHub Secrets
api = HfApi(token=os.getenv("HF_TOKEN"))

# Correct relative path to your deployment folder
deployment_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deployment")

print(f"Uploading deployment files from: {deployment_path}")

# Upload deployment files to your Hugging Face Space
api.upload_folder(
    folder_path=deployment_path,           # local folder with app.py & Dockerfile
    repo_id="absethi1894/Visit_with_Us",   # your HF Space repo
    repo_type="space",                     # deploying to a Space
    path_in_repo="",                       # root of the repo
    commit_message="Update deployment files via CI/CD"
)

