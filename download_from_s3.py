import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

os.makedirs("models", exist_ok=True)

local_path = "models/churn_model.joblib"

if not os.path.exists(local_path):
    print("Downloading model from S3...")
    s3.download_file(
        BUCKET_NAME,
        "models/churn_model.joblib",
        local_path
    )
    print("Model downloaded successfully!")
else:
    print("Model already exists locally.")