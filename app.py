from fastapi import FastAPI, UploadFile, File
from inference_sdk import InferenceHTTPClient
import shutil
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
API_URL = "https://serverless.roboflow.com"
API_KEY = os.environ.get("61pbg9qIAf5hCUDgNP06")  # use env var
WORKSPACE_NAME = "xyz-paco8"
WORKFLOW_ID = "detect-count-and-visualize"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# INITIALIZE CLIENT
# -----------------------------
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

# -----------------------------
# INITIALIZE FASTAPI
# -----------------------------
app = FastAPI(title="Roboflow FastAPI Deployment")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run workflow
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": file_path},
            use_cache=True
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}