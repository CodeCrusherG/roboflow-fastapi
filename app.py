from fastapi import FastAPI, UploadFile, File
from inference_sdk import InferenceHTTPClient
import tempfile, shutil

app = FastAPI()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="61pbg9qIAf5hCUDgNP06"
)

@app.get("/")
def home():
    return {"message": "Roboflow local API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = client.run_workflow(
            workspace_name="xyz-paco8",
            workflow_id="detect-count-and-visualize",
            images={"image": tmp_path},
            use_cache=True
        )
        return result

    except Exception as e:
        return {"error": str(e)}