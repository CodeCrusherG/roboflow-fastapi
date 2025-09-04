from fastapi import FastAPI, UploadFile
from inference_sdk import InferenceHTTPClient
import base64

app = FastAPI()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="61pbg9qIAf5hCUDgNP06"  # you can move this to env var later
)

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    image_base64 = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
    
    result = client.run_workflow(
        workspace_name="xyz-paco8",
        workflow_id="detect-count-and-visualize",
        images={"image": image_base64},
        use_cache=True
    )
    return result