# app.py
from fastapi import FastAPI, UploadFile, File
import requests
import uvicorn

app = FastAPI()

# Replace these with your Roboflow model info
ROBOFLOW_MODEL_NAME = "yellow_leafs-pr0v9/1"   # e.g., "yellow_leafs"
ROBOFLOW_MODEL_VERSION = "v1"   # e.g., "1"
ROBOFLOW_API_KEY = "61pbg9qIAf5hCUDgNP06"         # Your Roboflow private API key

ROBofLOW_API_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_NAME}/{ROBOFLOW_MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}"

@app.get("/")
def home():
    return {"message": "Roboflow Render API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_bytes = await file.read()

        # Forward the image to Roboflow API
        response = requests.post(
            ROBofLOW_API_URL,
            files={"file": (file.filename, image_bytes, file.content_type)}
        )

        # Return the JSON response from Roboflow
        return response.json()

    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}

# Optional: run locally with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)