
from fastapi import FastAPI, File, UploadFile
import logging
import requests

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)

# --- Initialize FastAPI ---
app = FastAPI()

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"status": "API is running"}

# --- Predict Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logging.info("🔄 Received /predict request")

        # Save uploaded file locally
        contents = await file.read()
        file_path = "temp.jpg"
        with open(file_path, "wb") as f:
            f.write(contents)
        logging.info(f"✅ File saved as {file_path}")

        # Call Roboflow Workflow API
        ROBFLOW_API_URL = "https://serverless.roboflow.com"
        API_KEY = "61pbg9qIAf5hCUDgNP06"  # <-- Replace with your actual key

        with open(file_path, "rb") as image_file:
            response = requests.post(
                ROBFLOW_API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": API_KEY,
                    "inputs": {
                        "image": {
                            "type": "base64",
                            "value": contents.decode("latin-1")  # send base64-compatible bytes
                        }
                    }
                }
            )

        if response.status_code != 200:
            logging.error(f"❌ Roboflow API returned {response.status_code}: {response.text}")
            return {"error": "Roboflow API call failed", "details": response.text}

        result = response.json()
        logging.info(f"✅ Prediction result: {result}")

        return {"predictions": result}

    except Exception as e:
        logging.error(f"❌ Error in /predict: {str(e)}")
        return {"error": str(e)}