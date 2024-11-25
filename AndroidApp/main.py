from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the model and crop data
try:
    model = joblib.load("crop_recommendation_model.pkl")
except FileNotFoundError:
    model = None

try:
    crop_data = pd.read_csv("crop_data.csv")
except FileNotFoundError:
    crop_data = None


# Pydantic model for soil data
class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}


@app.post("/recommend")
def recommend_crop(soil_data: SoilData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found.")
    
    input_data = np.array([[soil_data.N, soil_data.P, soil_data.K,
                            soil_data.temperature, soil_data.humidity,
                            soil_data.ph, soil_data.rainfall]])
    try:
        recommended_crop = model.predict(input_data)[0]
        return {"recommended_crop": recommended_crop}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_parameters(soil_data: SoilData, label: str):
    if crop_data is None:
        raise HTTPException(
            status_code=500, 
            detail="Crop data file not found."
        )
    
    # Ensure required columns exist
    required_columns = ["label", "N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    missing_columns = [col for col in required_columns if col not in crop_data.columns]
    
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"The following required columns are missing in the dataset: {', '.join(missing_columns)}"
        )
    
    # Filter crop data for the given crop_name
    filtered_crop = crop_data[crop_data["label"] == label]
    if filtered_crop.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Crop '{label}' not found in the dataset."
        )
    
    try:
        crop_info = filtered_crop.iloc[0]
        comparison = {}

        # Compare parameters
        for param in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            if param in soil_data.dict():
                difference = soil_data.dict()[param] - crop_info[param]
                comparison[param] = {
                    "status": "Good" if difference == 0 else ("Increase" if difference < 0 else "Decrease"),
                    "amount": abs(difference)
                }
            else:
                comparison[param] = {"status": "Not provided", "amount": None}

        return {"comparison": comparison}

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )


# Optional favicon endpoint to handle requests for favicon.ico
@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon available."}
