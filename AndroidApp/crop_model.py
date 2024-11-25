import joblib
import numpy as np

MODEL_PATH = "crop_recommendation_model.pkl"

def load_crop_model(model_path=MODEL_PATH):
    """
    Loads the crop recommendation model from the given file path.
    """
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the file exists.")

def recommend_crops(soil_data, model):
    """
    Predicts the crop based on soil data and a loaded model.
    
    Args:
    - soil_data (dict): Dictionary containing soil parameters.
    - model: Pre-trained crop recommendation model.

    Returns:
    - str: Recommended crop name.
    """
    try:
        # Ensure required keys are present
        required_keys = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
        missing_keys = [key for key in required_keys if key not in soil_data]
        if missing_keys:
            raise KeyError(f"Missing soil parameters: {', '.join(missing_keys)}")

        features = np.array([[soil_data[key] for key in required_keys]])
        recommended_crop = model.predict(features)[0]
        return recommended_crop
    except Exception as e:
        raise ValueError(f"Error in recommending crops: {str(e)}")
