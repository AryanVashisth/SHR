# crop_recommendation.py
import joblib

def load_crop_model():
    return joblib.load('crop_recommendation_model.pkl')

def recommend_crops(soil_data, model):
    features = [[
        soil_data["Nitrogen"], soil_data["Phosphorus"], soil_data["Potassium"],
        soil_data["Temperature"], soil_data["Humidity"], soil_data["pH"], soil_data["Rainfall"]
    ]]
    recommended_crop = model.predict(features)[0]
    return recommended_crop
