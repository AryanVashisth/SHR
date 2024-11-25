# fertilizer_recommendation.py
import pandas as pd
import joblib

def load_fertilizer_model():
    return joblib.load('fertilizer_recommendation_model.pkl')

def recommend_fertilizer(soil_data, model):
    df = pd.DataFrame([soil_data])
    recommendation = model.predict(df)[0]
    return recommendation
