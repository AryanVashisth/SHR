import streamlit as st
import random
import pandas as pd
import joblib
import numpy as np

# Generate synthetic soil data
def generate_synthetic_soil_data():
    return {
        "N": random.uniform(0, 100),   # Nitrogen
        "P": random.uniform(0, 100),   # Phosphorus
        "K": random.uniform(0, 100),   # Potassium
        "ph": random.uniform(5, 8),    # pH value
        "Moisture": random.uniform(0, 100),  # Moisture percentage
        "Temperature": random.uniform(15, 35),  # Temperature in Celsius
        "Humidity": random.uniform(50, 100),    # Humidity percentage
        "Rainfall": random.uniform(100, 300)    # Rainfall in mm
    }

# Assess soil health
def assess_soil_health(soil_data):
    if soil_data["ph"] < 5.5 or soil_data["ph"] > 7.5:
        return "Poor"
    if soil_data["Moisture"] < 20:
        return "Poor"
    if soil_data["N"] < 10 or soil_data["P"] < 10 or soil_data["K"] < 10:
        return "Poor"
    return "Good" if soil_data["Moisture"] > 50 else "Average"

# Load crop recommendation model
def load_crop_model():
    # Assuming model.pkl is in the same directory
    try:
        return joblib.load("crop_recommendation_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' exists in the directory.")
        return None

# Recommend crops based on soil data
def recommend_crops(soil_data, model):
    input_data = np.array([[soil_data["N"], soil_data["P"], soil_data["K"],
                            soil_data["Temperature"], soil_data["Humidity"],
                            soil_data["ph"], soil_data["Rainfall"]]])
    prediction = model.predict(input_data)
    return prediction[0]



def compare_parameters_with_crop_requirements(soil_data, crop_name, crop_data):
    # Convert soil_data from dictionary to DataFrame
    soil_data_df = pd.DataFrame([soil_data])  # Creates a DataFrame with a single row from the dictionary
    
    # Get mean values for the selected crop, using numeric_only=True
    crop_info = crop_data[crop_data["label"] == crop_name].mean(numeric_only=True)
    
    # Clean column/index names
    crop_info.index = crop_info.index.str.strip().str.lower()
    soil_data_df.columns = soil_data_df.columns.str.strip().str.lower()
    
    # Initialize comparison dictionary
    comparison = {}
    for param in ["n", "p", "k", "ph", "moisture", "temperature", "humidity", "rainfall"]:
        if param in soil_data_df.columns and param in crop_info.index:
            difference = soil_data_df[param].values[0] - crop_info[param]
            comparison[param] = f"{difference:.2f} (Adjust {'Up' if difference < 0 else 'Down'})"
        else:
            comparison[param] = "Parameter missing"
    
    return comparison



# Load crop data
def load_crop_data():
    try:
        crop_data = pd.read_csv("crop_data.csv")
        # Clean column names by stripping spaces and converting to lowercase
        crop_data.columns = crop_data.columns.str.strip().str.lower()
        return crop_data
    except FileNotFoundError:
        st.error("Crop data file not found. Please ensure 'crop_data.csv' exists in the directory.")
        return None

# Streamlit app layout
st.title("Soil Health and Crop Recommendation System")

# Display soil parameters
st.header("Current Soil Parameters")
soil_data = generate_synthetic_soil_data()
for key, value in soil_data.items():
    st.write(f"{key}: {value:.2f}")

# Soil Health Assessment
st.subheader("Soil Health Assessment")
health_status = assess_soil_health(soil_data)
st.write(f"Soil Health Status: {health_status}")

# Load model and crop data
model = load_crop_model()
crop_data = load_crop_data()

# Crop Recommendation
st.subheader("Crop Recommendation")
if st.button("Recommend Crop") and model is not None:
    recommended_crop = recommend_crops(soil_data, model)
    st.write(f"Recommended Crop: {recommended_crop}")

# Crop Parameter Comparison
st.subheader("Compare Crop Requirements")
if crop_data is not None:
    crop_to_compare = st.selectbox("Select a crop to compare", options=crop_data["label"].unique())
    if crop_to_compare:
        comparison = compare_parameters_with_crop_requirements(soil_data, crop_to_compare, crop_data)
        st.write(f"Comparison data for {crop_to_compare}:")
        for param, diff in comparison.items():
            st.write(f"{param}: {diff}")
