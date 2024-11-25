import streamlit as st
import random
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Generate adjustable soil data using sliders with tooltips and better range
def generate_adjustable_soil_data():
    soil_data = {
        "n": st.slider("Nitrogen (N)", 0, 100, 50, help="Adjust the nitrogen content of the soil."),
        "p": st.slider("Phosphorus (P)", 0, 100, 50, help="Adjust the phosphorus content of the soil."),
        "k": st.slider("Potassium (K)", 0, 100, 50, help="Adjust the potassium content of the soil."),
        "ph": st.slider("pH Level", 5.0, 8.0, 6.5, step=0.1, help="Adjust the pH level of the soil."),
        "moisture": st.slider("Moisture (%)", 0, 100, 50, help="Adjust the moisture level of the soil."),
        "temperature": st.slider("Temperature (°C)", 15, 35, 25, help="Adjust the temperature of the soil."),
        "humidity": st.slider("Humidity (%)", 50, 100, 75, help="Adjust the humidity level of the soil."),
        "rainfall": st.slider("Rainfall (mm)", 100, 300, 200, help="Adjust the rainfall level of the area.")
    }
    return soil_data

# Assess soil health based on selected parameters
def assess_soil_health(soil_data):
    if soil_data["ph"] < 5.5 or soil_data["ph"] > 7.5:
        return "Poor"
    if soil_data["moisture"] < 20:
        return "Poor"
    if soil_data["n"] < 10 or soil_data["p"] < 10 or soil_data["k"] < 10:
        return "Poor"
    return "Good" if soil_data["moisture"] > 50 else "Average"

# Load the crop recommendation model
def load_crop_model():
    try:
        return joblib.load("crop_recommendation_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'crop_recommendation_model.pkl' exists in the directory.")
        return None

# Recommend crops based on the soil data input
def recommend_crops(soil_data, model):
    input_data = np.array([[soil_data["n"], soil_data["p"], soil_data["k"],
                            soil_data["temperature"], soil_data["humidity"],
                            soil_data["ph"], soil_data["rainfall"]]])
    prediction = model.predict(input_data)
    return prediction[0]

# Compare soil data with ideal crop requirements
def compare_parameters_with_crop_requirements(soil_data, crop_name, crop_data):
    crop_info = crop_data[crop_data["label"] == crop_name].mean(numeric_only=True)
    comparison = {}

    for param in ["n", "p", "k", "ph", "moisture", "temperature", "humidity", "rainfall"]:
        if param in soil_data and param in crop_info.index:
            difference = soil_data[param] - crop_info[param]
            comparison[param] = difference
        else:
            comparison[param] = None
    return comparison

# Load crop data from a CSV file
def load_crop_data():
    try:
        crop_data = pd.read_csv("new_approach/crop_data.csv")
        crop_data.columns = crop_data.columns.str.strip().str.lower()
        return crop_data
    except FileNotFoundError:
        st.error("Crop data file not found. Please ensure 'crop_data.csv' exists in the directory.")
        return None

# Streamlit app layout
st.set_page_config(page_title="Soil Health & Crop Recommendation", layout="wide")
st.title("🌱 Interactive Soil Health & Crop Recommendation System 🌾")

# Adjustable Soil Parameters
st.header("🔧 Adjust Soil Parameters")
soil_data = generate_adjustable_soil_data()

# Soil Health Assessment
st.subheader("Soil Health Assessment 🌍")
health_status = assess_soil_health(soil_data)
st.write(f"**Soil Health Status**: {health_status}")

# Load model and crop data
model = load_crop_model()
crop_data = load_crop_data()

# Crop Recommendation
st.subheader("🌾 Crop Recommendation")
if st.button("Recommend Crop 🌻") and model is not None:
    recommended_crop = recommend_crops(soil_data, model)
    st.write(f"**Recommended Crop**: {recommended_crop}")

# Crop Parameter Comparison
st.subheader("Compare Soil Data with Crop Requirements 🌾")
if crop_data is not None:
    crop_to_compare = st.selectbox("Select a Crop to Compare", options=crop_data["label"].unique())
    if crop_to_compare:
        comparison = compare_parameters_with_crop_requirements(soil_data, crop_to_compare, crop_data)
        
        # Display comparison as text
        st.write(f"**Comparison data for {crop_to_compare}:**")
        for param, diff in comparison.items():
            if diff is not None:
                st.write(f"{param.capitalize()}: {diff:.2f} (Adjust {'Up' if diff < 0 else 'Down'})")
            else:
                st.write(f"{param.capitalize()}: Parameter not available")

        # Visual Comparison using an animated bar chart
        st.subheader("Visual Comparison of Soil Data vs Ideal Crop Requirements 📊")
        
        # Prepare data for Plotly animation
        soil_values = list(soil_data.values())
        crop_values = [
            crop_data[crop_data["label"] == crop_to_compare][col].mean() if col in crop_data.columns else None
            for col in soil_data.keys()
        ]

        valid_soil_values = [val for val, crop_val in zip(soil_values, crop_values) if crop_val is not None]
        valid_crop_values = [val for val in crop_values if val is not None]
        valid_params = [param for param, crop_val in zip(soil_data.keys(), crop_values) if crop_val is not None]
        
        # Create the bar chart with animation using Plotly
        fig = go.Figure()

        # Add bars for current soil data
        fig.add_trace(go.Bar(
            x=valid_params,
            y=valid_soil_values,
            name="Current Soil Data",
            marker_color='rgb(158,202,225)',
            width=0.4,
            offsetgroup=1
        ))

        # Add bars for ideal crop data
        fig.add_trace(go.Bar(
            x=valid_params,
            y=valid_crop_values,
            name=f"Ideal for {crop_to_compare}",
            marker_color='rgb(0,204,255)',
            width=0.4,
            offsetgroup=2
        ))

        fig.update_layout(
            title=f"Comparison of Current Soil vs Ideal Crop Data for {crop_to_compare}",
            xaxis_title="Soil Parameters",
            yaxis_title="Values",
            barmode='group',
            xaxis=dict(tickangle=45),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

# Add a footer
st.markdown("Made for better farming solutions!")
