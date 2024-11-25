import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Function to generate synthetic sensor data
def generate_sensor_data(num_readings=100):
    np.random.seed(42)
    base_timestamp = datetime.now()
    timestamps = [base_timestamp + timedelta(hours=i) for i in range(num_readings)]
    data = {
        'timestamp': timestamps,
        'pH': np.random.uniform(5.5, 7.5, num_readings),
        'nitrogen': np.random.uniform(100, 300, num_readings),
        'phosphorus': np.random.uniform(20, 80, num_readings),
        'potassium': np.random.uniform(100, 300, num_readings),
        'temperature': np.random.uniform(20, 35, num_readings),
        'humidity': np.random.uniform(30, 80, num_readings),
        'organic_matter': np.random.uniform(2, 6, num_readings)
    }
    return pd.DataFrame(data)

# Function to assess soil health
def assess_soil_health(df):
    avg_values = df.mean()
    healthy_ranges = {
        'pH': (6.0, 7.0),
        'nitrogen': (150, 250),
        'phosphorus': (25, 50),
        'potassium': (150, 250),
        'temperature': (20, 30),
        'humidity': (40, 60),
        'organic_matter': (3, 5)
    }
    assessment = {}
    overall_score = 0
    for param in healthy_ranges:
        value = avg_values[param]
        min_val, max_val = healthy_ranges[param]
        if min_val <= value <= max_val:
            status = "Optimal"
            score = 3
        elif value < min_val:
            status = "Low"
            score = 1
        else:
            status = "High"
            score = 1
        assessment[param] = {'value': round(value, 2), 'status': status, 'healthy_range': f"{min_val}-{max_val}"}
        overall_score += score
    max_score = len(healthy_ranges) * 3
    health_percentage = (overall_score / max_score) * 100
    if health_percentage >= 80:
        overall_status = "Excellent"
    elif health_percentage >= 60:
        overall_status = "Good"
    elif health_percentage >= 40:
        overall_status = "Fair"
    else:
        overall_status = "Poor"
    return assessment, overall_status, health_percentage

# Function to create gauge charts for parameters
def create_gauge_chart(value, title, max_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value/3], 'color': "lightgray"},
                {'range': [max_value/3, 2*max_value/3], 'color': "gray"},
                {'range': [2*max_value/3, max_value], 'color': "darkgray"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(height=250)
    return fig

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Crop Recommendation", "Data Analysis"])

    if page == "Home":
        st.title("🌾 Smart Crop Recommendation System")
        st.write("This application helps farmers make informed decisions about crop selection.")
    elif page == "Crop Recommendation":
        st.title("🌱 Crop Recommendation")
        sensor_data = generate_sensor_data(100)
        assessment, overall_status, health_percentage = assess_soil_health(sensor_data)
        st.write("### Soil Health Report")
        st.write(f"**Overall Status**: {overall_status} ({health_percentage:.1f}%)")
        for param, details in assessment.items():
            st.write(f"- {param}: {details['value']} ({details['status']}), Healthy Range: {details['healthy_range']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(create_gauge_chart(sensor_data['nitrogen'].mean(), "Nitrogen", 300), use_container_width=True)
        with col2:
            st.plotly_chart(create_gauge_chart(sensor_data['phosphorus'].mean(), "Phosphorus", 80), use_container_width=True)
        with col3:
            st.plotly_chart(create_gauge_chart(sensor_data['potassium'].mean(), "Potassium", 300), use_container_width=True)
    else:
        st.title("📊 Data Analysis")
        df = generate_sensor_data()
        st.write("### Parameter Distributions")
        param = st.selectbox("Select parameter", ['pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'organic_matter'])
        fig = px.histogram(df, x=param, title=f"Distribution of {param}")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
