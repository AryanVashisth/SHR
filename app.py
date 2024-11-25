# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('crop_data.csv')
    return df

def train_model(df):
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, X.columns

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
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Crop Recommendation", "Data Analysis"])
    
    if page == "Home":
        show_home_page()
    elif page == "Crop Recommendation":
        show_recommendation_page()
    else:
        show_analysis_page()

def show_home_page():
    st.title("🌾 Smart Crop Recommendation System")
    st.write("""
    Welcome to the Smart Crop Recommendation System! This application helps farmers make informed decisions 
    about crop selection based on soil conditions and environmental factors.
    
    ### Features:
    - Soil health assessment
    - Crop recommendations based on soil parameters
    - Data visualization and analysis
    - Machine learning-powered predictions
    
    ### How to use:
    1. Navigate to the 'Crop Recommendation' page to get personalized crop suggestions
    2. Input your soil parameters and environmental conditions
    3. View the recommended crops and detailed analysis
    4. Explore data patterns in the 'Data Analysis' page
    """)

def show_recommendation_page():
    st.title("🌱 Crop Recommendation")
    
    # Load data and train model
    df = load_data()
    model, feature_names = train_model(df)
    
    # Create input form
    with st.form("crop_recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.number_input("Nitrogen (N)", min_value=0, max_value=140)
            p = st.number_input("Phosphorus (P)", min_value=0, max_value=145)
            k = st.number_input("Potassium (K)", min_value=0, max_value=205)
            temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0)
            
        with col2:
            humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0)
            ph = st.number_input("pH", min_value=3.5, max_value=10.0)
            rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0)
            
        submitted = st.form_submit_button("Get Recommendation")
        
        if submitted:
            # Create input DataFrame with feature names
            input_data = pd.DataFrame(
                [[n, p, k, temperature, humidity, ph, rainfall]],
                columns=feature_names
            )
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Display results
            st.success(f"Recommended Crop: **{prediction}**")
            
            # Show top 3 recommendations with probabilities
            crop_probs = list(zip(model.classes_, probabilities))
            top_3 = sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]
            
            st.write("### Top 3 Recommendations:")
            for crop, prob in top_3:
                st.write(f"- {crop}: {prob*100:.2f}% confidence")
            
            # Display gauge charts for soil parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_n = create_gauge_chart(n, "Nitrogen Level", 140)
                st.plotly_chart(fig_n, use_container_width=True)
                
            with col2:
                fig_p = create_gauge_chart(p, "Phosphorus Level", 145)
                st.plotly_chart(fig_p, use_container_width=True)
                
            with col3:
                fig_k = create_gauge_chart(k, "Potassium Level", 205)
                st.plotly_chart(fig_k, use_container_width=True)
            
            # Display additional gauge charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_temp = create_gauge_chart(temperature, "Temperature (°C)", 44)
                st.plotly_chart(fig_temp, use_container_width=True)
                
            with col2:
                fig_humidity = create_gauge_chart(humidity, "Humidity (%)", 100)
                st.plotly_chart(fig_humidity, use_container_width=True)
                
            with col3:
                fig_ph = create_gauge_chart(ph, "pH Level", 10)
                st.plotly_chart(fig_ph, use_container_width=True)

def show_analysis_page():
    st.title("📊 Data Analysis")
    
    # Load data
    df = load_data()
    
    # Show dataset info
    st.write("### Dataset Overview")
    st.write(f"Number of records: {len(df)}")
    st.write(f"Number of unique crops: {len(df['label'].unique())}")
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose visualization",
        ["Crop Distribution", "Parameter Distributions", "Correlation Matrix"]
    )
    
    if viz_option == "Crop Distribution":
        fig = px.bar(
            df['label'].value_counts().reset_index(),
            x='index',
            y='label',
            title="Distribution of Crops in Dataset",
            labels={'index': 'Crop', 'label': 'Count'}
        )
        st.plotly_chart(fig)
        
    elif viz_option == "Parameter Distributions":
        param = st.selectbox(
            "Select parameter",
            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )
        fig = px.box(
            df,
            x='label',
            y=param,
            title=f"Distribution of {param} across different crops"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        
    else:  # Correlation Matrix
        corr = df.drop('label', axis=1).corr()
        fig = px.imshow(
            corr,
            title="Correlation Matrix of Soil Parameters",
            labels=dict(color="Correlation")
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()