import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Function to generate synthetic sensor data
def generate_sensor_data(num_readings=100):
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps
    base_timestamp = datetime.now()
    timestamps = [base_timestamp + timedelta(hours=i) for i in range(num_readings)]
    
    # Generate sensor readings with realistic ranges
    data = {
        'timestamp': timestamps,
        'pH': np.random.uniform(5.5, 7.5, num_readings),  # Healthy range: 6.0-7.0
        'nitrogen': np.random.uniform(100, 300, num_readings),  # mg/kg, Healthy range: 150-250
        'phosphorus': np.random.uniform(20, 80, num_readings),  # mg/kg, Healthy range: 25-50
        'potassium': np.random.uniform(100, 300, num_readings),  # mg/kg, Healthy range: 150-250
        'temperature': np.random.uniform(20, 35, num_readings),  # °C, Healthy range: 20-30
        'humidity': np.random.uniform(30, 80, num_readings),  # %, Healthy range: 40-60
        'organic_matter': np.random.uniform(2, 6, num_readings)  # %, Healthy range: 3-5
    }
    
    return pd.DataFrame(data)

# Function to assess soil health
def assess_soil_health(df):
    # Calculate average values for each parameter
    avg_values = df.mean()
    
    # Define healthy ranges for each parameter
    healthy_ranges = {
        'pH': (6.0, 7.0),
        'nitrogen': (150, 250),
        'phosphorus': (25, 50),
        'potassium': (150, 250),
        'temperature': (20, 30),
        'humidity': (40, 60),
        'organic_matter': (3, 5)
    }
    
    # Assess each parameter
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
            
        assessment[param] = {
            'value': round(value, 2),
            'status': status,
            'healthy_range': f"{min_val}-{max_val}"
        }
        overall_score += score
    
    # Calculate overall health status
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

# Function to recommend crops based on soil parameters
def recommend_crops(soil_params):
    # Define crop requirements (simplified for demonstration)
    crop_requirements = {
        'Rice': {
            'pH': (5.5, 6.5),
            'nitrogen': (120, 200),
            'phosphorus': (20, 40),
            'potassium': (150, 250),
            'temperature': (22, 30),
            'humidity': (60, 75)
        },
        'Wheat': {
            'pH': (6.0, 7.0),
            'nitrogen': (150, 250),
            'phosphorus': (25, 45),
            'potassium': (160, 240),
            'temperature': (15, 25),
            'humidity': (40, 60)
        },
        'Corn': {
            'pH': (5.8, 7.0),
            'nitrogen': (140, 220),
            'phosphorus': (30, 50),
            'potassium': (130, 200),
            'temperature': (20, 30),
            'humidity': (50, 65)
        },
        'Soybeans': {
            'pH': (6.0, 7.0),
            'nitrogen': (100, 180),
            'phosphorus': (25, 45),
            'potassium': (140, 220),
            'temperature': (20, 28),
            'humidity': (45, 65)
        }
    }
    
    # Calculate suitability scores for each crop
    crop_scores = {}
    
    for crop, requirements in crop_requirements.items():
        score = 0
        for param, (min_val, max_val) in requirements.items():
            if param in soil_params:
                value = soil_params[param]['value']
                if min_val <= value <= max_val:
                    score += 1
        
        crop_scores[crop] = (score / len(requirements)) * 100
    
    # Sort crops by suitability score
    recommended_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommended_crops

# Main execution
def main():
    # Generate synthetic sensor data
    print("Generating sensor data...")
    sensor_data = generate_sensor_data(100)
    
    # Assess soil health
    print("\nAnalyzing soil health...")
    assessment, overall_status, health_percentage = assess_soil_health(sensor_data)
    
    # Print soil health report
    print("\n=== Soil Health Report ===")
    print(f"Overall Status: {overall_status} ({health_percentage:.1f}%)\n")
    print("Parameter Analysis:")
    for param, details in assessment.items():
        print(f"{param}:")
        print(f"  Current Value: {details['value']}")
        print(f"  Status: {details['status']}")
        print(f"  Healthy Range: {details['healthy_range']}\n")
    
    # Get crop recommendations
    print("=== Crop Recommendations ===")
    recommendations = recommend_crops(assessment)
    print("\nRecommended crops by suitability:")
    for crop, score in recommendations:
        print(f"{crop}: {score:.1f}% suitable")

if __name__ == "__main__":
    main()