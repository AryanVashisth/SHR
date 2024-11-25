# soil_health_assessment.py
import random

def generate_synthetic_soil_data():
    soil_data = {
        "Nitrogen": random.uniform(10, 100),
        "Phosphorus": random.uniform(10, 100),
        "Potassium": random.uniform(10, 100),
        "Temperature": random.uniform(10, 40),
        "Humidity": random.uniform(60, 90),
        "pH": random.uniform(5.5, 7.5),
        "Rainfall": random.uniform(50, 300)
    }
    return soil_data

def assess_soil_health(soil_data):
    nitrogen, phosphorus, potassium, pH = (
        soil_data["Nitrogen"], soil_data["Phosphorus"],
        soil_data["Potassium"], soil_data["pH"]
    )

    # Example criteria for soil health
    if nitrogen > 50 and phosphorus > 30 and potassium > 30 and 6 <= pH <= 7:
        return "Good"
    elif nitrogen > 30 and phosphorus > 20 and potassium > 20 and 5.5 <= pH <= 7.5:
        return "Average"
    else:
        return "Poor"
