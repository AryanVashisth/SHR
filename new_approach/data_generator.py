# data_generator.py
import random

def generate_soil_data():
    return {
        "moisture": random.uniform(10, 60),
        "nitrogen": random.uniform(0, 100),
        "phosphorus": random.uniform(0, 100),
        "potassium": random.uniform(0, 100),
        "pH": random.uniform(5.5, 8.5)
    }
