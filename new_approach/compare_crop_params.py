# compare_crop_params.py
import pandas as pd
def crop_requirements(crop_name):
    crop_data = pd.read_csv('crop_data.csv')
    return crop_data[crop_data['crop_name'] == crop_name].iloc[0].to_dict()

def compare_soil_to_crop(soil_data, crop_name):
    crop_params = crop_requirements(crop_name)
    differences = {}
    for param in soil_data:
        if soil_data[param] < crop_params[param]:
            differences[param] = f"Increase {param} by {crop_params[param] - soil_data[param]}"
        elif soil_data[param] > crop_params[param]:
            differences[param] = f"Decrease {param} by {soil_data[param] - crop_params[param]}"
    return differences
