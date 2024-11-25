import pandas as pd

CROP_DATA_PATH = "crop_data.csv"

def load_crop_data(csv_path=CROP_DATA_PATH):
    """
    Loads crop data from a CSV file.

    Args:
    - csv_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Pandas DataFrame containing crop data.
    """
    try:
        crop_data = pd.read_csv(csv_path)
        crop_data.columns = crop_data.columns.str.strip().str.capitalize()  # Standardize column names
        return crop_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Crop data file not found at {csv_path}. Ensure the file exists.")
    except Exception as e:
        raise ValueError(f"Error loading crop data: {str(e)}")

def crop_requirements(crop_name, crop_data):
    """
    Retrieves the crop requirements for a given crop name.

    Args:
    - crop_name (str): Name of the crop.
    - crop_data (DataFrame): DataFrame containing crop parameters.

    Returns:
    - dict: Crop requirements as a dictionary.
    """
    try:
        crop_row = crop_data[crop_data["Crop_name"].str.lower() == crop_name.lower()]
        if crop_row.empty:
            raise ValueError(f"Crop '{crop_name}' not found in the dataset.")
        return crop_row.iloc[0].to_dict()
    except Exception as e:
        raise ValueError(f"Error retrieving crop requirements: {str(e)}")

def compare_soil_to_crop(soil_data, crop_name, crop_data):
    """
    Compares soil data to the ideal crop parameters and suggests adjustments.

    Args:
    - soil_data (dict): Dictionary containing soil parameters.
    - crop_name (str): Name of the crop.
    - crop_data (DataFrame): DataFrame containing crop parameters.

    Returns:
    - dict: Dictionary of suggested adjustments for soil parameters.
    """
    try:
        crop_params = crop_requirements(crop_name, crop_data)
        differences = {}
        for param in soil_data:
            if param in crop_params:
                if soil_data[param] < crop_params[param]:
                    differences[param] = f"Increase {param} by {crop_params[param] - soil_data[param]:.2f}"
                elif soil_data[param] > crop_params[param]:
                    differences[param] = f"Decrease {param} by {soil_data[param] - crop_params[param]:.2f}"
        return differences
    except Exception as e:
        raise ValueError(f"Error comparing soil data to crop requirements: {str(e)}")
