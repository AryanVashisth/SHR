from flask import Flask, request, jsonify
from crop_model import load_crop_model, recommend_crops
from crop_data_handler import load_crop_data, compare_soil_to_crop

app = Flask(__name__)

# Paths for resources
MODEL_PATH = "crop_recommendation_model.pkl"
CSV_PATH = "crop_data.csv"

# Load resources
try:
    model = load_crop_model(MODEL_PATH)
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}. Ensure the file exists.")

try:
    crop_data = load_crop_data(CSV_PATH)
except FileNotFoundError:
    crop_data = None
    print(f"Error: Crop data file not found at {CSV_PATH}. Ensure the file exists.")

@app.route("/recommend", methods=["POST"])
def recommend_crop_route():
    """
    Endpoint to recommend a crop based on soil data.
    """
    if model is None:
        return jsonify({"error": "Model file not found"}), 500
    try:
        data = request.json
        if not data:
            raise KeyError("No JSON data received.")
        
        prediction = recommend_crops(model, data)
        return jsonify({"recommended_crop": prediction})
    except KeyError as e:
        return jsonify({"error": f"Missing or invalid parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare_parameters_route():
    """
    Endpoint to compare soil parameters with crop requirements.
    """
    if crop_data is None:
        return jsonify({"error": "Crop data file not found"}), 500
    try:
        data = request.json
        if not data:
            raise KeyError("No JSON data received.")
        
        crop_name = data.get("crop_name")
        soil_data = data.get("soil_data")
        if not crop_name or not soil_data:
            raise KeyError("Both 'crop_name' and 'soil_data' are required.")

        comparison = compare_soil_to_crop(crop_data, crop_name, soil_data)
        return jsonify({"comparison": comparison})
    except KeyError as e:
        return jsonify({"error": f"Missing or invalid parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
