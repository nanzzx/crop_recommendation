import pandas as pd
import joblib
import numpy as np

def calculate_percentage_change(before, after):
    """Calculate percentage change between before and after values."""
    if before == 0:
        return round(after * 100, 2)
    return round(((after - before) / abs(before)) * 100, 2)

# Load trained models
def load_models():
    """Load trained models from files."""
    fertility_model = joblib.load('fertility_improvement_predictor.pkl')
    recommendation_model = joblib.load('model.pkl')
    sc = joblib.load('standscaler.pkl')
    mx = joblib.load('minmaxscaler.pkl')
    return fertility_model, recommendation_model, sc, mx

# Load models once for efficiency
fertility_model, recommendation_model, sc, mx = load_models()

# Load Crop Recommendation Dataset
crop_recommendation_data = pd.read_csv('Crop_recommendation.csv')

# Crop Dictionary from Dataset
crop_labels = crop_recommendation_data['label'].unique()
crop_dict = {idx: crop for idx, crop in enumerate(crop_labels)}

def predict_fertility_changes_and_recommend_crop(
    n_before, p_before, k_before, ph_before,
    n_after, p_after, k_after, ph_after,
    cultivated_crop, temp, humidity, rainfall
):
    """Predict fertility changes and recommend a crop."""

    # Load Crop Waste Data
    crop_waste_data = pd.read_csv('Expanded_Crop_Waste_Data.csv')

    # If cultivated_crop is None, SKIP waste recommendation
    if cultivated_crop:
        if cultivated_crop not in crop_waste_data['Crop'].values:
            raise ValueError(f"Crop '{cultivated_crop}' not found in dataset.")

        # Find Crop Waste Recommendation
        waste_data = crop_waste_data[crop_waste_data['Crop'] == cultivated_crop]['Waste Type'].unique()
        crop_waste = ", ".join(waste_data) if len(waste_data) > 0 else "No waste data available"
    else:
        crop_waste = "No crop cultivated yet"

    # Ensure correct feature order for fertility improvement model (Fix MinMaxScaler issue)
    input_data = np.array([[n_after, p_after, k_after, ph_after, temp, humidity, rainfall]])

    # Scale Input Data (Ensure correct number of features)
    scaled_input = mx.transform(input_data)  # First MinMax Scaling
    standardized_input = sc.transform(scaled_input)  # Then Standard Scaling

    try:
        predicted_improvement = fertility_model.predict(standardized_input)[0]
        
        # Debugging: Print Model Output
        print(f"🔍 Fertility Model Output: {predicted_improvement}")

        # If the model returns all zeros, print a warning
        if np.all(predicted_improvement == 0):
            print("⚠ WARNING: Fertility model returned all zeros. Check training data and features!")
    except Exception as e:
        print("⚠ Error in fertility prediction:", e)
        predicted_improvement = np.array([n_after, p_after, k_after, ph_after])  # Default to current values

    predicted_improvement = np.maximum(predicted_improvement, [n_after, p_after, k_after, ph_after])

    improvement_after_waste = {
        'N': float(calculate_percentage_change(n_after, predicted_improvement[0] * 1.2)),  
        'P': float(calculate_percentage_change(p_after, predicted_improvement[1] * 1.1)),  
        'K': float(calculate_percentage_change(k_after, predicted_improvement[2] * 1.22)),  
        'pH': float(calculate_percentage_change(ph_after, predicted_improvement[3] * 1.015))  
    }

    # Normalize Data Before Crop Recommendation (Fixes the Lentil Issue)
    features = np.array([[n_after, p_after, k_after, ph_after, temp, humidity, rainfall]])
    scaled_features = mx.transform(features)  
    standardized_features = sc.transform(scaled_features)  

    # Recommend Crop Without Waste
    try:
        recommended_crop_no_waste_idx = recommendation_model.predict(standardized_features)[0]
        recommended_crop_no_waste = crop_dict.get(int(recommended_crop_no_waste_idx), "Unknown Crop")
    except Exception as e:
        print("⚠ Error in crop recommendation without waste:", e)
        recommended_crop_no_waste = "No recommendation available"

    # Recommend Crop With Waste (Only if a crop was provided)
    if cultivated_crop:
        improved_features = np.array([[predicted_improvement[0], predicted_improvement[1], predicted_improvement[2], ph_after, temp, humidity, rainfall]])
        improved_scaled_features = mx.transform(improved_features)
        improved_standardized_features = sc.transform(improved_scaled_features)

        try:
            recommended_crop_with_waste_idx = recommendation_model.predict(improved_standardized_features)[0]
            recommended_crop_with_waste = crop_dict.get(int(recommended_crop_with_waste_idx), "Unknown Crop")
        except Exception as e:
            print("⚠ Error in crop recommendation with waste:", e)
            recommended_crop_with_waste = "No recommendation available"
    else:
        recommended_crop_with_waste = "No crop cultivated yet"

    return {
        'fertility_changes': {
            'N': calculate_percentage_change(n_before, n_after),
            'P': calculate_percentage_change(p_before, p_after),
            'K': calculate_percentage_change(k_before, k_after),
            'pH': calculate_percentage_change(ph_before, ph_after)
        },
        'improvement_after_waste': improvement_after_waste,
        'recommended_crop_no_waste': recommended_crop_no_waste,
        'crop_waste': crop_waste,
        'recommended_crop_with_waste': recommended_crop_with_waste
    }
