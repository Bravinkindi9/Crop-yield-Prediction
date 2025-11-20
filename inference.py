import joblib
import pandas as pd
import os

ARTIFACTS_DIR = r"C:\Users\USER\Desktop\Projects_git\Crop yield dataset\artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model_pipeline.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'y_scaler.pkl')

def predict_yield(area, item, year, rainfall, pesticides, temperature):
    # Load artifacts
    try:
        pipeline = joblib.load(MODEL_PATH)
        y_scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Model or scaler not found. Please run train_and_save.py first.")
        return

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Area': [area],
        'Item': [item],
        'Year': [year],
        'Rainfall_mm': [rainfall],
        'Pesticides_tonnes': [pesticides],
        'Temperature_C': [temperature]
    })

    print("Input Data:")
    print(input_data)

    # Predict (pipeline handles preprocessing)
    scaled_prediction = pipeline.predict(input_data)

    # Inverse transform
    prediction = y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))

    print(f"\nPredicted Yield (hg/ha): {prediction[0][0]:.2f}")
    return prediction[0][0]

if __name__ == "__main__":
    # Sample prediction
    # Values taken roughly from dataset range or imagination
    predict_yield(
        area='India', 
        item='Maize', 
        year=2025, 
        rainfall=1000.0, 
        pesticides=500.0, 
        temperature=25.0
    )
