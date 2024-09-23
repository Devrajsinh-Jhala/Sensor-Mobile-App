import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

feature_mapping = {
    'mean_accel_x': 'tBodyAcc-mean()-X',
    'std_accel_x': 'tBodyAcc-std()-X',
    'max_accel_x': 'tBodyAcc-max()-X',
    'min_accel_x': 'tBodyAcc-min()-X',
    'mad_accel_x': 'tBodyAcc-mad()-X',
    'skewness_accel_x': 'fBodyAcc-skewness()-X',
    'kurtosis_accel_x': 'fBodyAcc-kurtosis()-X',
    'energy_accel_x': 'tBodyAcc-energy()-X',
    'freq_mean_accel_x': 'fBodyAcc-mean()-X',
    'freq_std_accel_x': 'fBodyAcc-std()-X',
    'freq_max_accel_x': 'fBodyAcc-max()-X',
    'freq_entropy_accel_x': 'fBodyAcc-entropy()-X',
    'mean_accel_y': 'tBodyAcc-mean()-Y',
    'std_accel_y': 'tBodyAcc-std()-Y',
    'max_accel_y': 'tBodyAcc-max()-Y',
    'min_accel_y': 'tBodyAcc-min()-Y',
    'mad_accel_y': 'tBodyAcc-mad()-Y',
    'skewness_accel_y': 'fBodyAcc-skewness()-Y',
    'kurtosis_accel_y': 'fBodyAcc-kurtosis()-Y',
    'energy_accel_y': 'tBodyAcc-energy()-Y',
    'freq_mean_accel_y': 'fBodyAcc-mean()-Y',
    'freq_std_accel_y': 'fBodyAcc-std()-Y',
    'freq_max_accel_y': 'fBodyAcc-max()-Y',
    'freq_entropy_accel_y': 'fBodyAcc-entropy()-Y',
    'mean_accel_z': 'tBodyAcc-mean()-Z',
    'std_accel_z': 'tBodyAcc-std()-Z',
    'max_accel_z': 'tBodyAcc-max()-Z',
    'min_accel_z': 'tBodyAcc-min()-Z',
    'mad_accel_z': 'tBodyAcc-mad()-Z',
    'skewness_accel_z': 'fBodyAcc-skewness()-Z',
    'kurtosis_accel_z': 'fBodyAcc-kurtosis()-Z',
    'energy_accel_z': 'tBodyAcc-energy()-Z',
    'freq_mean_accel_z': 'fBodyAcc-mean()-Z',
    'freq_std_accel_z': 'fBodyAcc-std()-Z',
    'freq_max_accel_z': 'fBodyAcc-max()-Z',
    'freq_entropy_accel_z': 'fBodyAcc-entropy()-Z'
}


# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class SensorData(BaseModel):
    sensor_data: List[dict]

# Function to extract features from sensor data (similar to the one you had)
def extract_features(sensor_data):
    
    df = pd.DataFrame(sensor_data)
    features = {}
    
    def calculate_statistics(axis_data, axis_name):
        features[f'mean_{axis_name}'] = np.mean(axis_data)
        features[f'std_{axis_name}'] = np.std(axis_data)
        features[f'max_{axis_name}'] = np.max(axis_data)
        features[f'min_{axis_name}'] = np.min(axis_data)
        features[f'mad_{axis_name}'] = np.mean(np.abs(axis_data - np.mean(axis_data)))
        features[f'skewness_{axis_name}'] = skew(axis_data)
        features[f'kurtosis_{axis_name}'] = kurtosis(axis_data)
        features[f'energy_{axis_name}'] = np.sum(axis_data**2) / len(axis_data)

    def calculate_frequency_features(axis_data, axis_name):
        freqs, power = welch(axis_data, fs=50, nperseg=256)
        features[f'freq_mean_{axis_name}'] = np.mean(power)
        features[f'freq_std_{axis_name}'] = np.std(power)
        features[f'freq_max_{axis_name}'] = np.max(power)
        features[f'freq_entropy_{axis_name}'] = -np.sum(power * np.log2(power + 1e-8))

    # Sensor data extraction as per your logic
    accel_data = df[df['sensor'] == 'accelerometer']
    gyro_data = df[df['sensor'] == 'gyroscope']
    mag_data = df[df['sensor'] == 'magnetometer']

    if not accel_data.empty:
        for axis in ['x', 'y', 'z']:
            calculate_statistics(accel_data[axis], f'accel_{axis}')
            calculate_frequency_features(accel_data[axis], f'accel_{axis}')

    if not gyro_data.empty:
        for axis in ['x', 'y', 'z']:
            calculate_statistics(gyro_data[axis], f'gyro_{axis}')
            calculate_frequency_features(gyro_data[axis], f'gyro_{axis}')

    if not mag_data.empty:
        for axis in ['x', 'y', 'z']:
            calculate_statistics(mag_data[axis], f'mag_{axis}')
            calculate_frequency_features(mag_data[axis], f'mag_{axis}')

    features_df = pd.DataFrame([features])

    # Filter features to match your desired output
    filtered_features = features_df[list(feature_mapping.keys())]
    return filtered_features

# Function to handle prediction request
def predict_activity(sensor_data):
    features_df = extract_features(sensor_data)
    

    # Load your pre-trained model
    model = joblib.load('random_forest_model.joblib')
    # features_df = features_df.astype(float)  # Convert DataFrame values to float
    # features_list = features_df.to_dict(orient="records")

    # Perform prediction
    prediction = model.predict(features_df)
    return {
        "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    }

# FastAPI route for predictions
@app.post("/predict")
async def predict(sensor_data: SensorData):
    sensor_data = sensor_data.sensor_data
    if not sensor_data:
        return {"error": "No sensor data provided"}

    # Get prediction
    prediction = predict_activity(sensor_data)
    print("Prediction to be sent: ",prediction)
    return {"prediction": prediction}
