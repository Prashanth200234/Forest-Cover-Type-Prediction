import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Load the trained model and scaler
model = joblib.load("forest_cover_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names based on the dataset and PDF
feature_names = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
    'Wilderness_Area3', 'Wilderness_Area4'
] + [f'Soil_Type{i}' for i in range(1, 41)]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form data
        input_data = [float(request.form.get(name, 0)) for name in feature_names]
        
        # Convert to DataFrame for consistency
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Return result as plain text without brackets
        return f'"Predicted Cover Type": {int(prediction)}'
    except Exception as e:
        return f'"error": {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
