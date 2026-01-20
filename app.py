from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import pickle

app = Flask(__name__)

# Load the trained model
# Ensure the path matches your directory structure
MODEL_PATH = os.path.join('model', 'wine_cultivar_model.pkl')

print("Loading model from:", MODEL_PATH)
try:
   with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']      # This is the actual sklearn model
        scaler = model_data['scaler']    # Optional if you want to scale inputs
        selected_features = model_data['features']
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Please run the training script first.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        if model is None:
            return render_template('index.html', prediction_text="Error: Model file is missing.")
            
        try:
            # Get values from the form based on the 6 selected features
            alcohol = float(request.form['alcohol'])
            malic_acid = float(request.form['malic_acid'])
            ash = float(request.form['ash'])
            total_phenols = float(request.form['total_phenols'])
            color_intensity = float(request.form['color_intensity'])
            proline = float(request.form['proline'])

            # Create numpy array for prediction
            # The order must match the training feature order
            features = np.array([[alcohol, malic_acid, ash, total_phenols, color_intensity, proline]])

            # Predict (The pipeline handles scaling automatically)
            prediction = model.predict(features)
            
            # Map prediction to Class Name (0, 1, 2 -> Cultivar 1, 2, 3)
            # The sklearn dataset targets are 0, 1, 2. The prompt usually expects 1, 2, 3.
            cultivar_class = prediction[0] + 1 
            
            prediction_text = f"Predicted Origin: Cultivar {cultivar_class}"

        except ValueError:
            prediction_text = "Error: Please enter valid numerical values."
        except Exception as e:
            prediction_text = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)