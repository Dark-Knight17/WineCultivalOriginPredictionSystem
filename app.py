from flask import Flask, request, render_template
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join('model', 'wine_cultivar_model.pkl')
print("Loading model from:", MODEL_PATH)

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']       # Logistic Regression model
    scaler = model_data['scaler']     # StandardScaler
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
            # Read form values (must match selected features)
            alcohol = float(request.form['alcohol'])
            malic_acid = float(request.form['malic_acid'])
            ash = float(request.form['ash'])
            total_phenols = float(request.form['total_phenols'])
            color_intensity = float(request.form['color_intensity'])
            proline = float(request.form['proline'])

            # Create feature array in the same order as training
            features = np.array([[alcohol, malic_acid, ash, total_phenols, color_intensity, proline]])

            # Scale features using saved scaler
            features_scaled = scaler.transform(features)

            # Predict class
            prediction = model.predict(features_scaled)

            # Map prediction to Cultivar 1, 2, 3
            cultivar_class = prediction[0] + 1
            prediction_text = f"Predicted Origin: Cultivar {cultivar_class}"

        except ValueError:
            prediction_text = "Error: Please enter valid numerical values."
        except Exception as e:
            prediction_text = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
