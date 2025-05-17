from flask import Flask, request, render_template
import pickle
import numpy as np

# Load models and scaler
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read input values from the form
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])
    profile_score = float(request.form['profile_score'])

    # Prepare and scale input
    features = np.array([[cgpa, iq, profile_score]])
    scaled_features = scaler.transform(features)

    # Predict using the model
    prediction = svc_model.predict(scaled_features)[0]

    # Interpret result
    result = "Placed" if prediction == 1 else "Not Placed"

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
