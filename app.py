from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        risk_tolerance = request.form['risk_tolerance']
        investment_horizon = request.form['investment_horizon']
        financial_exp = request.form['financial_experience']

        # Convert form inputs to numeric or categorical encodings as per your model
        # Example (customize this based on your model’s training data):
        features = np.array([[risk_tolerance, investment_horizon, financial_exp]])

        # Predict using your model
        suggested_stocks = model.predict(features)

        # If your model returns stock indices or names, handle it here
        suggested_stocks_list = suggested_stocks[:6]  # show top 5–6

        return render_template('result.html', stocks=suggested_stocks_list)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
