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
        # Get form inputs
        risk_tolerance = request.form['risk_tolerance']
        investment_horizon = request.form['investment_horizon']
        financial_exp = request.form['financial_experience']

        # Encode inputs to match your model training
        # (You must match the encoding used while training!)
        risk_map = {'low': 0, 'medium': 1, 'high': 2}
        horizon_map = {'short': 0, 'medium': 1, 'long': 2}
        exp_map = {'beginner': 0, 'intermediate': 1, 'expert': 2}

        encoded_features = [
            risk_map[risk_tolerance],
            horizon_map[investment_horizon],
            exp_map[financial_exp]
        ]

        # Convert to NumPy array and reshape for model
        features = np.array(encoded_features).reshape(1, -1)

        # Predict using model
        suggested_stocks = model.predict(features)

        # Take top 5–6 stocks
        suggested_stocks_list = suggested_stocks[:6] if len(suggested_stocks) >= 6 else suggested_stocks

        return render_template('result.html', stocks=suggested_stocks_list)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)


