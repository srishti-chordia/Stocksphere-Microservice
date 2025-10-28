from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        age_group = request.form['age_group']
        investment_horizon = request.form['investment_horizon']
        financial_exp = request.form['financial_experience']
        annual_income = request.form['annual_income']
        risk_tolerance = request.form['risk_tolerance']

        # Encoding maps — must match your training encodings
        age_map = {'18-25': 0, '26-35': 1, '36-50': 2, '50+': 3}
        horizon_map = {'short': 0, 'medium': 1, 'long': 2}
        exp_map = {'beginner': 0, 'intermediate': 1, 'expert': 2}
        income_map = {'low': 0, 'medium': 1, 'high': 2}
        risk_map = {'low': 0, 'medium': 1, 'high': 2}

        # Create input array in the same feature order used during training
        features = np.array([
            age_map[age_group],
            horizon_map[investment_horizon],
            exp_map[financial_exp],
            income_map[annual_income],
            risk_map[risk_tolerance]
        ]).reshape(1, -1)

        # Predict probabilities and get top 5–6 stocks
        probs = model.predict_proba(features)[0]
        top_indices = probs.argsort()[-6:][::-1]
        top_stocks = encoders['Ticker'].inverse_transform(top_indices)

        return render_template('result.html', stocks=top_stocks)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
