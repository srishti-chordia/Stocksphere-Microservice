from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form inputs
        age_group = request.form['age_group']
        investment_horizon = request.form['investment_horizon']
        financial_experience = request.form['financial_experience']
        annual_income = request.form['annual_income']
        risk_tolerance = request.form['risk_tolerance']

        # Convert categorical inputs into numeric form
        mapping = {
            'Low': 0, 'Medium': 1, 'High': 2,
            'Beginner': 0, 'Intermediate': 1, 'Expert': 2,
            'Short-term': 0, 'Medium-term': 1, 'Long-term': 2,
            '18-25': 0, '26-40': 1, '41-60': 2, '60+': 3,
            'Below 5L': 0, '5L-10L': 1, '10L-25L': 2, '25L+': 3
        }

        features = [
            mapping.get(age_group, 0),
            mapping.get(investment_horizon, 0),
            mapping.get(financial_experience, 0),
            mapping.get(annual_income, 0),
            mapping.get(risk_tolerance, 0)
        ]

        features = np.array(features).reshape(1, -1)

        # Make prediction
        predicted_indices = model.predict(features)
        print("Model prediction:", predicted_indices)  # Debugging info

        # Stock category mapping (you can tweak as you like)
        stock_map = {
            0: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],  # Global tech
            1: ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'ICICIBANK.NS'],  # Indian blue chips
            2: ['TSLA', 'META', 'NFLX', 'ADBE', 'AMD'],  # Growth stocks
            3: ['BA', 'NKE', 'KO', 'PEP', 'MCD'],  # Consumer & industrial
            4: ['JPM', 'GS', 'BAC', 'MS', 'V']  # Financial sector
        }

        # Handle all kinds of outputs gracefully
        pred = predicted_indices[0]
        try:
            # If model returns something like np.float or np.int
            pred = int(pred)
        except:
            # If model returns string labels like 'High'
            label_map = {'Low': 0, 'Medium': 1, 'High': 2}
            pred = label_map.get(str(pred).capitalize(), 0)

        # Get corresponding stock suggestions
        stocks = stock_map.get(pred, stock_map[0])

        return render_template('result.html', stocks=stocks)

    except Exception as e:
        return render_template('result.html', stocks=[f"Error: {str(e)}"])

if __name__ == '__main__':
    app.run(debug=True)
