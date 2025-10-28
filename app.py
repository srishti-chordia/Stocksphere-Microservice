from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "AgeGroup": [request.form["age"]],
        "Horizon": [request.form["horizon"]],
        "Experience": [request.form["experience"]],
        "Income": [request.form["income"]],
        "RiskTolerance": [request.form["risk"]],
    }

    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = encoders[col].transform(df[col])

    probs = model.predict_proba(df)[0]
    top_indices = probs.argsort()[-6:][::-1]
    top_tickers = encoders["Ticker"].inverse_transform(top_indices)

    return render_template("index.html", result=top_tickers)


if __name__ == "__main__":
    app.run(debug=True)
