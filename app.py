import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train import FEATURES

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT")  
_model = None
SCHEMA = {
    "age": (int, 18, 100),
    "annual_income": (float, 0, 10_000_000),
    "browsing_time": (float, 0, 1000),
    "pages_visited": (int, 0, 1000),
    "previous_purchases": (int, 0, 1000),
    "time_on_site": (float, 0, 1000),
    "discount_offered": (float, 0, 100),
}


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def clean(raw):
    rec = {}
    for field, (typ, lo, hi) in SCHEMA.items():
        if raw.get(field) in (None, ""):
            raise ValueError(f"Missing field: {field}")
        try:
            val = typ(float(raw[field]))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid number for {field}")
        if not lo <= val <= hi:
            raise ValueError(f"{field} must be between {lo} and {hi}")
        rec[field] = val
    gender = str(raw.get("gender", "")).strip().capitalize()
    if gender not in ("Male", "Female"):
        raise ValueError("gender must be Male or Female")
    rec["gender"] = gender
    return rec


def predict(rec):
    if ENDPOINT_NAME:
        import boto3
        client = boto3.client("sagemaker-runtime")
        resp = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType="application/json",
            Accept="application/json", Body=json.dumps([rec]))
        out = json.loads(resp["Body"].read())
        return out["predictions"][0], out["probabilities"][0], "SageMaker endpoint"
    df = pd.DataFrame([rec])[FEATURES]
    p = float(get_model().predict_proba(df)[0, 1])
    return int(p >= 0.5), p, "local model"


@app.route("/")
def index():
    return render_template("index.html", result=None, values={}, error=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    values = request.form.to_dict()
    try:
        label, prob, source = predict(clean(values))
    except ValueError as e:
        return render_template("index.html", result=None, values=values, error=str(e)), 400
    result = {"label": "Likely to PURCHASE" if label else "Unlikely to purchase",
              "purchase": label, "probability": round(prob * 100, 1), "source": source}
    return render_template("index.html", result=result, values=values, error=None)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        label, prob, source = predict(clean(request.get_json(force=True) or {}))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"prediction": label, "probability": prob, "source": source})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
