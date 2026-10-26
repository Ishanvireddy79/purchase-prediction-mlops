import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "annual_income", "browsing_time", "pages_visited",
           "previous_purchases", "time_on_site", "discount_offered"]
CATEGORICAL = ["gender"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "purchase"


def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", estimator)])


def train(train_dir, model_dir):
    df = pd.read_csv(os.path.join(train_dir, "customer_data.csv"))
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    results, best_name, best_pipe = {}, None, None
    for name, est in candidates.items():
        pipe = build_pipeline(est).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "auc": roc_auc_score(y_te, proba),
        }
        print(name, {k: round(v, 4) for k, v in results[name].items()})
        if best_name is None or results[name]["f1"] > results[best_name]["f1"]:
            best_name, best_pipe = name, pipe

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_pipe, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print("Best model:", best_name)
    return best_pipe


# ---- SageMaker inference handlers (used when deployed as an endpoint) ----
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type="application/json"):
    data = json.loads(request_body)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)[FEATURES]


def predict_fn(input_data, model):
    proba = model.predict_proba(input_data)[:, 1]
    return {"predictions": [int(p >= 0.5) for p in proba],
            "probabilities": [float(p) for p in proba]}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    args = parser.parse_args()
    train(args.train, args.model_dir)
