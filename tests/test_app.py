import os

import pytest

import app as app_module
from train import train

SAMPLE = {"age": 35, "gender": "Female", "annual_income": 70000,
          "browsing_time": 25, "pages_visited": 12, "previous_purchases": 3,
          "time_on_site": 30, "discount_offered": 15}


@pytest.fixture(scope="session", autouse=True)
def model_file():
    if not os.path.exists(app_module.MODEL_PATH):
        from generate_data import generate
        os.makedirs("data", exist_ok=True)
        if not os.path.exists("data/customer_data.csv"):
            generate().to_csv("data/customer_data.csv", index=False)
        train("data", os.path.dirname(app_module.MODEL_PATH) or ".")


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Customer Purchase Prediction" in r.data


def test_api_prediction(client):
    r = client.post("/api/predict", json=SAMPLE)
    body = r.get_json()
    assert r.status_code == 200
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_form_prediction(client):
    r = client.post("/predict", data=SAMPLE)
    assert r.status_code == 200 and b"Purchase probability" in r.data


def test_missing_field_rejected(client):
    bad = dict(SAMPLE)
    bad.pop("age")
    assert client.post("/api/predict", json=bad).status_code == 400


def test_invalid_gender_rejected(client):
    bad = dict(SAMPLE, gender="Other")
    assert client.post("/api/predict", json=bad).status_code == 400
