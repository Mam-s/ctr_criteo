from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# a real row from the validation day (2015-02-17), missing columns are not sent
AD = {
    "integer_feature_1": 13,
    "integer_feature_2": 454,
    "integer_feature_3": 40,
    "integer_feature_4": 58,
    "integer_feature_5": 24,
    "integer_feature_6": 3,
    "integer_feature_7": 0,
    "integer_feature_8": 0,
    "integer_feature_9": 21,
    "integer_feature_10": 1,
    "integer_feature_11": 4,
    "integer_feature_12": 550,
    "integer_feature_13": 45,
    "categorical_feature_1": "3fa1c964",
    "categorical_feature_2": "070c56a5",
    "categorical_feature_3": "6c07ed6a",
    "categorical_feature_4": "6521d620",
    "categorical_feature_5": "9dc88528",
    "categorical_feature_6": "6fcd6dcb",
    "categorical_feature_7": "9f443d6b",
    "categorical_feature_8": "038e402d",
    "categorical_feature_9": "e25a4c11",
    "categorical_feature_10": "348aa4dc",
    "categorical_feature_11": "80c2a990",
    "categorical_feature_12": "7577e06f",
    "categorical_feature_13": "a77a4a56",
    "categorical_feature_15": "2e9aae1e",
    "categorical_feature_18": "b8170bba",
    "categorical_feature_19": "9512c20b",
    "categorical_feature_20": "549b4765",
    "categorical_feature_21": "610f46a4",
    "categorical_feature_22": "3d2a1419",
    "categorical_feature_24": "3d0b8fa3",
    "categorical_feature_25": "30436bfc",
    "categorical_feature_26": "962813c6",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_real_ad():
    response = client.post("/predict", json=AD)
    assert response.status_code == 200
    probability = response.json()["click_probability"]
    assert 0 <= probability <= 1


def test_predict_is_deterministic():
    first = client.post("/predict", json=AD).json()["click_probability"]
    second = client.post("/predict", json=AD).json()["click_probability"]
    assert first == second


def test_predict_all_missing():
    # every value missing: the API should still return a probability
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert 0 <= response.json()["click_probability"] <= 1


def test_predict_unseen_category():
    # a value never seen in training must not crash the API
    ad = dict(AD, categorical_feature_2="never_seen_before")
    response = client.post("/predict", json=ad)
    assert response.status_code == 200
    assert 0 <= response.json()["click_probability"] <= 1


def test_predict_negative_value():
    # integer_feature_8 uses -1 in the data
    ad = dict(AD, integer_feature_8=-1)
    response = client.post("/predict", json=ad)
    assert response.status_code == 200


def test_predict_wrong_type():
    response = client.post("/predict", json={"integer_feature_1": "abc"})
    assert response.status_code == 422
