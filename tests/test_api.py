
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == 0


def test_invalid_prediction():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5
        }
    )

    assert response.status_code == 422


def test_missing_features():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422
