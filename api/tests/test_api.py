from fastapi.testclient import TestClient

from app.main import ClassProbability, PredictionResponse, app


class FakePredictor:
    def predict(self, pixels):
        probabilities = [ClassProbability(digit=i, probability=0.01) for i in range(10)]
        probabilities[3] = ClassProbability(digit=3, probability=0.91)
        return PredictionResponse(
            digit=3,
            confidence=0.91,
            probabilities=sorted(probabilities, key=lambda item: item.probability, reverse=True),
            model="test_model",
        )


def test_predict_returns_ranked_result():
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        response = client.post("/predict", json={"pixels": [0.5] * 784})
        assert response.status_code == 200
        assert response.json()["digit"] == 3
        assert response.json()["probabilities"][0]["digit"] == 3


def test_predict_rejects_wrong_length():
    with TestClient(app) as client:
        response = client.post("/predict", json={"pixels": [0.5] * 10})
        assert response.status_code == 422


def test_predict_rejects_empty_drawing():
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        response = client.post("/predict", json={"pixels": [0.0] * 784})
        assert response.status_code == 422
