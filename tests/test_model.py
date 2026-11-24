
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from src.model import train_model, evaluate_model

model, X_test, y_test = train_model()

iris = load_iris()


def test_minimum_accuracy():
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy >= 0.85


def test_prediction_shape():
    predictions = model.predict(X_test)
    assert predictions.shape == (len(X_test),)


def test_prediction_classes():
    predictions = model.predict(X_test)
    assert set(predictions).issubset({0, 1, 2})


def test_prediction_invariance():
    original = model.predict(X_test)

    shuffled = X_test.sample(frac=1, random_state=10)
    shuffled_predictions = model.predict(shuffled)

    restored = pd.Series(
        shuffled_predictions,
        index=shuffled.index
    ).loc[X_test.index].to_numpy()

    np.testing.assert_array_equal(original, restored)


def test_directional_behavior():
    flower = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=iris.feature_names
    )

    prediction = model.predict(flower)

    assert prediction[0] == 0
