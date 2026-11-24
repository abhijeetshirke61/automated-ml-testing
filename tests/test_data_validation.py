
import pytest
import pandas as pd
from sklearn.datasets import load_iris
from src.data_validation import validate_data

iris = load_iris()

valid_data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)


def test_valid_data():
    assert validate_data(valid_data)


def test_null_values():
    data = valid_data.copy()
    data.iloc[0, 0] = None

    with pytest.raises(ValueError):
        validate_data(data)


def test_negative_values():
    data = valid_data.copy()
    data.iloc[0, 0] = -5

    with pytest.raises(ValueError):
        validate_data(data)


def test_out_of_range_values():
    data = valid_data.copy()
    data.iloc[0, 0] = 100

    with pytest.raises(ValueError):
        validate_data(data)


def test_missing_column():
    data = valid_data.drop(columns=["sepal width (cm)"])

    with pytest.raises(ValueError):
        validate_data(data)


def test_wrong_column_order():
    data = valid_data[valid_data.columns[::-1]]

    with pytest.raises(ValueError):
        validate_data(data)

def test_invalid_data_type():
    data = valid_data.copy()
    data["sepal length (cm)"] = "invalid"

    with pytest.raises(ValueError):
        validate_data(data)
        