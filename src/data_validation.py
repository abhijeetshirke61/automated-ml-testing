
import pandas as pd

REQUIRED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]


def validate_data(data):

    if list(data.columns) != REQUIRED_COLUMNS:
        raise ValueError("Invalid dataset schema")

    if data.isnull().values.any():
        raise ValueError("Null values found")

    if not all(
        pd.api.types.is_numeric_dtype(data[col])
        for col in REQUIRED_COLUMNS
    ):
        raise ValueError("Invalid data type")

    if (data < 0).values.any() or (data > 10).values.any():
        raise ValueError("Values out of range")

    return True
