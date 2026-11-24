
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="ML Prediction API")

model = joblib.load(Path(__file__).with_name("model.pkl"))


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.get("/")
def home():
    return {"message": "ML Prediction API Running"}


@app.post("/predict")
def predict(data: FlowerInput):

    # Correct feature order
    values = pd.DataFrame([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ])

    prediction = model.predict(values)

    return {"prediction": int(prediction[0])}
