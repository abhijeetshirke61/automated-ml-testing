
import pandas as pd
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_PATH = Path(__file__).with_name("model.pkl")


def train_model():
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
    model, X_test, y_test = train_model()
    accuracy = evaluate_model(model, X_test, y_test)

    print("Model Accuracy:", accuracy)
    print("Model saved successfully!")
