import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


MODEL_DIR = "models"


class MachineLearningService:

    def train_model(
        self,
        dataset_path: str,
        target_column: str,
        model_name: str
    ):

        os.makedirs(MODEL_DIR, exist_ok=True)

        df = pd.read_csv(dataset_path)

        X = df.drop(columns=[target_column])

        y = df[target_column]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        model_path = (
            f"{MODEL_DIR}/{model_name}.pkl"
        )

        joblib.dump(model, model_path)

        return {
            "model_path": model_path,
            "accuracy": accuracy
        }

    def load_model(
        self,
        model_path: str
    ):

        return joblib.load(model_path)