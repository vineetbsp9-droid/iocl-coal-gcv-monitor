import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

FEATURE_COLUMNS = [
    "Moisture",
    "Ash",
    "Volatile_Matter",
    "Fixed_Carbon",
    "Sulfur",
]


def train_model():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "iocl_coal_petcoke_dataset.csv"
    model_path = project_dir / "coal_gcv_model.pkl"

    df = pd.read_csv(data_path)
    required_columns = FEATURE_COLUMNS + ["GCV"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")

    X = df[FEATURE_COLUMNS]
    y = df["GCV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(
        f"Mean Absolute Error: "
        f"{mean_absolute_error(y_test, predictions):.2f} kcal/kg"
    )
    print(f"R2 Score (Accuracy): {r2_score(y_test, predictions) * 100:.2f}%")

    joblib.dump(model, model_path)
    print(f"Model saved successfully as '{model_path.name}'!")


if __name__ == "__main__":
    train_model()
