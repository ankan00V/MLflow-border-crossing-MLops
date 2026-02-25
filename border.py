# BorderFlow MLflow Multi-Model Example

import warnings
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# ==============================
# Metrics function (Regression)
# ==============================
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # ==============================
    # Load dataset
    # ==============================
    try:
        data = pd.read_csv("Border_Crossing_Entry_Data.csv")
    except Exception as e:
        logger.exception("Unable to load CSV. Error: %s", e)

    # ==============================
    # Feature Engineering
    # ==============================
    data["Date"] = pd.to_datetime(data["Date"])

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["IsWeekend"] = (data["Date"].dt.weekday >= 5).astype(int)

    categorical_cols = ["Border", "Port Name", "Measure"]

    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    features = ["Border", "Port Name", "Measure", "Year", "Month", "Day", "IsWeekend"]
    target = "Value"

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_x = train[features]
    test_x = test[features]
    train_y = train[target]
    test_y = test[target]

    # ==============================
    # MLflow Experiment
    # ==============================
    mlflow.set_experiment("BorderFlow_Experiment")

    models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(),
        "SVR": SVR(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
    }

    # ==============================
    # Train + Log each model
    # ==============================
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            model.fit(train_x, train_y)
            predictions = model.predict(test_x)

            rmse, mae, r2 = eval_metrics(test_y, predictions)

            print(f"\n{model_name} Results:")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2: {r2}")

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=f"{model_name}_BorderModel",
                )
            else:
                mlflow.sklearn.log_model(model, "model")

    print("\n✅ All models trained and logged to MLflow.")
