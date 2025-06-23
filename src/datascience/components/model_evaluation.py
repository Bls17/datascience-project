import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json


os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/Bls17/datascience-project.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = "Bls17"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "61087f41394d9ae01fd32f4ba34d8d0bd3cd9f48"


class ModelEvaluation:
    def __init__(self, config=ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mse = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mse, r2

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with file store
            # Model registry does not work with file store or unsupported backends (like DagsHub)
            if (
                tracking_url_type_store != "file"
                and "dagshub" not in self.config.mlflow_uri
            ):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="ElasticnetModel",
                )
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
