"""Predict using model."""
import os
import pandas as pd

from .utils import Utils
from ..features import Preprocessor
from ..data import DataLoader
from ..data import DataValidation
from ..data import make_dataset
from ..logger import AppLogger

MODE = "test"


class Prediction:
    """Class for Prediction."""

    def __init__(self, path: str) -> None:
        """Initialize required variables."""
        self.base = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        if not os.path.exists(f"{self.base}/logs/"):
            os.makedirs(f"{self.base}/logs/")
        logger_path = f"{self.base}/logs/prediction.log"
        self.logger = AppLogger().get_logger(logger_path)
        if path is not None:
            self.pred_data_val = DataValidation(path, mode=MODE)
            make_dataset(path, MODE)

    def predict(self):
        """Predict using saved model."""
        try:
            # deletes the existing prediction file from last run
            self.pred_data_val.delete_prediction_file()
            self.logger.info("start of prediction")
            data = DataLoader("test").get_data()

            preprocessor = Preprocessor(mode=MODE)
            is_null_present = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            cols_to_drop = preprocessor.get_cols_with_zero_std_dev(data)
            data = preprocessor.remove_columns(data, cols_to_drop)
            utils = Utils()
            kmeans = utils.load_model("KMeans")

            clusters = kmeans.predict(
                data.drop(["Wafer"], axis=1)
            )  # drops the first column for cluster prediction

            data["clusters"] = clusters
            clusters = data["clusters"].unique()
            path = f"{self.base}/data/processed/test/Predictions.csv"
            for i in clusters:
                cluster_data = data[data["clusters"] == i]
                wafer_names = list(cluster_data["Wafer"])
                cluster_data = data.drop(labels=["Wafer"], axis=1)
                cluster_data = cluster_data.drop(["clusters"], axis=1)
                model_name = utils.find_model_file(i)
                self.logger.info("model_name %s", str(model_name))
                model = utils.load_model(model_name)
                result = list(model.predict(cluster_data))
                col_names = ["Wafer", "Prediction"]
                result_data = list(zip(wafer_names, result))
                result = pd.DataFrame(result_data, columns=col_names)
                result["Wafer"] = result["Wafer"].astype(int)
                result["Prediction"] = result["Prediction"].astype(int)
                result.Prediction.replace({0: -1}, inplace=True)
                # appends result to prediction file
                result.to_csv(path, header=True, mode="a+", index=False)

            data = pd.read_csv(path)
            data = data.sort_values(by="Wafer")
            data.drop_duplicates(inplace=True, keep=False)
            data = data[data["Prediction"].astype(int) == 1]
            data = data.reset_index(drop=True)
            self.logger.info("End of Prediction")

        except Exception as exception:
            self.logger.error("error occurred while running the prediction")
            self.logger.exception(exception)
            raise Exception from exception

        return path, data
