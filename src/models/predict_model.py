"""Predict using model."""
import os

import pandas as pd

from ..data import DataLoader, DataValidation, make_dataset
from ..features import Preprocessor
from ..logger import AppLogger
from .constants import COLUMNS_WITH_ZERO_STD_DEV
from .utils import Utils

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

    def predict(self) -> pd.DataFrame:
        """Predict using saved model."""
        try:
            # deletes the existing prediction file from last run
            self.pred_data_val.delete_prediction_file()
            self.logger.info("start of prediction")
            data = DataLoader("test").get_data()

            # create a copy of actual data to save to prediction file
            data_copy = data.copy()

            preprocessor = Preprocessor(mode=MODE)

            # handle if any  null values exist
            is_null_present = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            # columns with zero std are found during EDA
            # and stored to COLUMNS_WITH_ZERO_STD_DEV
            cols_to_drop = COLUMNS_WITH_ZERO_STD_DEV
            data = preprocessor.remove_columns(data, cols_to_drop)

            # load the clustering model
            utils = Utils()
            kmeans = utils.load_model("KMeans")

            # drops the first column for cluster prediction
            clusters = kmeans.predict(data.drop(["Wafer"], axis=1))

            # append cluster data
            data["clusters"] = clusters
            data_copy["clusters"] = clusters
            clusters = data["clusters"].unique()
            path = f"{self.base}/data/processed/test/Predictions.csv"
            header_flag = True

            for i in clusters:
                # grab only corresponding cluster data
                cluster_data = pd.DataFrame(data[data["clusters"] == i])
                result = pd.DataFrame(data_copy[data_copy["clusters"] == i])
                result.drop(columns=["clusters"], inplace=True)
                cluster_data.drop(columns=["clusters", "Wafer"], inplace=True)

                # find and load model
                model_name = utils.find_model_file(i)
                self.logger.info("model_name %s", str(model_name))
                model = utils.load_model(model_name)

                # predict and store results
                result["Prediction"] = list(model.predict(cluster_data))
                result["Prediction"] = result["Prediction"].astype(int)

                # change back to correct class
                result.Prediction.replace({0: -1}, inplace=True)

                # appends result to prediction file
                result.to_csv(path, header=header_flag, mode="a+", index=False)
                # add headers only first time
                header_flag = False

            data = pd.read_csv(path)
            data = data[data["Prediction"].astype(int) == 1]
            data = data.reset_index(drop=True)
            data = data[["Wafer", "Prediction"]]
            self.logger.info("End of Prediction")

        except Exception as exception:
            self.logger.error("error occurred while running the prediction")
            self.logger.exception(exception)
            raise Exception from exception

        return data
