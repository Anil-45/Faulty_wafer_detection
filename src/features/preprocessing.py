"""Preprocessing."""

import os
from pandas import DataFrame
import numpy as np
from sklearn.impute import KNNImputer

from ..logger import AppLogger


class Preprocessor:
    """Class to clean and transform the data before training."""

    def __init__(self, mode: str) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        if not os.path.exists(f"{path}/logs/"):
            os.makedirs(f"{path}/logs/")
        self.logger = AppLogger().get_logger(f"{path}/logs/preprocessing.log")

        if "train" == str(mode):
            self.processed_path = f"{path}/data/processed/train/"
        else:
            self.processed_path = f"{path}/data/processed/test/"

    def remove_columns(self, data: DataFrame, columns: list) -> DataFrame:
        """Remove the given columns from a pandas dataframe."""
        try:
            useful_data = data.drop(labels=columns, axis=1)
            self.logger.info("columns removal success")
            return useful_data

        except Exception as exception:
            self.logger.error("remove columns failed")
            self.logger.exception(exception)
            raise Exception from exception

    def separate_label_feature(self, data: DataFrame, label_column_name: str):
        """Separate the features and label columns."""
        try:
            X = data.drop(
                labels=label_column_name, axis=1
            )  # drop the columns specified and separate the feature columns
            Y = data[label_column_name]  # Filter the Label columns
            self.logger.info("label separation Successful")
            return X, Y

        except Exception as exception:
            self.logger.error("separate label features failed")
            self.logger.exception(exception)
            raise Exception from exception

    def is_null_present(self, data: DataFrame) -> bool:
        """Check the presence of null values."""
        try:
            null_present = False
            null_counts = data.isna().sum()
            for null_count in null_counts:
                if null_count > 0:
                    null_present = True
                    break
            if null_present:
                df_with_null = DataFrame()
                df_with_null["columns"] = data.columns
                df_with_null["n_missing"] = np.asarray(data.isna().sum())
                df_with_null.to_csv(f"{self.processed_path}null_values.csv")
            return null_present

        except Exception as exception:
            self.logger.error("null check failed")
            self.logger.exception(exception)
            raise Exception from exception

    def impute_missing_values(self, data: DataFrame) -> DataFrame:
        """Replace all the missing values using KNN Imputer."""
        try:
            imputer = KNNImputer(
                n_neighbors=3, weights="uniform", missing_values=np.nan
            )
            # impute the missing values
            data_mod = imputer.fit_transform(data)
            # convert the nd-array returned in the step above to a Dataframe
            df_mod = DataFrame(data=data_mod, columns=data.columns)
            self.logger.info("Imputing missing values successful")
            return df_mod

        except Exception as exception:
            self.logger.error("impute missing values failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_cols_with_zero_std_dev(self, data: DataFrame) -> list:
        """Find out the columns which have a standard deviation of zero."""
        data_description = data.describe()
        col_with_zero_std = list()
        try:
            for col in data.columns:
                # check if standard deviation is zero
                if 0 == data_description[col]["std"]:
                    col_with_zero_std.append(col)
            return col_with_zero_std

        except Exception as exception:
            self.logger.error("get columns with zero std failed")
            self.logger.exception(exception)
            raise Exception from exception
