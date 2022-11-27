"""This is the Entry point for Training the Machine Learning Model."""

import os
from sklearn.model_selection import train_test_split

from .clustering import KMeansClustering
from .tuner import ModelFinder
from .utils import Utils

from ..features import Preprocessor
from ..data import DataLoader
from ..data import DataValidation
from ..data import make_dataset
from ..logger import AppLogger

MODE = "train"


class Train:
    """Class to train model."""

    def __init__(self, train_path: str) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        if not os.path.exists(f"{path}/logs/"):
            os.makedirs(f"{path}/logs/")
        self.logger = AppLogger().get_logger(f"{path}/logs/train_model.log")
        if train_path is not None:
            self.pred_data_val = DataValidation(train_path, mode=MODE)
            make_dataset(train_path, MODE)

    def train(self) -> None:
        """Train the model."""
        # Logging the start of Training
        self.logger.info("Start of Training")
        try:
            # Getting the data from the source
            data_loader = DataLoader(mode=MODE)
            data = data_loader.get_data()

            # preprocessing
            preprocessor = Preprocessor(mode=MODE)
            # remove the unnamed column as it doesn't contribute to prediction.
            data = preprocessor.remove_columns(data, ["Wafer"])

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, "Output")
            Y = Y.replace({-1: 0})
            # Y = np.array(Y).reshape(0, 1)
            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                # missing value imputation
                X = preprocessor.impute_missing_values(X)

            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero,
            # it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop
            cols_to_drop = preprocessor.get_cols_with_zero_std_dev(X)

            # drop the columns obtained above
            X = preprocessor.remove_columns(X, cols_to_drop)

            # Applying the clustering approach
            kmeans = KMeansClustering()
            #  using the elbow plot to find the number of optimum clusters
            number_of_clusters = kmeans.elbow_plot(X)

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset
            # consisting of the corresponding cluster assignments.
            X["Labels"] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X["Cluster"].unique()

            # parsing all the clusters and looking for the best ML
            # algorithm to fit on individual cluster

            for i in list_of_clusters:
                # filter the data for one cluster
                cluster_data = X[X["Cluster"] == i]

                # Prepare the feature and Label columns
                features = cluster_data.drop(["Labels", "Cluster"], axis=1)
                label = cluster_data["Labels"]

                # splitting the data into training and
                # test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(
                    features, label, test_size=0.3, random_state=355
                )

                model_finder = ModelFinder()

                # getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(
                    x_train, y_train, x_test, y_test
                )

                # saving the best model to the directory.
                utils = Utils()
                utils.save_model(best_model, best_model_name + str(i))

            # logging the successful Training
            self.logger.info("Successful End of Training")

        except Exception as exception:
            # logging the unsuccessful Training
            self.logger.error("Unsuccessful End of Training")
            self.logger.exception(exception)
            raise Exception from exception
