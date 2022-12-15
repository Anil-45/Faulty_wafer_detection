"""Model Utils."""

import os
import pickle
import shutil

from ..logger import AppLogger


class Utils:
    """It provides the functions to save and load models."""

    def __init__(self) -> None:
        """Initialize the required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        self.logger = AppLogger().get_logger(f"{path}/logs/utils.log")
        self.model_directory = f"{path}/models/"
        self.cluster_number = 0
        self.list_of_model_files = []
        self.list_of_files = []

    def save_model(self, model, filename: str) -> None:
        """Save the model to a file.

        Raises:
            Exception
        """
        try:
            path = os.path.join(self.model_directory, filename)

            # create separate directory for each cluster
            if os.path.isdir(path):
                # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)

            with open(path + "/" + filename + ".sav", "wb") as file:
                pickle.dump(model, file)

            self.logger.info("model saved to %s.sav", filename)

        except Exception as exception:
            self.logger.error("model save %s failed", filename)
            self.logger.exception(exception)
            raise Exception from exception

    def load_model(self, filename) -> pickle:
        """Load model from file to memory.

        Raises:
            Exception
        """
        try:
            name = f"{self.model_directory}{filename}/{filename}.sav"
            self.logger.info("loading %s", str(name))
            with open(name, "rb") as f:
                self.logger.info("model loaded from %s", filename)
                return pickle.load(f)

        except Exception as exception:
            self.logger.error("model load %s failed", filename)
            self.logger.exception(exception)
            raise Exception from exception

    def find_model_file(self, cluster_number: int) -> str:
        """Find model file based on cluster number."""
        try:
            self.cluster_number = cluster_number
            self.list_of_files = os.listdir(self.model_directory)

            for file in self.list_of_files:
                try:
                    if file.index(str(self.cluster_number)) != -1:
                        model_name = file
                except ValueError:
                    continue

            model_name = model_name.split(".")[0]
            self.logger.info(model_name)
            return model_name

        except Exception as exception:
            self.logger.error("find model failed")
            self.logger.exception(exception)
            raise Exception from exception
