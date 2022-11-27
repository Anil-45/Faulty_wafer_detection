"""Clustering."""

import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from .utils import Utils
from . import constants as const
from ..logger import AppLogger

matplotlib.use("agg")


class KMeansClustering:
    """KMeans Wrapper."""

    def __init__(self) -> None:
        """Initialize required variables."""
        self.path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        if not os.path.exists(f"{self.path}/logs/"):
            os.makedirs(f"{self.path}/logs/")
        logger_path = f"{self.path}/logs/clustering.log"
        self.logger = AppLogger().get_logger(logger_path)

    def elbow_plot(self, data) -> int:
        """Elbow plot to decide no of clusters.

        Raises:
            Exception
        """
        wcss = list()
        try:
            for i in range(1, const.MAX_NUMBER_OF_CLUSTERS):
                model = KMeans(n_clusters=i, init="k-means++", random_state=42)
                model.fit(data)
                wcss.append(model.inertia_)

            plt.plot(range(1, const.MAX_NUMBER_OF_CLUSTERS), wcss)

            # plot elbow
            plt.title("Elbow Plot")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            plt.savefig(f"{self.path}/reports/figures/K_Means_Elbow.PNG")

            # finding optimum value
            knee = KneeLocator(
                range(1, const.MAX_NUMBER_OF_CLUSTERS),
                wcss,
                curve="convex",
                direction="decreasing",
            )
            self.logger.info("optimum number of clusters:%s", str(knee.knee))
            return knee.knee

        except Exception as exception:
            self.logger.error("Elbow plot failed")
            self.logger.exception(exception)
            raise Exception from exception

    def create_clusters(self, data, number_of_clusters: int):
        """Create clusters.

        Raises:
            Exception
        """
        try:
            kmeans = KMeans(
                n_clusters=number_of_clusters, init="k-means++", random_state=4
            )

            # fit and predict
            y_kmeans = kmeans.fit_predict(data)

            # saving the model
            Utils().save_model(kmeans, "KMeans")

            # store cluster information
            data["Cluster"] = y_kmeans
            self.logger.info("%s clusters created", str(number_of_clusters))
            return data

        except Exception as exception:
            self.logger.error("create clusters failed")
            self.logger.exception(exception)
            raise Exception from exception
