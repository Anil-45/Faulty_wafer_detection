"""Tuner."""

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from ..logger import AppLogger


class ModelFinder:
    """Find the model with best accuracy and AUC score."""

    def __init__(self) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        self.logger = AppLogger().get_logger(f"{path}/logs/tuner.log")

    def get_best_params_for_random_forest(
        self, train_x, train_y
    ) -> RandomForestClassifier:
        """Get the parameters which give the best accuracy."""
        try:
            # initializing with different combination of parameters
            param_grid = {
                "n_estimators": [10, 50, 100, 130],
                "criterion": ["gini", "entropy"],
                "max_depth": range(2, 4, 1),
                "max_features": ["sqrt", "log2"],
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(
                estimator=RandomForestClassifier(),
                param_grid=param_grid,
                cv=5,
                verbose=3,
            )
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_["criterion"]
            max_depth = grid.best_params_["max_depth"]
            max_features = grid.best_params_["max_features"]
            n_estimators = grid.best_params_["n_estimators"]

            # creating a new model with the best parameters
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
            )
            # training the new model
            clf.fit(train_x, train_y)
            self.logger.info("Random Forest best:%s", str(grid.best_params_))
            return clf

        except Exception as exception:
            self.logger.error("get best params for Random Forest failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_best_params_for_xgboost(self, train_x, train_y) -> XGBClassifier:
        """Get the parameters which give the best accuracy."""
        try:
            # initializing with different combination of parameters
            param_grid_xgboost = {
                "learning_rate": [0.5, 0.1, 0.01, 0.001],
                "max_depth": [3, 5, 10, 20],
                "n_estimators": [10, 50, 100, 200],
            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(
                XGBClassifier(objective="binary:logistic"),
                param_grid_xgboost,
                verbose=3,
                cv=5,
            )
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_["learning_rate"]
            max_depth = grid.best_params_["max_depth"]
            n_estimators = grid.best_params_["n_estimators"]

            # creating a new model with the best parameters
            xgb = XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )
            # training the new model
            xgb.fit(train_x, train_y)
            self.logger.info("XGBoost best params: %s", str(grid.best_params_))
            return xgb

        except Exception as exception:
            self.logger.error("get best params for XGBoost failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """Get model which has the best AUC score."""
        try:
            # create best model for XGBoost
            model_xgb = self.get_best_params_for_xgboost(train_x, train_y)
            # Predictions using the XGBoost Model
            prediction_xgboost = model_xgb.predict(test_x)

            if 1 == len(test_y.unique()):
                # In case of single label, then roc_auc_score returns error.
                # We will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                self.logger.info("Accuracy for XGBoost:%s", str(xgboost_score))
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)
                self.logger.info("AUC for XGBoost:%s", str(xgboost_score))

            # create best model for Random Forest
            model_rf = self.get_best_params_for_random_forest(train_x, train_y)
            # prediction using the Random Forest Algorithm
            prediction_rf = model_rf.predict(test_x)

            if 1 == len(test_y.unique()):
                # In case of single label, roc_auc_score returns error.
                # We will use accuracy in that case
                rf_score = accuracy_score(test_y, prediction_rf)
                self.logger.info("Accuracy RandomForest:%s", str(rf_score))
            else:
                rf_score = roc_auc_score(test_y, prediction_rf)
                self.logger.info("AUC for RandomForest:%s", str(rf_score))

            # comparing the two models
            if rf_score < xgboost_score:
                return "XGBoost", model_xgb
            else:
                return "RandomForest", model_rf

        except Exception as exception:
            self.logger.error("get best model failed")
            self.logger.exception(exception)
            raise Exception from exception
