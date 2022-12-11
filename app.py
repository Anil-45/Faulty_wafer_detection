"""Application entry point."""

from wsgiref import simple_server

import os
from datetime import datetime
from flask import Flask, request, render_template, Response
from flask import send_from_directory
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from werkzeug.utils import secure_filename

from src.models.train_model import Train
from src.models.predict_model import Prediction
from src.logger import AppLogger

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

HOST = "0.0.0.0"
PORT = 5000
PRED_FOLDER = "data/processed/test/"
PRED_FILE = "Predictions.csv"
FILE_EXT = set(["csv"])
EXTERNAL_TRAIN = "data/external/train/"
EXTERNAL_TEST = "data/external/test/"
DEFAULT_TRAIN = "data/raw/train/"
DEFAULT_TEST = "data/raw/test/"

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=["GET"])
@cross_origin()
def home_page():
    """Home Page."""
    return render_template("index.html")


def _train(path: str) -> Response:
    """Train Model implementation."""
    try:
        if 0 == _get_file_count(path=path):
            return Response("No files found to train")
        else:
            logger.info("Started training")
            Train(path).train()
            logger.info("Training complete")
    except Exception as exception:
        logger.exception(exception)
        return Response("Something went wrong! %s\n", str(exception))
    return Response("Training completed.")


@app.route("/train", methods=["POST"])
@cross_origin()
def default_train():
    """Train Model.

    Performs data validation, handling and trains model.
    """
    return _train(path=DEFAULT_TRAIN)


def _predict(path: str) -> Response:
    """Predict from model."""
    try:
        if 0 == _get_file_count(path=path):
            return Response("No files found to predict")
        else:
            logger.info("Started Prediction")
            logger.info(path)
            path, predictions = Prediction(path=path).predict()
            logger.info("Completed Prediction")
            return Response(predictions.to_html())
    except Exception as exception:
        return Response("Something went wrong! %s\n" % str(exception))


@app.route("/predict", methods=["POST"])
@cross_origin()
def default_predict():
    """Predict from Model.

    Performs data validation, handling and prediction.
    """
    return _predict(path=DEFAULT_TEST)


@app.route("/download", methods=["GET"])
@cross_origin()
def download_predictions():
    """Download prediction."""
    file = os.path.join(PRED_FOLDER, PRED_FILE)
    if os.path.exists(file):
        logger.info("Sending predictions file.")
        return send_from_directory(PRED_FOLDER, PRED_FILE, as_attachment=True)
    else:
        return Response("No prediction file found to download")


def _is_file_ext_supported(filename):
    """Check extension of file."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in FILE_EXT


def _get_datetime() -> str:
    """Get date in DDMMYYYY format."""
    return datetime.now().strftime("%d%m%Y_%H%M%S")


def _get_filename() -> str:
    """Get filename as per DSA."""
    return f"wafer_{_get_datetime()}.csv"


def _get_file_count(path: str) -> int:
    """Get count of number of files."""
    if not os.path.exists(path):
        return 0
    else:
        return len(os.listdir(path))


@app.route("/upload_train", methods=["GET", "POST"])
@cross_origin()
def upload_and_train() -> Response:
    """Upload files and train."""
    if request.method == "POST":
        if not os.path.exists(EXTERNAL_TRAIN):
            os.makedirs(EXTERNAL_TRAIN, mode=664)
        else:
            for file in os.listdir(EXTERNAL_TRAIN):
                os.remove(os.path.join(EXTERNAL_TRAIN, file))

        files = request.files.getlist("files")
        logger.info("Uploaded %s files", str(len(files)))
        for file in files:
            if _is_file_ext_supported(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(EXTERNAL_TRAIN, filename)
                file.save(path)
        return _train(path=EXTERNAL_TRAIN)


@app.route("/upload_test", methods=["GET", "POST"])
@cross_origin()
def upload_and_test() -> Response:
    """Upload files and predict."""
    if request.method == "POST":
        if not os.path.exists(EXTERNAL_TEST):
            os.makedirs(EXTERNAL_TEST)
        else:
            for file in os.listdir(EXTERNAL_TEST):
                os.remove(os.path.join(EXTERNAL_TEST, file))

        files = request.files.getlist("files")
        logger.info("Uploaded %s files", str(len(files)))
        for file in files:
            if _is_file_ext_supported(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(EXTERNAL_TEST, filename)
                file.save(path)

        return _predict(path=EXTERNAL_TEST)


if "__main__" == __name__:
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    logger = AppLogger().get_logger("logs/app.log")

    port = int(os.getenv("PORT", PORT))
    server = simple_server.make_server(HOST, port=port, app=app)
    server.serve_forever()
