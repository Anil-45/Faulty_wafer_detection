"""Data Validation.

Performs validations as per Data Sharing Agreement schemas.
"""

import json
import os
import re
import shutil
from datetime import datetime
import pandas as pd

from . import constants as const
from ..logger import AppLogger


class DataValidation:
    """Class for handling all validations on raw data."""

    def __init__(self, folder_path: str, mode: str) -> None:
        """Initialize the required variables."""
        self.file_dir = folder_path
        self.date_stamp_len = const.DATE_STAMP_LEN
        self.time_stamp_len = const.TIME_STAMP_LEN
        self.path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."

        if "train" == str(mode):
            self.number_of_columns = const.NO_OF_COLUMNS
            self.schema_path = f"{self.path}/src/data/{const.STR_TRAIN_SCHEMA}"
            self.accepted_dir = f"{self.path}/data/interim/train/accepted/"
            self.rejected_dir = f"{self.path}/data/interim/train/rejected/"
            self.archive_dir = f"{self.path}/data/processed/train/archive"
        else:
            self.number_of_columns = int(const.NO_OF_COLUMNS) - 1
            self.schema_path = f"{self.path}/src/data/{const.STR_TEST_SCHEMA}"
            self.accepted_dir = f"{self.path}/data/interim/test/accepted/"
            self.rejected_dir = f"{self.path}/data/interim/test/rejected/"
            self.archive_dir = f"{self.path}/data/processed/test/archive"

        self.column_dict = dict()
        if not os.path.exists(f"{self.path}/logs/"):
            os.makedirs(f"{self.path}/logs/")
        logger_path = f"{self.path}/logs/data_validation.log"
        self.logger = AppLogger().get_logger(logger_path)

    def read_schema(self) -> None:
        """Read default schema."""
        try:
            with open(self.schema_path, "r", encoding="utf-8") as file:
                schema_dict = json.load(file)
                file.close()

            # file_name_pattern = schema_dict[const.STR_FILE_NAME]
            self.date_stamp_len = schema_dict[const.STR_DATE_STAMP_LEN]
            self.time_stamp_len = schema_dict[const.STR_TIME_STAMP_LEN]
            self.number_of_columns = schema_dict[const.STR_NO_OF_COLUMNS]
            self.column_dict = schema_dict[const.STR_COLUMN_DICT]

        except FileNotFoundError:
            self.logger.error("cannot open schema.json")
        except KeyError:
            self.logger.error("Key error %s", str(KeyError))
        except ValueError:
            self.logger.error("Value error %s", str(ValueError))

    def create_interim_data_dir(self) -> None:
        """Create data directories.

        Inside data/interim folder, two folder will be created, named
        accepted and rejected. If the data violates the data sharing
        agreement it will be rejected for training, else it will moved
        to accepted folder.

        Raises:
            OSError
        """
        try:
            # accepted data
            if not os.path.isdir(self.accepted_dir):
                os.makedirs(self.accepted_dir)

            # rejected data
            if not os.path.isdir(self.rejected_dir):
                os.makedirs(self.rejected_dir)

        except OSError as exception:
            self.logger.error("Error while creating directory for data")
            self.logger.exception(exception)
            raise OSError from exception

    def delete_interim_accepted_dir(self) -> None:
        """Delete interim accepted data directory.

        Raises:
            shutil.ExecError
        """
        try:
            # accepted data
            if os.path.isdir(self.accepted_dir):
                shutil.rmtree(self.accepted_dir)
                self.logger.info("deleted interim accepted data folder")

        except shutil.ExecError as exception:
            self.logger.error("error while deleting interim accepted dir")
            self.logger.exception(exception)
            raise shutil.ExecError from exception

    def delete_interim_rejected_dir(self) -> None:
        """Delete interim rejected data directory.

        Raises:
            shutil.ExecError
        """
        try:
            # rejected data
            if os.path.isdir(self.rejected_dir):
                shutil.rmtree(self.rejected_dir)
                self.logger.info("deleted interim rejected data folder")

        except shutil.ExecError as exception:
            self.logger.error("Error while deleting interim rejected data dir")
            self.logger.exception(exception)
            raise shutil.ExecError from exception

    def archive_rejected_data(self) -> None:
        """Archive the rejected data.

        Raises:
            OSError
        """
        date = datetime.now().date()
        time = datetime.now().strftime("%H%M%S")

        try:
            source = self.rejected_dir
            archive_dir = self.archive_dir
            if not os.path.isdir(archive_dir):
                os.makedirs(archive_dir)

            dest = f"{archive_dir}/archive_{date}_{time}"
            if not os.path.isdir(dest):
                os.makedirs(dest)
            files = os.listdir(source)
            for file in files:
                if file not in os.listdir(dest):
                    shutil.move(source + file, dest)

            self.logger.info("moved the rejected data to archive")
            self.delete_interim_rejected_dir()

        except OSError as exception:
            self.logger.exception(exception)
            raise OSError from exception

    def validate_filename(self) -> None:
        """Validate filename as per Data Sharing Agreement.

        It is moved to interim accepted if it follows DSA
        else moved to interim rejected.

        Raises:
            shutil.SameFileError
        """
        # delete the directories if already present.
        self.delete_interim_accepted_dir()
        self.delete_interim_rejected_dir()

        # create new directories
        self.create_interim_data_dir()
        files = list(os.listdir(self.file_dir))

        accepted_dir = self.accepted_dir
        rejected_dir = self.rejected_dir
        try:
            for filename in files:

                # checking file name pattern
                if re.match(const.STR_FILE_NAME_PATTERN, filename):
                    split_file_ext = re.split(".csv", filename)
                    split_date_time = re.split("_", split_file_ext[0])

                    # checking date stamp length
                    if len(split_date_time[1]) == self.date_stamp_len:

                        # checking time stamp length
                        if len(split_date_time[2]) == self.time_stamp_len:
                            shutil.copy(self.file_dir + filename, accepted_dir)
                            self.logger.info("file::%s:: accepted", filename)
                        else:
                            shutil.copy(self.file_dir + filename, rejected_dir)
                            self.logger.warning("file::%s::rejected", filename)
                            self.logger.warning(
                                "time stamp len expected %d but was %s",
                                self.time_stamp_len,
                                str(split_date_time[2]),
                            )
                    else:
                        shutil.copy(self.file_dir + filename, rejected_dir)
                        self.logger.warning("file::%s::rejected", filename)
                        self.logger.warning(
                            "date stamp len expected %d but was %s",
                            self.date_stamp_len,
                            str(split_date_time[1]),
                        )
                else:
                    shutil.copy(self.file_dir + filename, rejected_dir)
                    self.logger.warning("file::%s::rejected", filename)
                    self.logger.warning("violates file name pattern agreement")
        except shutil.SameFileError as exception:
            self.logger.exception(exception)
            raise shutil.SameFileError from exception
        except OSError as exception:
            self.logger.exception(exception)
            raise OSError from exception

    def validate_no_of_col(self) -> None:
        """Validate the number of columns.

        It is moved to interim accepted folder if it follows
        train_schema else moved to interim rejected folder.

        Raises:
            OSError
        """
        try:
            for file in os.listdir(self.accepted_dir):
                csv = pd.read_csv(self.accepted_dir + file)
                if csv.shape[1] == self.number_of_columns:
                    pass
                else:
                    shutil.move(self.accepted_dir + file, self.rejected_dir)
                    self.logger.warning("file::%s::rejected", file)
                    self.logger.warning(
                        "number of columns expected %d but was %d",
                        self.number_of_columns,
                        csv.shape[1],
                    )
        except OSError as exception:
            self.logger.exception(exception)
            raise OSError from exception

    def validate_columns(self) -> None:
        """Validate columns.

        If complete data in a column is empty then
        the file is moved to interim rejected dir.
        Renames unnamed column to Wafer.

        Raises:
            OSError
        """
        try:
            accepted_dir = self.accepted_dir
            for file in os.listdir(accepted_dir):
                csv = pd.read_csv(accepted_dir + file)
                count = 0
                for columns in csv:
                    # n_rows = len(csv[columns])
                    n_values = csv[columns].count()
                    if 0 == n_values:
                        count += 1
                        shutil.move(accepted_dir + file, self.rejected_dir)
                        self.logger.warning("file::%s::rejected", file)
                        self.logger.warning("entire column is empty")
                        break
                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv(
                        accepted_dir + file,
                        index=None,
                        header=True,
                    )
        except OSError as exception:
            self.logger.exception(exception)
            raise OSError from exception

    def validate_data_files(self) -> dict:
        """Validate all the data files in interim accepted dir.

        Performs filename check, number of columns check and
        empty column check.
        """
        self.read_schema()
        self.validate_filename()
        self.validate_no_of_col()
        self.validate_columns()
        return self.column_dict

    def delete_prediction_file(self) -> None:
        """Delete prediction file."""
        if os.path.exists(f"{self.path}/data/processed/test/predictions.csv"):
            os.remove(f"{self.path}/data/processed/test/predictions.csv")
