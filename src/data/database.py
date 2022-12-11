"""Database."""

import shutil
import sqlite3
import os
import csv
import pandas as pd

from ..logger import AppLogger


class DatabaseOperation:
    """Handle all the SQL operations."""

    def __init__(self, mode: str) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        if not os.path.exists(f"{path}/logs/"):
            os.makedirs(f"{path}/logs/")
        self.logger = AppLogger().get_logger(f"{path}/logs/database.log")
        self.mode = mode
        if "train" == str(mode):
            self.path = f"{path}/data/processed/train/"
            self.rejected_dir = f"{path}/data/interim/train/rejected/"
            self.accepted_dir = f"{path}/data/interim/train/accepted/"
        else:
            self.path = f"{path}/data/processed/test/"
            self.rejected_dir = f"{path}/data/interim/test/rejected/"
            self.accepted_dir = f"{path}/data/interim/test/accepted/"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.accepted_dir):
            os.makedirs(self.accepted_dir)

        if not os.path.exists(self.rejected_dir):
            os.makedirs(self.rejected_dir)

    def db_connection(self, db_name: str) -> sqlite3.Connection:
        """Database Connection.

        Creates database if it doesn't exists and
        then opens the connection to the DB.
        """
        try:
            self.logger.info("opening %s%s.db", self.path, db_name)
            connection = sqlite3.connect(f"{self.path}{db_name}.db")
            self.logger.info("%s database opened successfully", db_name)
            return connection
        except ConnectionError as connection_error:
            self.logger.error("error while connecting to db: %s", db_name)
            self.logger.exception(connection_error)
            raise ConnectionError from connection_error

    def db_create_table(self, db_name: str, column_names: dict) -> None:
        """Create table.

        Create a table in the given database which will
        be used to insert the accepted after validation.
        """
        try:
            connection = self.db_connection(db_name=db_name)
            if "train" == str(self.mode):
                # if train, append the data to old data
                query = (
                    "SELECT count(name)  FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'accepted'"
                )
                cur = connection.cursor()
                cur.execute(query)
                if 1 == cur.fetchone()[0]:
                    connection.close()
                    self.logger.info("table already exists")
                    return
            else:
                # if test, we can remove old data
                query = "DROP TABLE IF EXISTS accepted;"
                connection.execute(query)

            query = ""

            for key, value in column_names.items():
                query += f"'{key}' {value},"

            query = query[:-1]
            query = f"CREATE TABLE accepted ({query});"
            connection.execute(query)
            connection.close()
            self.logger.info("table created successfully")
            self.logger.info("closed %s database successfully", db_name)

        except sqlite3.Error as exception:
            self.logger.exception(exception)
            raise Exception from exception

        except Exception as exception:
            self.logger.error("error while creating table")
            self.logger.exception(exception)
            connection.close()
            self.logger.info("closed %s successfully", db_name)
            raise Exception from exception

    def db_insert_to_table(self, db_name: str) -> None:
        """Insert the accepted files into the above created table."""
        connection = self.db_connection(db_name=db_name)
        files = os.listdir(self.accepted_dir)
        for file in files:
            file_path = str(self.accepted_dir) + str(file)
            try:
                data = pd.read_csv(file_path)
                data.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                data.rename(columns={"Good/Bad": "Output"}, inplace=True)
                data.to_sql(
                    "accepted",
                    connection,
                    if_exists="append",
                    index=False
                )

            except Exception as exception:
                self.logger.exception(exception)
                connection.rollback()
                self.logger.info("error while inserting to table")
                shutil.move(file_path, self.rejected_dir)
                self.logger.info("%s file moved to rejected dir", file)
                connection.close()

        connection.close()

    def db_select_data_to_csv(self, db_name: str) -> None:
        """Export the data in accepted table as a CSV file."""
        try:
            connection = self.db_connection(db_name=db_name)
            query = "SELECT * FROM accepted"
            cursor = connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()

            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            # Make the CSV output directory
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

            file_name = f"{self.path}input.csv"
            # Open CSV file for writing.
            csv_file = csv.writer(
                open(file_name, "w", newline="", encoding="utf-8"),
                delimiter=",",
                lineterminator="\r\n",
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )

            # Add the headers and data to the CSV file.
            csv_file.writerow(headers)
            csv_file.writerows(results)

            self.logger.info("exported file successfully")

        except Exception as exception:
            self.logger.info("file export failed")
            self.logger.exception(exception)
