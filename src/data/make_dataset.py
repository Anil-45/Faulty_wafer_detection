"""Make dataset."""
import os

from ..logger import AppLogger
from .data_validation import DataValidation
from .database import DatabaseOperation
from .transform import DataTransform


def make_dataset(path: str, mode: str) -> None:
    """Make dataset.

    Args:
        path (str): raw data path
        mode (str): train or test

    Raises:
        Exception
    """
    data_val = DataValidation(path, mode)
    data_transform = DataTransform(mode)
    database = DatabaseOperation(mode)
    path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
    logger = AppLogger().get_logger(f"{path}/logs/make_dataset.log")

    try:
        logger.info("validation of %s raw data started", mode)
        # perform validations based on train_schema
        col_names = data_val.validate_data_files()
        logger.info("validation of %s raw data complete", mode)
        logger.info("no of columns is %s", str(len(col_names)))

        logger.info("data transformation started")
        # replacing blanks in the csv file with "Null"
        data_transform.replace_missing_with_null()
        logger.info("data transformation completed")

        logger.info("creating %sing database and tables.", mode)
        # Create table with columns given in schema
        database.db_create_table(mode, column_names=col_names)
        logger.info("table creation completed")

        logger.info("insertion of data into table started")
        # insert csv files in the table
        database.db_insert_to_table(mode)
        logger.info("insertion of data completed")

        logger.info("deleting interim accepted folder")
        # Delete the accepted folder after loading files in table
        data_val.delete_interim_accepted_dir()
        logger.info("%s accepted folder deleted", mode)

        logger.info("moving rejected files to archive")
        logger.info("deleting interim rejected folder")
        # Move the rejected files to archive folder
        data_val.archive_rejected_data()
        logger.info("archived rejected data")
        logger.info("deleted interim rejected folder")

        logger.info("extracting csv file from table")
        # export data in table to csv file
        database.db_export_data_to_csv(mode)

    except Exception as exception:
        logger.exception(exception)
        raise Exception from exception


if "__main__" == __name__:
    base_path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
    # make_dataset(f"{base_path}/data/raw/train/", "train")
    make_dataset(f"{base_path}/data/raw/test/", "test")
