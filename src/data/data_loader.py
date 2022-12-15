"""Data loader."""
import os

import pandas as pd


class DataLoader:
    """Load the data."""

    def __init__(self, mode: str) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        if "train" == str(mode):
            self.file_path = f"{path}/data/processed/train/input.csv"
        else:
            self.file_path = f"{path}/data/processed/test/input.csv"

    def get_data(self) -> pd.DataFrame:
        """Get data."""
        try:
            data = pd.read_csv(self.file_path)
            return data
        except Exception as exception:
            raise Exception from exception
