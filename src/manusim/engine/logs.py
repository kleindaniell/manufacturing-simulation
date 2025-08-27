from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from pathlib import Path


class Logger:
    """Base class for event logging"""

    def __init__(self, logs_save_path: str = None, mem_size: int = 10000):
        self.mem_size = mem_size
        self.log_index = {}
        self.logs_save_path = logs_save_path
        if logs_save_path:
            self.logs_save_path = Path(logs_save_path)

    def create_log(self, variable: str, keys: list[str]):
        """Create a new attribute with multiple sub-keys"""

        # if attribute already exists, keep it and update
        existing = getattr(self, variable, False)
        if not existing:
            var_dict = {}
            self.log_index[variable] = {}

            for key in keys:
                var_dict[key] = np.zeros((self.mem_size, 2), dtype=np.float32)
                self.log_index[variable][key] = 0

        setattr(self, variable, var_dict)

    def log(self, variable: str, key: str, value: Tuple[float, float]):
        """
        Logs a new value for a given variable and key.
        Args:
            variable (str): The name of the variable to log.
            key (str): The key for the log entry (e.g., product name).
            value (Tuple[float, float]): The value to log (time, value).
        """
        index = self.log_index[variable][key] % self.mem_size

        if self.log_index[variable][key] >= self.mem_size and self.logs_save_path:
            self.save_logs_to_file(variable, key)

        getattr(self, variable)[key][index] = value

        self.log_index[variable][key] = index + 1

    def get_log(self, variable: str, key: str) -> np.ndarray:
        return getattr(self, variable)[key]

    def get_variable_logs(self, variable: str, saved_logs=False) -> pd.DataFrame:

        df_list = []
        # Read saved logs
        if saved_logs:
            df_saved = self.read_saved_logs(variable)
            if not df_saved.empty:
                df_list.append(df_saved)

        # Get memory logs
        variable_logs = getattr(self, variable)
        for key in variable_logs.keys():
            df_temp = self.get_logs(variable, key)
            df_list.append(df_temp)

        return pd.concat(df_list, ignore_index=True)

    def get_logs(self, variable: str, key: str) -> pd.DataFrame:

        # Read memory logs
        df = pd.DataFrame(
            getattr(self, variable)[key][: self.log_index[variable][key]],
            columns=["time", "value"],
        )
        df["variable"] = variable
        df["key"] = key

        return df

    def save_logs_to_file(self, variable: str, key: str):
        """
        Saves the logs for a given variable and key to a file.
        """
        self.logs_save_path.mkdir(exist_ok=True, parents=True)

        file_save_path = Path(f"{self.logs_save_path}/log_{variable}_{key}.csv")
        header = True
        if file_save_path.exists():
            header = False

        df = self.get_logs(variable, key)
        df.to_csv(file_save_path, index=False, header=header, mode="a")
        self.restart_log(variable, key)

    def restart_log(self, variable: str, key: str):
        self.log_index[variable][key] = 0
        getattr(self, variable)[key] = np.zeros((self.mem_size, 2), dtype=np.float32)

    def _log_dict_to_df(self, field_name: str = None) -> pd.DataFrame:
        df_list = []
        for attr, values in self.__dict__.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    if self.log_index[attr][key]:
                        df_tmp = pd.DataFrame(
                            val[: self.log_index[attr][key]], columns=["time", "value"]
                        )
                        df_tmp["variable"] = attr
                        if field_name:
                            df_tmp[field_name] = key
                        df_list.append(df_tmp)

        return (
            pd.concat(df_list, ignore_index=True)
            if df_list
            else pd.DataFrame(columns=["time", "value", "variable", field_name])
        )

    def save_all_logs(self):

        for variable, value in self.log_index.items():
            for key in value.keys():
                self.save_logs_to_file(variable=variable, key=key)

    def read_saved_logs(self, variable: str) -> pd.DataFrame:

        file_list = list(self.logs_save_path.rglob(f"**/*log_{variable}*.csv"))
        df_list = []
        if len(file_list) > 0:
            for file_path in file_list:
                if file_path.is_file():
                    df_tmp = pd.read_csv(file_path, low_memory=False)
                    if not df_tmp.empty:
                        df_list.append(df_tmp)

        if len(df_list) > 0:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()
