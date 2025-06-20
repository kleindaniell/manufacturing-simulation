from abc import ABC
import pandas as pd
from pathlib import Path
import numpy as np

from typing import Literal, List, Union


class Metrics:
    def __init__(
        self,
        experiment_folder: Path,
    ):
        if not isinstance(experiment_folder, Path):
            self.experiment_folder = Path(experiment_folder).resolve()

    def read_logs(
        self, log: Literal["general", "products", "resources", "all"] = "all"
    ) -> None:
        self.general_logs = (
            self._read_log_files("**/*general_log*")
            if log in ["general", "all"]
            else None
        )
        self.products_logs = (
            self._read_log_files("**/*products_log*")
            if log in ["products", "all"]
            else None
        )
        self.resources_logs = (
            self._read_log_files("**/*resources_log*")
            if log in ["resources", "all"]
            else None
        )

    def _read_log_files(self, glob_pattern) -> pd.DataFrame:
        df = pd.DataFrame()

        general_files = list(self.experiment_folder.rglob(glob_pattern))
        for path_ in general_files:
            file_path = Path(path_)
            if file_path.is_file():
                run = int(file_path.parent.stem.split("_")[-1])
                df_tmp = pd.read_csv(file_path, low_memory=False)
                df_tmp["run"] = run
                df = df_tmp.copy() if df.empty else pd.concat([df, df_tmp])

        return df
