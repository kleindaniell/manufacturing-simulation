from abc import ABC
from enum import Enum
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from scipy import stats


class MetricProducts(str, Enum):
    DELIVERED_ONTIME = "delivered_ontime"
    DELIVERED_LATE: str = "delivered_late"
    LOST_SALES: str = "lost_sales"
    FLOW_TIME: str = "flow_time"
    LEAD_TIME: str = "lead_time"
    TARDINESS: str = "tardiness"
    EARLINESS: str = "earliness"
    WIP: str = "wip"
    FINISHED_GOODS: str = "finished_goods"
    RELEASED: str = "released"


class MetricResources(str, Enum):
    UTILIZATION: str = "utilization"
    BREAKDOWN: str = "breakdown"
    SETUP: str = "setup"
    QUEUE: str = "queue"


class ExperimentMetrics:
    def __init__(self, experiment_folder: Path, custom_metrics: list = None):
        self.experiment_folder = experiment_folder
        if not isinstance(self.experiment_folder, Path):
            self.experiment_folder = Path(self.experiment_folder).resolve()
        self.custom_metrics = custom_metrics
        self.logs = pd.DataFrame()

    def read_logs(
        self, log: Literal["general", "products", "resources", "all"] = "all"
    ) -> None:

        df_list = []
        if log in ["products", "all"]:
            for metric in MetricProducts:
                glob_pattern = f"**/*log_{metric.value}*"
                df_tmp = self._read_log_files(glob_pattern)
                if not df_tmp.empty:
                    df_list.append(df_tmp)

        if log in ["resources", "all"]:
            for metric in MetricResources:
                glob_pattern = f"**/*log_{metric.value}*"
                df_tmp = self._read_log_files(glob_pattern)
                if not df_tmp.empty:
                    df_list.append(df_tmp)

        if log in ["general", "all"] and self.custom_metrics:
            for metric in self.custom_metrics:
                glob_pattern = f"**/*log_{metric}*"
                df_tmp = self._read_log_files(glob_pattern)
                if not df_tmp.empty:
                    df_list.append(df_tmp)

        self.logs = pd.concat(df_list, ignore_index=True)
        self.logs["experiment"] = self.experiment_folder.name

    def _read_log_files(self, glob_pattern) -> pd.DataFrame:
        file_list = list(self.experiment_folder.rglob(glob_pattern))
        
        if len(file_list) > 0:
            df_list = []
            for file_path in file_list:
                if file_path.is_file():
                    df_tmp = pd.read_csv(file_path, low_memory=False)
                    if not df_tmp.empty:
                        run = int(file_path.parent.parent.stem.split("_")[-1])
                        df_tmp["run"] = run
                        df_list.append(df_tmp)

        if len(df_list)>0:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_means(self) -> pd.DataFrame:

        self.means = (
            self.logs.drop(["time", "run"], axis=1)
            .groupby(["variable", "key", "experiment"])
            .mean()
            .reset_index()
        )

        return self.means

    def calculate_stats(
        self,
        confidence: float = 0.95,
        precision: float = 0.05,
        variables: list = None,
        aggregate_variables: bool = True,
    ) -> pd.DataFrame:

        df_list = []
        if variables:
            metric_list = variables
        else:
            metric_list = self.logs["variable"].unique()

        for metric in metric_list:
            metric_df = self.logs.loc[self.logs["variable"] == metric]

            if aggregate_variables:
                metric_df = (
                    metric_df.drop(["time", "key"], axis=1)
                    .groupby(["experiment", "variable", "run"])
                    .mean()
                    .reset_index()
                )
                metric_stats = metric_df.loc[[0], ["experiment", "variable"]]

                result = self._calculate_stats(
                    metric_df["value"], confidence, precision
                )
                metric_stats = pd.concat([metric_stats, pd.DataFrame(result)], axis=1)

                df_list.append(metric_stats)

            else:
                for key in metric_df["key"].unique():
                    key_df = (
                        metric_df.loc[metric_df["key"] == key]
                        .drop("time", axis=1)
                        .groupby(["experiment", "variable", "run", "key"])
                        .mean()
                        .reset_index()
                    )

                    metric_stats = key_df.loc[[0], ["experiment", "variable", "key"]]

                    result = self._calculate_stats(
                        key_df["value"], confidence, precision
                    )
                    metric_stats = pd.concat(
                        [metric_stats, pd.DataFrame(result)], axis=1
                    )

                    df_list.append(metric_stats)

        return pd.concat(df_list, ignore_index=True)

    def _calculate_stats(
        self, data: pd.Series, confidence: float = 0.95, precision: float = 0.05
    ) -> dict:

        size = len(data)
        mean = data.mean()
        std = data.std()
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - (alpha / 2), size - 1)
        error = t_crit * (std / np.sqrt(size))
        error_p = 0 if mean == 0 else error / mean
        ci_low = mean - error
        ci_high = mean + error

        n_size = np.square(t_crit * std / precision)

        return {
            "size": [size],
            "size_ideal": [n_size],
            "mean": [mean],
            "std": [std],
            "confidence": [confidence],
            "precision": [precision],
            "t_crit": [t_crit],
            "error": [error],
            "error_p": [error_p],
            "ci_low": [ci_low],
            "ci_high": [ci_high],
        }

        pass

    def save_stats(self, confidence, precision):

        stats_df = self.calculate_stats(confidence, precision)
        save_path = self.experiment_folder / "experiment_stats.csv"
        stats_df.to_csv(save_path, index=False)

        return stats_df
