from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from scipy import stats


class AggMethod:
    sum: str = "sum"
    mean: str = "mean"
    count: str = "count"
    last: str = "last"


@dataclass
class MetricParams:
    agg: str
    timebase: bool


class MetricProducts(Enum):
    deliveredOntime = MetricParams(AggMethod.sum, False)
    deliveredLate = MetricParams(AggMethod.sum, False)
    lostSales = MetricParams(AggMethod.sum, False)
    flowTime = MetricParams(AggMethod.mean, False)
    leadTime = MetricParams(AggMethod.mean, False)
    tardiness = MetricParams(AggMethod.mean, False)
    earliness = MetricParams(AggMethod.mean, False)
    wip = MetricParams(AggMethod.mean, False)
    finishedGoods = MetricParams(AggMethod.mean, False)
    released = MetricParams(AggMethod.mean, False)


class MetricResources(Enum):
    utilization = MetricParams(AggMethod.sum, True)
    breakdown = MetricParams(AggMethod.sum, True)
    setup = MetricParams(AggMethod.sum, True)
    queue = MetricParams(AggMethod.mean, False)


@dataclass
class ExperimentMetrics:
    def __init__(
        self, experiment_folder: Path, config: DictConfig = None, custom_metrics: list = None
    ):
        self.experiment_folder = experiment_folder
        if not isinstance(self.experiment_folder, Path):
            self.experiment_folder = Path(self.experiment_folder).resolve()
        self.custom_metrics = custom_metrics
        self.logs = pd.DataFrame()
        
        self.params = {}
        if config:
            self.params = config
        else:
            self.read_params()

    def read_params(self) -> None:
        params_path = self.experiment_folder / ".hydra" / "config.yaml"
        self.params = yaml.safe_load(params_path.read_text())

    def read_logs(self) -> None:

        glob_pattern = f"**/*log_*.csv"
        self.logs = self._read_log_files(glob_pattern)
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

        if len(df_list) > 0:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_runs_stats(self, custom_metrics: Enum = None) -> pd.DataFrame:

        df_list = []
        for metric in MetricProducts:
            df_list.append(self._calculate_metric(metric))

        for metric in MetricResources:
            df_list.append(self._calculate_metric(metric))

        if custom_metrics:
            for metric in custom_metrics:
                df_list.append(self._calculate_metric(metric))

        self.runs_metrics = pd.concat(df_list, ignore_index=True)

        return self.runs_metrics

    def _calculate_metric(self, metric: Enum) -> pd.DataFrame:

        group_keys = ["experiment", "variable", "key", "run"]
        df_metric = self.logs.loc[self.logs["variable"] == metric.name]

        if df_metric.empty:
            return pd.DataFrame()

        if metric.value.agg == AggMethod.mean:
            df_metric = (
                df_metric.drop("time", axis=1).groupby(group_keys).mean().reset_index()
            )
        elif metric.value.agg == AggMethod.sum:
            df_metric = (
                df_metric.drop("time", axis=1).groupby(group_keys).sum().reset_index()
            )
        elif metric.value.agg == AggMethod.count:
            df_metric = (
                df_metric.drop("time", axis=1).groupby(group_keys).count().reset_index()
            )

        if metric.value.timebase:
            run_until = self.params["simulation"]["run_until"]
            warmup = self.params["simulation"]["warmup"]
            df_metric["value"] = df_metric["value"] / (run_until - warmup)

        return df_metric

    def calculate_experiment_stats(
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
            metric_list = self.runs_metrics["variable"].unique()
            print(metric_list)

        for metric in metric_list:
            metric_df = self.runs_metrics.loc[self.runs_metrics["variable"] == metric]
            print(metric_df)
            if aggregate_variables:
                metric_df = (
                    metric_df.drop("key", axis=1)
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

        if mean != 0:
            n_size = np.square(t_crit * std / (precision * mean))
        else:
            n_size = 0

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

        self.runs_metrics = self.calculate_runs_stats()
        stats_df = self.calculate_experiment_stats(confidence, precision)
        save_path = self.experiment_folder / "experiment_stats.csv"
        stats_df.to_csv(save_path, index=False)

        return stats_df
