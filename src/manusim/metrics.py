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


@dataclass(eq=False)
class MetricParams:
    agg: str
    timebase: bool
    missing_fill: Literal["zero", "nan"] = "zero"


class MetricProducts(Enum):
    deliveredOntime = MetricParams(AggMethod.sum, False, "zero")
    deliveredLate = MetricParams(AggMethod.sum, False, "zero")
    lostSales = MetricParams(AggMethod.sum, False, "zero")
    flowTime = MetricParams(AggMethod.mean, False, "nan")
    leadTime = MetricParams(AggMethod.mean, False, "nan")
    tardiness = MetricParams(AggMethod.mean, False, "nan")
    earliness = MetricParams(AggMethod.mean, False, "nan")
    wip = MetricParams(AggMethod.mean, False, "nan")
    finishedGoods = MetricParams(AggMethod.mean, False, "nan")
    released = MetricParams(AggMethod.mean, False, "nan")


class MetricResources(Enum):
    utilization = MetricParams(AggMethod.sum, True, "zero")
    breakdown = MetricParams(AggMethod.sum, True, "zero")
    setup = MetricParams(AggMethod.sum, True, "zero")
    queue = MetricParams(AggMethod.mean, False, "nan")


class MetricGeneral(Enum):
    wip_general = MetricParams(AggMethod.mean, False, "nan")
    finishedGoods_general = MetricParams(AggMethod.mean, False, "nan")


STANDARD_METRICS: tuple[Enum, ...] = (
    *MetricProducts,
    *MetricResources,
    *MetricGeneral,
)


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
        df_list = []

        if len(file_list) > 0:
            for file_path in file_list:
                if file_path.is_file():
                    df_tmp = pd.read_csv(file_path, low_memory=False)
                    if not df_tmp.empty:
                        run = int(file_path.parent.parent.stem.split("_")[-1])
                        df_tmp["run"] = run
                        df_list.append(df_tmp)

        if len(df_list) > 0:
            return pd.concat(df_list, ignore_index=True)
        return pd.DataFrame()

    def _metric_family(self, metric: Enum):
        if metric in MetricGeneral:
            return MetricGeneral
        if metric in MetricResources:
            return MetricResources
        return MetricProducts

    def _config_section_keys(self, section: str) -> list[str]:
        if not self.params:
            return []
        block = self.params.get(section) if hasattr(self.params, "get") else None
        if block is None:
            return []
        if hasattr(block, "keys"):
            return list(block.keys())
        return []

    def _number_of_runs(self) -> int | None:
        if not self.params:
            return None
        experiment = self.params.get("experiment")
        if experiment is None:
            return None
        n = (
            experiment.get("number_of_runs")
            if hasattr(experiment, "get")
            else None
        )
        return int(n) if n else None

    def _runs_grid(self) -> pd.DataFrame:
        cols = ["experiment", "run"]
        if not self.logs.empty and "run" in self.logs.columns:
            runs = self.logs[cols].drop_duplicates()
            if "experiment" not in runs.columns:
                runs = runs.copy()
                runs["experiment"] = self.experiment_folder.name
            return runs

        n_runs = self._number_of_runs()
        if n_runs:
            return pd.DataFrame(
                {
                    "experiment": [self.experiment_folder.name] * n_runs,
                    "run": list(range(1, n_runs + 1)),
                }
            )
        return pd.DataFrame(columns=cols)

    def _keys_for_metric(self, metric: Enum) -> list[str]:
        if metric in MetricGeneral:
            return ["general"]

        family = self._metric_family(metric)
        family_names = {m.name for m in family}
        if not self.logs.empty and "variable" in self.logs.columns:
            keys = (
                self.logs.loc[self.logs["variable"].isin(family_names), "key"]
                .dropna()
                .unique()
            )
            if len(keys):
                return keys.tolist()

        section = "products" if metric in MetricProducts else "resources"
        return self._config_section_keys(section)

    def _missing_fill_value(self, metric: Enum) -> float:
        if metric.value.missing_fill == "nan":
            return np.nan
        return 0.0

    def _synthetic_metric_frame(self, metric: Enum) -> pd.DataFrame:
        group_keys = ["experiment", "variable", "key", "run"]
        runs = self._runs_grid()
        keys = self._keys_for_metric(metric)
        if runs.empty or not keys:
            return pd.DataFrame(columns=[*group_keys, "value"])

        grid = runs.merge(pd.DataFrame({"key": keys}), how="cross")
        grid["variable"] = metric.name
        grid["value"] = self._missing_fill_value(metric)
        return grid[[*group_keys, "value"]]

    def calculate_runs_stats(self, custom_metrics: Enum = None) -> pd.DataFrame:

        df_list = [self._calculate_metric(metric) for metric in STANDARD_METRICS]

        if custom_metrics:
            for metric in custom_metrics:
                df_list.append(self._calculate_metric(metric))

        self.runs_metrics = pd.concat(df_list, ignore_index=True)

        return self.runs_metrics

    def _calculate_metric(self, metric: Enum) -> pd.DataFrame:

        group_keys = ["experiment", "variable", "key", "run"]
        df_metric = self.logs.loc[self.logs["variable"] == metric.name]

        if df_metric.empty:
            return self._synthetic_metric_frame(metric)

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
            metric_list = [m.name for m in STANDARD_METRICS]

        for metric in metric_list:
            metric_df = self.runs_metrics.loc[self.runs_metrics["variable"] == metric]

            if aggregate_variables:
                if not metric_df.empty:
                    metric_df = (
                        metric_df.drop("key", axis=1)
                        .groupby(["experiment", "variable", "run"], dropna=False)
                        .mean(numeric_only=True)
                        .reset_index()
                    )
                    metric_stats = metric_df.loc[[0], ["experiment", "variable"]]
                    values = metric_df["value"]
                else:
                    metric_stats = pd.DataFrame(
                        {
                            "experiment": [self.experiment_folder.name],
                            "variable": [metric],
                        }
                    )
                    values = pd.Series(dtype=float)

                result = self._calculate_stats(values, confidence, precision)
                metric_stats = pd.concat([metric_stats, pd.DataFrame(result)], axis=1)

                df_list.append(metric_stats)

            else:
                for key in metric_df["key"].unique():
                    key_df = (
                        metric_df.loc[metric_df["key"] == key]
                        .groupby(["experiment", "variable", "run", "key"], dropna=False)
                        .mean(numeric_only=True)
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

        valid = data.dropna()
        size = len(valid)

        if size == 0:
            return {
                "size": [0],
                "size_ideal": [np.nan],
                "mean": [np.nan],
                "std": [np.nan],
                "confidence": [confidence],
                "precision": [precision],
                "t_crit": [np.nan],
                "error": [np.nan],
                "error_p": [np.nan],
                "ci_low": [np.nan],
                "ci_high": [np.nan],
            }

        mean = valid.mean()
        std = valid.std()
        alpha = 1 - confidence

        if size < 2:
            t_crit = np.nan
            error = np.nan
            error_p = np.nan
            ci_low = np.nan
            ci_high = np.nan
            n_size = np.nan
        else:
            t_crit = stats.t.ppf(1 - (alpha / 2), size - 1)
            error = t_crit * (std / np.sqrt(size))
            if mean == 0 or np.isnan(mean):
                error_p = 0.0 if mean == 0 else np.nan
            else:
                error_p = error / mean
            ci_low = mean - error
            ci_high = mean + error

            if mean != 0 and not np.isnan(mean):
                n_size = np.square(t_crit * std / (precision * mean))
            else:
                n_size = 0.0

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

    def save_stats(self, confidence, precision):

        self.runs_metrics = self.calculate_runs_stats()
        stats_df = self.calculate_experiment_stats(confidence, precision)
        save_path = self.experiment_folder / "experiment_stats.csv"
        stats_df.to_csv(save_path, index=False)

        return stats_df
