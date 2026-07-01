from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from scipy import stats

_LOG_READ_COLS = ["time", "value", "variable", "key"]
_LOG_READ_DTYPES = {
    "time": "float32",
    "value": "float32",
    "variable": "string",
    "key": "string",
}
_METRICS_NOT_PASSED = object()
_RUNS_METRICS_COLS = ["experiment", "variable", "key", "run", "value"]
_STANDARD_RUN_METRICS_FILES = (
    "metrics_products.csv",
    "metrics_resources.csv",
    "metrics_general.csv",
)


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
        self.custom_metrics = custom_metrics or []
        self.logs = pd.DataFrame()
        self.runs_metrics = pd.DataFrame(columns=_RUNS_METRICS_COLS)
        self._metrics: list[str] | None = None

        self.params = {}
        if config:
            self.params = config
        else:
            self.read_params()

    def read_params(self) -> None:
        params_path = self.experiment_folder / ".hydra" / "config.yaml"
        self.params = yaml.safe_load(params_path.read_text())

    def read_logs(
        self, metrics: list[str | Enum] | None = _METRICS_NOT_PASSED
    ) -> None:
        if metrics is _METRICS_NOT_PASSED:
            pass
        elif metrics is None:
            self._metrics = None
        else:
            self._metrics = self._normalize_metric_names(metrics)

        variables = set(self._metrics) if self._metrics is not None else None
        self.logs = self._read_log_files(variables)
        if not self.logs.empty:
            self.logs["experiment"] = self.experiment_folder.name

    def _normalize_metric_names(self, metrics: list[str | Enum]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for metric in metrics:
            name = metric.name if isinstance(metric, Enum) else metric
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def _metric_names(self) -> list[str]:
        if self._metrics is not None:
            return self._metrics
        names = [m.name for m in STANDARD_METRICS]
        for metric in self.custom_metrics:
            if metric.name not in names:
                names.append(metric.name)
        return names

    def _known_variable_names(self) -> set[str]:
        names = {m.name for m in STANDARD_METRICS}
        names.update(m.name for m in self.custom_metrics)
        if self._metrics is not None:
            names.update(self._metrics)
        return names

    def _parse_log_variable(self, stem: str) -> str | None:
        if not stem.startswith("log_"):
            return None
        remainder = stem[4:]
        known = self._known_variable_names()
        matches = [
            name
            for name in known
            if remainder.startswith(f"{name}_") or remainder == name
        ]
        if not matches:
            return None
        return max(matches, key=len)

    def _active_metric_enums(self) -> list[Enum]:
        lookup = {m.name: m for m in STANDARD_METRICS}
        for metric in self.custom_metrics:
            lookup[metric.name] = metric
        return [lookup[name] for name in self._metric_names() if name in lookup]

    def _read_log_files(self, variables: set[str] | None = None) -> pd.DataFrame:
        file_list = [
            path
            for path in self.experiment_folder.rglob("**/*log_*.csv")
            if path.is_file()
        ]
        if variables is not None:
            file_list = [
                path
                for path in file_list
                if self._parse_log_variable(path.stem) in variables
            ]

        df_list = []
        for file_path in file_list:
            df_tmp = pd.read_csv(
                file_path,
                usecols=lambda c: c in _LOG_READ_COLS,
                dtype=_LOG_READ_DTYPES,
            )
            if df_tmp.empty:
                continue
            run = int(file_path.parent.parent.stem.split("_")[-1])
            df_tmp["run"] = run
            df_list.append(df_tmp)

        if df_list:
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

    def read_runs_metrics(
        self,
        *,
        source: Literal["auto", "cache", "saved", "logs"] = "auto",
        include_custom: bool = False,
    ) -> pd.DataFrame:
        if source == "auto":
            cached = self._read_runs_metrics_cache()
            if cached is not None:
                self.runs_metrics = cached
                return self.runs_metrics
            saved = self._read_saved_runs_metrics(include_custom=include_custom)
            if saved is not None:
                self.runs_metrics = self._supplement_runs_metrics_from_logs(saved)
                return self.runs_metrics
            self.read_logs()
            return self.calculate_runs_stats()

        if source == "cache":
            cached = self._read_runs_metrics_cache()
            if cached is None:
                raise FileNotFoundError(
                    f"No runs_metrics.csv in {self.experiment_folder}"
                )
            self.runs_metrics = cached
            return self.runs_metrics

        if source == "saved":
            saved = self._read_saved_runs_metrics(include_custom=include_custom)
            if saved is None:
                raise FileNotFoundError(
                    f"No saved run metrics in {self.experiment_folder}"
                )
            self.runs_metrics = self._supplement_runs_metrics_from_logs(saved)
            return self.runs_metrics

        self.read_logs()
        return self.calculate_runs_stats()

    def _read_runs_metrics_cache(self) -> pd.DataFrame | None:
        cache_path = self.experiment_folder / "runs_metrics.csv"
        if not cache_path.is_file():
            return None
        df = pd.read_csv(cache_path)
        return self._filter_runs_metrics(df)

    def _read_saved_runs_metrics(self, include_custom: bool = False) -> pd.DataFrame | None:
        run_folders = sorted(
            path
            for path in self.experiment_folder.iterdir()
            if path.is_dir() and path.name.startswith("run_")
        )
        if not run_folders:
            return None

        df_list = []
        for run_folder in run_folders:
            run_metrics = self._read_saved_run_metrics(
                run_folder, include_custom=include_custom
            )
            if not run_metrics.empty:
                df_list.append(run_metrics)

        if not df_list:
            return None

        return self._filter_runs_metrics(pd.concat(df_list, ignore_index=True))

    def _read_saved_run_metrics(
        self, run_folder: Path, include_custom: bool = False
    ) -> pd.DataFrame:
        run_id = self._parse_run_id(run_folder)
        metric_files = list(_STANDARD_RUN_METRICS_FILES)

        if include_custom:
            standard = set(_STANDARD_RUN_METRICS_FILES)
            metric_files.extend(
                path.name
                for path in run_folder.glob("metrics_*.csv")
                if path.name not in standard
            )

        df_list = []
        for file_name in metric_files:
            file_path = run_folder / file_name
            if not file_path.is_file():
                continue
            df_wide = pd.read_csv(file_path)
            df_long = self._wide_metrics_csv_to_long(df_wide, run_id)
            if not df_long.empty:
                df_list.append(df_long)

        if not df_list:
            return pd.DataFrame(columns=_RUNS_METRICS_COLS)

        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def _parse_run_id(run_folder: Path) -> int:
        return int(run_folder.name.split("_")[-1])

    def _wide_metrics_csv_to_long(
        self, df_wide: pd.DataFrame, run_id: int
    ) -> pd.DataFrame:
        if df_wide.empty:
            return pd.DataFrame(columns=_RUNS_METRICS_COLS)

        df = df_wide.copy()
        key_col = "key" if "key" in df.columns else df.columns[0]
        if key_col != "key":
            df = df.rename(columns={key_col: "key"})
        id_vars = ["key"]
        value_vars = [col for col in df.columns if col != "key"]
        if not value_vars:
            return pd.DataFrame(columns=_RUNS_METRICS_COLS)

        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="variable",
            value_name="value",
        )
        df_long["experiment"] = self.experiment_folder.name
        df_long["run"] = run_id
        return df_long[_RUNS_METRICS_COLS]

    def _filter_runs_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=_RUNS_METRICS_COLS)

        allowed = set(self._metric_names())
        filtered = df.loc[df["variable"].isin(allowed), _RUNS_METRICS_COLS]
        return filtered.reset_index(drop=True)

    def _supplement_runs_metrics_from_logs(
        self, runs_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        loaded_vars = set(runs_metrics["variable"].unique())
        missing_enums = [
            metric
            for metric in self._active_metric_enums()
            if metric.name not in loaded_vars
        ]
        if not missing_enums:
            return runs_metrics

        self.read_logs(metrics=[metric.name for metric in missing_enums])
        supplements = []
        for metric in missing_enums:
            if (
                self.logs.empty
                or "variable" not in self.logs.columns
                or self.logs.loc[self.logs["variable"] == metric.name].empty
            ):
                supplements.append(self._synthetic_metric_frame(metric))
            else:
                supplements.append(self._calculate_metric(metric))

        if not supplements:
            return runs_metrics

        return pd.concat([runs_metrics, *supplements], ignore_index=True)

    def calculate_runs_stats(self) -> pd.DataFrame:
        df_list = [
            self._calculate_metric(metric) for metric in self._active_metric_enums()
        ]
        if df_list:
            self.runs_metrics = pd.concat(df_list, ignore_index=True)
        else:
            self.runs_metrics = pd.DataFrame(
                columns=["experiment", "variable", "key", "run", "value"]
            )
        return self.runs_metrics

    def _calculate_metric(self, metric: Enum) -> pd.DataFrame:

        group_keys = ["experiment", "variable", "key", "run"]
        if self.logs.empty or "variable" not in self.logs.columns:
            return self._synthetic_metric_frame(metric)

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
            metric_list = self._metric_names()

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
        if self.runs_metrics.empty:
            self.runs_metrics = self.calculate_runs_stats()

        stats_df = self.calculate_experiment_stats(confidence, precision)
        self.runs_metrics.to_csv(
            self.experiment_folder / "runs_metrics.csv", index=False
        )
        save_path = self.experiment_folder / "experiment_stats.csv"
        stats_df.to_csv(save_path, index=False)

        return stats_df
