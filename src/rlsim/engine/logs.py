from dataclasses import dataclass, field, fields
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ProductLogs:
    # Products Logs
    delivered_ontime: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    delivered_late: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    lost_sales: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    flow_time: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    lead_time: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    tardiness: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    earliness: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    wip_log: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    fg_log: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    released: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def __init__(self, products, **kwargs):
        # Product Logs
        self.delivered_ontime = {p: [] for p in products}
        self.delivered_late = {p: [] for p in products}
        self.lost_sales = {p: [] for p in products}
        self.flow_time = {p: [] for p in products}
        self.lead_time = {p: [] for p in products}
        self.tardiness = {p: [] for p in products}
        self.earliness = {p: [] for p in products}
        self.wip_log = {p: [] for p in products}
        self.fg_log = {p: [] for p in products}
        self.released = {p: [] for p in products}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert logs to dataframes"""

        logs_df = pd.DataFrame(
            columns=["time", "value", "variable", "product"],
        )

        for attr in self.__dict__.keys():
            values: dict = getattr(self, attr)

            for product in values:
                if len(values[product]) > 0:
                    df_tmp = pd.DataFrame(values[product], columns=["time", "value"])
                    df_tmp["variable"] = attr
                    df_tmp["product"] = product
                    logs_df = (
                        df_tmp.copy() if logs_df.empty else pd.concat([logs_df, df_tmp])
                    )

        return logs_df

    def calculate_metrics(self) -> pd.DataFrame:
        logs = self.to_dataframe()
        df_result = pd.DataFrame()
        # Sum
        for var in ["delivered_ontime", "delivered_late", "lost_sales"]:
            df_tmp = logs.loc[logs["variable"] == var]
            df_var = df_tmp.pivot_table("value", "product", "variable", "sum")
            df_var.columns.name = None
            df_var.loc["mean", :] = df_var.mean()
            df_var.loc["total", :] = df_var.sum()
            df_var = df_var.fillna(value=0)
            df_var = np.round(df_var, 0).astype(np.int32)
            df_result = (
                df_var.copy()
                if df_result.empty
                else pd.concat([df_result, df_var], axis=1)
            )
        # Mean
        for var in ["flow_time", "lead_time", "tardiness", "earliness"]:
            df_tmp = logs.loc[logs["variable"] == var]
            df_var = df_tmp.pivot_table("value", "product", "variable", "mean")
            df_var.columns.name = None
            df_var.loc["mean", :] = df_var.mean()
            df_var.loc["total", :] = df_tmp["value"].mean()
            df_var = df_var.fillna(value=0)
            df_var = np.round(df_var, 3).astype(np.float32)
            df_result = (
                df_var.copy()
                if df_result.empty
                else pd.concat([df_result, df_var], axis=1)
            )

        # Mean agg
        for var in ["wip_log", "fg_log", "released"]:
            df_tmp = logs.loc[logs["variable"] == var]
            df_var = df_tmp.pivot_table("value", "product", "variable", "mean")
            df_var.columns.name = None
            df_var.loc["mean", :] = df_var.mean()
            df_var.loc["total", :] = (
                df_tmp[["time", "value"]].groupby("time").sum()["value"].mean()
            )
            df_var = df_var.fillna(value=0)
            df_result = (
                df_var.copy()
                if df_result.empty
                else pd.concat([df_result, df_var], axis=1)
            )

        return df_result.fillna(0)


class ResourceLogs:
    utilization: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    breakdowns: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    setups: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def __init__(self, resources: List[str], **kwagrs):
        self.utilization = {r: [] for r in resources}
        self.breakdowns = {r: [] for r in resources}
        self.setups = {r: [] for r in resources}

        for key, value in kwagrs.items():
            setattr(self, key, value)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=["time", "value", "variable", "resource"],
        )
        for attr in self.__dict__.keys():
            values: dict = getattr(self, attr)

            for resource in values:
                if len(values[resource]) > 0:
                    df_tmp = pd.DataFrame(values[resource], columns=["time", "value"])
                    df_tmp["variable"] = attr
                    df_tmp["resource"] = resource
                    df = df_tmp.copy() if df.empty else pd.concat([df, df_tmp])
        return df.reset_index(drop=True)

    def calculate_metrics(self) -> pd.DataFrame:
        logs = self.to_dataframe()
        df_result = pd.DataFrame()
        # Sum
        for var in ["utilization"]:
            df_tmp = logs.loc[logs["variable"] == var]
            df_var = df_tmp.pivot_table("value", "resource", "variable", "sum")
            df_var.columns.name = None
            df_var.loc["mean", :] = df_var.mean()
            df_var = df_var.fillna(value=0)
            df_var = np.round(df_var, 3).astype(np.float32)
            df_result = (
                df_var.copy()
                if df_result.empty
                else pd.concat([df_result, df_var], axis=1)
            )
        # Mean
        for var in ["breakdowns", "setups"]:
            df_tmp = logs.loc[logs["variable"] == var]
            if not df_tmp.empty:
                df_var = df_tmp.pivot_table("value", "resource", "variable", "mean")
                df_var.columns.name = None
                df_var.columns = [f"{var}_mean"]
                df_var.loc["mean", :] = df_var.mean()
                df_var = df_var.fillna(value=0)
                df_var = np.round(df_var, 3).astype(np.float32)
                df_result = (
                    df_var.copy()
                    if df_result.empty
                    else pd.concat([df_result, df_var], axis=1)
                )
        # Count
        for var in ["breakdowns", "setups"]:
            df_tmp = logs.loc[logs["variable"] == var]
            if not df_tmp.empty:
                df_var = df_tmp.pivot_table("value", "resource", "variable", "count")
                df_var.columns.name = None
                df_var.columns = [f"{var}_count"]
                df_var.loc["mean", :] = df_var.mean()
                df_var = df_var.fillna(value=0)
                df_result = (
                    df_var.copy()
                    if df_result.empty
                    else pd.concat([df_result, df_var], axis=1)
                )
        return df_result.fillna(0)


class GeneralLogs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=["time", "value", "variable"],
        )
        for attr in self.__dict__.keys():
            values: dict = getattr(self, attr)

            if len(values) > 0:
                df_tmp = pd.DataFrame(values, columns=["time", "value"])
                df_tmp["variable"] = attr
                df = df_tmp.copy() if df.empty else pd.concat([df, df_tmp])
        return df.reset_index(drop=True)
