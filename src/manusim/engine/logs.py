from typing import List

import numpy as np
import pandas as pd


class BaseLogs:
    """Base class for event logging"""

    def _log_dict_to_df(self, field_name: str = None) -> pd.DataFrame:
        df_list = []
        for attr, values in self.__dict__.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    if val:
                        df_tmp = pd.DataFrame(val, columns=["time", "value"])
                        df_tmp["variable"] = attr
                        if field_name:
                            df_tmp[field_name] = key
                        df_list.append(df_tmp)

        return (
            pd.concat(df_list, ignore_index=True)
            if df_list
            else pd.DataFrame(columns=["time", "value", "variable", field_name])
        )


class ProductLogs(BaseLogs):
    def __init__(
        self,
        products: List[str],
        **kwargs,
    ):
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
        return self._log_dict_to_df("product")

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


class ResourceLogs(BaseLogs):
    def __init__(self, resources: List[str], **kwagrs):
        self.utilization = {r: [] for r in resources}
        self.breakdowns = {r: [] for r in resources}
        self.setups = {r: [] for r in resources}
        self.queues = {r: [] for r in resources}

        for key, value in kwagrs.items():
            setattr(self, key, value)

    def to_dataframe(self) -> pd.DataFrame:
        return self._log_dict_to_df("resource")

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


class GeneralLogs(BaseLogs):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dataframe(self) -> pd.DataFrame:
        return self._log_dict_to_df()
