from dataclasses import dataclass, field, fields
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ProductMetrics:
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

    def __init__(self, products):
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

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=["time", "value", "variable", "product"],
        )
        for fi in fields(self):
            values: dict = getattr(self, fi.name)

            for product in values:
                if len(values[product]) > 0:
                    df_tmp = pd.DataFrame(values[product], columns=["time", "value"])
                    df_tmp["variable"] = fi.name
                    df_tmp["product"] = product
                    df = df_tmp.copy() if df.empty else pd.concat([df, df_tmp])
        return df.reset_index(drop=True)


@dataclass
class ResourceMetrics:
    utilization: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    breakdowns: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    setups: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def __init__(self, resources: List[str]):
        self.utilization = {r: [] for r in resources}
        self.breakdowns = {r: [] for r in resources}
        self.setups = {r: [] for r in resources}

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=["time", "value", "variable", "resource"],
        )
        for fi in fields(self):
            values: dict = getattr(self, fi.name)

            for resource in values:
                if len(values[resource]) > 0:
                    df_tmp = pd.DataFrame(values[resource], columns=["time", "value"])
                    df_tmp["variable"] = fi.name
                    df_tmp["resource"] = resource
                    df = df_tmp.copy() if df.empty else pd.concat([df, df_tmp])
        return df.reset_index(drop=True)
