from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import pandas as pd
import simpy


class Stores:
    def __init__(
        self,
        env: simpy.Environment,
        resources: dict,
        products: dict,
        warmup: int = 0,
        log_interval: int = 72,
        training: bool = False,
        seed: int = None,
    ):
        self.env = env
        self.resources: Dict[str, dict] = resources
        self.products: Dict[str, dict] = products
        self.warmup = warmup
        self.log_interval = log_interval
        self.training = training
        self.seed = seed

        self._create_process_data()
        self._create_resources_stores()
        self._create_products_stores()

        self.log_products = ProductMetrics(self.products.keys())
        self.log_resources = ResourceMetrics(self.resources.keys())

        self.total_wip_log: List[Tuple[float, float]] = []

        if not training:
            self._register_log()

    def _create_process_data(self) -> None:
        self.processes_name_list = {}
        self.processes_value_list = {}
        self.processes = {}

        for product in self.products:
            processes: dict = self.products[product].get("processes")
            self.processes_name_list[product] = list(processes.keys())
            self.processes_value_list[product] = list(processes.values())

    def _create_resources_stores(self) -> None:
        self.resource_output: Dict[str, simpy.FilterStore] = {}
        self.resource_input: Dict[str, simpy.FilterStore] = {}
        self.resource_processing: Dict[str, simpy.Store] = {}
        self.resource_finished: Dict[str, simpy.Store] = {}
        self.resource_transport: Dict[str, simpy.Store] = {}

        for resource in self.resources:
            self.resource_output[resource] = simpy.FilterStore(self.env)
            self.resource_input[resource] = simpy.FilterStore(self.env)
            self.resource_processing[resource] = simpy.Store(self.env)
            self.resource_transport[resource] = simpy.Store(self.env)
            self.resource_finished[resource] = simpy.Store(self.env)

    def _create_products_stores(self) -> None:

        # Outbound Stores
        self.finished_goods: Dict[str, simpy.Container] = {}

        # Demand Orders stores
        self.inbound_demand_orders = simpy.FilterStore(self.env)
        self.outbound_demand_orders: Dict[str, simpy.Store] = {}
        self.wip: Dict[str, simpy.Container] = {}

        for product in self.products:
            self.finished_goods[product] = simpy.Container(self.env)
            self.outbound_demand_orders[product] = simpy.Store(self.env)
            self.wip[product] = simpy.Container(self.env)

    def _register_log(self) -> None:
        def register_product_log():
            yield self.env.timeout(self.warmup)
            while True:
                total_wip = 0
                for product in self.products.keys():

                    self.log_products.fg_log[product].append(
                        (self.env.now, self.finished_goods[product].level)
                    )

                    wip = self.wip[product].level
                    self.log_products.wip_log[product].append((self.env.now, wip))
                    total_wip += wip

                self.total_wip_log.append((self.env.now, total_wip))

                yield self.env.timeout(self.log_interval)

        self.env.process(register_product_log())


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


@dataclass
class ProductionOrder:
    product: str
    quantity: int
    schedule: Optional[float] = None
    released: Optional[int] = None
    duedate: Optional[float] = None
    finished: Optional[bool] = None
    priority: Optional[int] = None
    process_total: Optional[int] = None
    process_finished: Optional[int] = None
    id: int = field(init=False)

    _next_id = 1

    def __post_init__(self):
        self.id = ProductionOrder._next_id
        ProductionOrder._next_id += 1

    def to_dict(self) -> dict:
        keys = [
            "product",
            "quantity",
            "schedule",
            "released",
            "duedate",
            "finished",
            "priority",
            "process_total",
            "process_finished",
            "id",
        ]

        dict_tmp = {key: self.__dict__[key] for key in keys if key in self.__dict__}
        return dict_tmp


@dataclass
class DemandOrder:
    product: str
    quantity: int
    duedate: Optional[float] = None
    arived: Optional[float] = None
    delivered: Optional[int] = None
    id: int = field(init=False)

    _next_id = 1

    def __post_init__(self):
        self.id = DemandOrder._next_id
        DemandOrder._next_id += 1

    def to_dict(self) -> dict:
        keys = [
            "product",
            "quantity",
            "duedate",
            "arived",
            "delivered",
            "id",
        ]

        dict_tmp = {key: self.__dict__[key] for key in keys if key in self.__dict__}
        return dict_tmp
