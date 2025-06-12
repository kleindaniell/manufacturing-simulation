from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import simpy
import pandas as pd


class Stores:
    def __init__(
        self,
        env: simpy.Environment,
        resources: dict,
        products: dict,
        warmup: int = 0,
        log_interval: int = 72,
        training: bool = False,
    ):
        self.env = env
        self.resources = resources
        self.products = products
        self.warmup = warmup
        self.log_interval = log_interval
        self.training = training

        self._create_process_data()
        self._create_resources_stores()
        self._create_products_stores()

        self.metrics_perf = GeneralMetrics(self.products.keys())
        self.metrics_prod = ProductMetrics(self.products.keys())
        self.metrics_res = ResourceMetrics(self.resources.keys())

        self.total_wip_log: List[Tuple[float, float]] = []

        if not training:
            self._register_log()

    def _create_process_data(self) -> None:
        self.processes_name_list = {}
        self.processes_value_list = {}
        self.processes = {}

        for product in self.products:
            processes = self.products[product].get("processes")
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

                    self.metrics_prod.fg_log[product].append(
                        (self.env.now, self.finished_goods[product].level)
                    )

                    wip = self.wip[product].level
                    self.metrics_prod.wip_log[product].append((self.env.now, wip))
                    total_wip += wip

                self.total_wip_log.append((self.env.now, total_wip))

                yield self.env.timeout(self.log_interval)

        self.env.process(register_product_log())


@dataclass
class GeneralMetrics:
    products: List[str]
    delivered_ontime: Dict[str, int] = field(default_factory=dict)
    delivered_late: Dict[str, int] = field(default_factory=dict)
    lost_sales: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for product in self.products:
            self.delivered_ontime[product] = 0
            self.delivered_late[product] = 0
            self.lost_sales[product] = 0


@dataclass
class ProductMetrics:
    products: List[str]
    flow_time: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    lead_time: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    tardiness: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    earliness: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    wip_log: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    fg_log: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self):
        for product in self.products:
            self.flow_time[product] = []
            self.lead_time[product] = []
            self.tardiness[product] = []
            self.earliness[product] = []
            self.wip_log[product] = []
            self.fg_log[product] = []


@dataclass
class ResourceMetrics:
    resources: List[str]
    resource_utilization: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict
    )
    resource_breakdowns: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict
    )
    resource_setup: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self):
        for resource in self.resources:
            self.resource_utilization[resource] = []
            self.resource_breakdowns[resource] = []
            self.resource_setup[resource] = []


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
