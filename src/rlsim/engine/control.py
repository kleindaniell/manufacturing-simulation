from dataclasses import dataclass, field
from typing import Optional

import simpy


class Stores:
    def __init__(
        self,
        env: simpy.Environment,
        resources: dict,
        products: dict,
    ):
        self.env = env
        self.resources = resources
        self.products = products

        self._create_process_data()
        self._create_resources_stores()
        self._create_products_stores()

    def _create_process_data(self) -> None:
        self.processes_name_list = {}
        self.processes_value_list = {}
        self.processes = {}

        for product in self.products:
            processes = self.products[product].get("processes")
            self.processes_name_list[product] = list(processes.keys())
            self.processes_value_list[product] = list(processes.values())

    def _create_resources_stores(self) -> None:
        self.resource_output = {}
        self.resource_input = {}
        self.resource_processing = {}
        self.resource_finished = {}
        self.resource_transport = {}
        self.resource_utilization = {}
        self.resource_breakdowns = {}
        self.resource_setup = {}

        for resource in self.resources:
            self.resource_output[resource] = simpy.FilterStore(self.env)
            self.resource_input[resource] = simpy.FilterStore(self.env)
            self.resource_processing[resource] = simpy.Store(self.env)
            self.resource_transport[resource] = simpy.Store(self.env)
            self.resource_finished[resource] = simpy.Store(self.env)
            self.resource_utilization[resource] = 0
            self.resource_breakdowns[resource] = []
            self.resource_setup[resource] = []

    def _create_products_stores(self) -> None:
        # Outbound Stores
        self.finished_goods = {}
        # Demand Orders stores
        self.inbound_demand_orders = simpy.FilterStore(self.env)
        self.outbound_demand_orders = {}
        # KPIs
        self.delivered_ontime = {}
        self.delivered_late = {}
        self.lost_sales = {}
        for product in self.products:
            self.finished_goods[product] = simpy.Container(self.env)
            self.outbound_demand_orders[product] = simpy.FilterStore(self.env)
            self.delivered_ontime[product] = simpy.Container(self.env)
            self.delivered_late[product] = simpy.Container(self.env)
            self.lost_sales[product] = simpy.Container(self.env)


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
