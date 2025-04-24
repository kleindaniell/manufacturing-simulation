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
        self.resource_transport = {}
        self.resource_utilization = {}
        self.resource_breakdowns = {}

        for resource in self.resources:
            self.resource_output[resource] = simpy.FilterStore(self.env)
            self.resource_input[resource] = simpy.FilterStore(self.env)
            self.resource_processing[resource] = simpy.Store(self.env)
            self.resource_transport[resource] = simpy.Store(self.env)
            self.resource_utilization[resource] = 0
            self.resource_breakdowns[resource] = []

    def _create_products_stores(self) -> None:
        # Outbound
        self.finished_orders = {}
        self.finished_goods = {}
        # Inbound
        self.demand_orders = {}
        # KPIs
        self.delivered_ontime = {}
        self.delivered_late = {}
        self.lost_sales = {}
        self.wip = {}
        self.total_wip = simpy.Container
        for product in self.products:
            self.finished_orders[product] = simpy.FilterStore(self.env)
            self.finished_goods[product] = simpy.Container(self.env)
            self.demand_orders[product] = simpy.FilterStore(self.env)
            self.delivered_ontime[product] = simpy.Container(self.env)
            self.delivered_late[product] = simpy.Container(self.env)
            self.lost_sales[product] = simpy.Container(self.env)
            self.wip[product] = simpy.Container(self.env)


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

    def _process(self):
        self.process_total = len(self.stores.products[self.product]["processes"])

        first_process = next(iter(self.stores.products[self.product]["processes"]))
        first_resource = self.stores.products[self.product]["processes"][first_process][
            "resource"
        ]
        if self.schedule > self.env.now:
            delay = self.schedule - self.env.now
            yield self.env.timeout(delay)
        else:
            self.released = self.env.now

        # Add order to first resource input
        yield self.stores.resource_input[first_resource].put(self.to_dict())
