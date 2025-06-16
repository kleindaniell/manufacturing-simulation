from typing import Dict

import simpy


class SimulationStores:
    def __init__(self, env: simpy.Environment, products: dict, resources: dict):
        self._env = env
        self.products = products
        self.resources = resources

        self._create_process_data()
        self._create_products_stores()
        self._create_resources_stores()

    def _create_process_data(self) -> None:
        self.processes_name_list = {
            product: list(self.products[product].get("processes").keys())
            for product in self.products
        }
        self.processes_value_list = {
            product: list(self.products[product].get("processes").values())
            for product in self.products
        }

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
