from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import simpy

from rlsim.engine.orders import ProductionOrder, DemandOrder


class SimulationStores:
    def __init__(self, env: simpy.Environment, products: dict, resources: dict):
        self._env = env
        self.products: Dict[str, Any] = products
        self.resources: Dict[str, Any] = resources

        self._create_process_data()
        self._create_products_stores()
        self._create_resources_stores()

        # self.total_wip_log = List[Tuple[float, float]] = []

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
            self.resource_output[resource] = simpy.FilterStore(self._env)
            self.resource_input[resource] = simpy.FilterStore(self._env)
            self.resource_processing[resource] = simpy.Store(self._env)
            self.resource_transport[resource] = simpy.Store(self._env)
            self.resource_finished[resource] = simpy.Store(self._env)

    def _create_products_stores(self) -> None:
        # Outbound Stores
        self.finished_goods: Dict[str, simpy.Container] = {}

        # Demand Orders stores
        self.inbound_demand_orders = simpy.FilterStore(self._env)
        self.outbound_demand_orders: Dict[str, simpy.Store] = {}
        self.wip: Dict[str, simpy.Container] = {}

        for product in self.products:
            self.finished_goods[product] = simpy.Container(self._env)
            self.outbound_demand_orders[product] = simpy.Store(self._env)
            self.wip[product] = simpy.Container(self._env)

    def simulation_snapshot(self) -> pd.DataFrame:
        resources_list = self.resources.keys()
        products_list = self.products.keys()

        df_status = pd.DataFrame(
            np.zeros((len(products_list), len(resources_list))),
            index=products_list,
            columns=resources_list,
        )

        for resource in resources_list:
            orders_queue: List[ProductionOrder] = self.resource_input[resource].items
            orders_queue.extend(self.resource_output[resource].items)
            orders_queue.extend(self.resource_transport[resource].items)
            for order in orders_queue:
                product = order.product
                df_status.loc[product, resource] += order.quantity

        df_status["wip_total"] = df_status.sum(axis=1)

        df_status["fg"] = 0
        for product in products_list:
            df_status.loc[product, "fg"] = self.finished_goods[product].level
        df_status.loc["total", :] = df_status.sum(axis=0)

        return df_status
