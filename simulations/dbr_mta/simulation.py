import random
from dataclasses import asdict
from pathlib import Path
from typing import List
from time import time


import simpy
import yaml
import json
import numpy as np
import pandas as pd

from rlsim.engine.control import ProductionOrder
from rlsim.engine.inbound import Inbound
from rlsim.engine.monitor import Monitor
from rlsim.engine.outbound import Outbound
from rlsim.engine.production import Production
from rlsim.stores.dbr_mta_store import DBR_stores
from rlsim.scheduler.dbr_mta_scheduler import DBR_MTA


class Simulation:
    def __init__(
        self,
        run_until: int,
        resources_cfg: dict,
        products_cfg: dict,
        schedule_interval: int,
        set_constraint: int = None,
        monitor_interval: int = 0,
        warmup: int = 0,
        warmup_monitor: int = 0,
        log_interval: int = 72,
        seed: int = None,
    ):
        super().__init__()
        random.seed(seed)
        self.env = simpy.Environment()

        # Parameters
        self.resources_config = resources_cfg
        self.products_config = products_cfg
        self.warmup = warmup
        self.warmup_monitor = warmup_monitor
        self.run_until = run_until
        self.monitor_interval = monitor_interval
        self.log_interval = log_interval
        self.schedule_interval = schedule_interval
        self.set_constraint = set_constraint

        self.stores = DBR_stores(
            env=self.env,
            resources=self.resources_config,
            products=self.products_config,
            warmup=self.warmup,
            log_interval=self.log_interval,
        )
        if self.set_constraint:
            self.stores.contraint_resource = self.set_constraint

        self.monitor = Monitor(
            self.stores,
            self.monitor_interval,
            warmup=self.warmup_monitor,
            show_print=True,
        )
        callback = self.order_selection_callback()
        self.production = Production(self.stores, order_selection_fn=callback)

        self.scheduler = DBR_MTA(
            self.stores,
            self.schedule_interval,
            constraint_buffer_size=2000,
        )
        self.inboud = Inbound(self.stores, self.products_config)
        self.outbound = Outbound(
            self.stores, self.products_config, delivery_mode="instantly"
        )

    def run_simulation(self):

        self.env.run(until=self.run_until)

    def order_selection_callback(self):
        def order_selection(store: DBR_stores, resource):
            orders: List[ProductionOrder] = store.resource_input[resource].items
            # print(f"\n {self.env.now}")
            # print(f"{resource}")
            for id, productionOrder in enumerate(orders):
                ahead_orders: List[ProductionOrder] = []
                for resource_ in self.stores.resources.keys():
                    ahead_orders.extend(self.stores.resource_input[resource_].items)
                    ahead_orders.extend(self.stores.resource_output[resource_].items)
                    ahead_orders.extend(self.stores.resource_transport[resource_].items)
                    ahead_orders.extend(
                        self.stores.resource_processing[resource_].items
                    )

                product = productionOrder.product
                released = productionOrder.released
                ahead_quantity = [
                    order.quantity
                    for order in ahead_orders
                    if order.released < released and order.product == product
                ]
                orders[id].priority = (
                    sum(ahead_quantity) + self.stores.finished_goods[product].level
                ) / self.stores.shipping_buffer[product]

                # print(f"{productionOrder.product}: {productionOrder.priority}")

            order = sorted(orders, key=lambda x: x.priority)[0]
            # print(f"SELECTED: {order.product}: {order.priority}")
            return order.id

        return order_selection


if __name__ == "__main__":
    resource_path = Path("simulations/dbr_mta/config/resources.yaml")
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("simulations/dbr_mta/config/products.yaml")
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

    run_until = 200001
    schedule_interval = 48
    monitor_interval = 50000
    warmup = 100000
    warmup_monitor = 0

    start_time = time()

    sim = Simulation(
        run_until=run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        schedule_interval=schedule_interval,
        monitor_interval=monitor_interval,
        warmup=warmup,
        warmup_monitor=warmup_monitor,
    )

    sim.run_simulation()

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    df_products = sim.stores.log_products.to_dataframe()
    df_products.to_csv(Path("simulations/dbr_mta/data/products.csv"), index=False)
    df_resources = sim.stores.log_resources.to_dataframe()
    df_resources.to_csv(Path("simulations/dbr_mta/data/resources.csv"), index=False)
