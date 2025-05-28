from rlsim.production import Production
from rlsim.control import Stores, ProductionOrder
from rlsim.monitor import Monitor
from rlsim.inbound import Inbound
from rlsim.outbound import Outbound
from rlsim.simple_scheduler import SimpleScheduler

from typing import List

import simpy
import random
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from time import sleep


class Simulation:
    def __init__(
        self,
        run_until: int,
        resources_cfg: dict,
        products_cfg: dict,
        schedule_interval: int,
        set_constraint: int = None,
        monitor_interval: int = 0,
        warmup: bool = False,
        seed: int = None,
    ):
        super().__init__()
        random.seed(seed)
        self.env = simpy.Environment()

        # Parameters
        self.resources_config = resources_cfg
        self.products_config = products_cfg
        self.warmup = warmup
        self.run_until = run_until
        self.monitor_interval = monitor_interval
        self.schedule_interval = schedule_interval
        self.constraint = set_constraint

        self.stores = Stores(self.env, self.resources_config, self.products_config)
        self.monitor = Monitor(self.stores, self.monitor_interval)
        callback = self.order_selection_callback(self.make_shipping_buffer_test())
        self.production = Production(self.stores, warmup=0, order_selection_fn=callback)

        self.scheduler = SimpleScheduler(self.stores, self.schedule_interval)
        self.inboud = Inbound(self.stores, self.products_config)
        self.outbound = Outbound(
            self.stores, self.products_config, delivery_mode="asReady"
        )

    def run_simulation(self):
        print(self.run_until)
        self.env.run(until=self.run_until)

    def make_shipping_buffer_test(self):
        return {
            x: self.products_config[x]["shipping_buffer"]
            for x in self.products_config.keys()
        }

    def order_selection_callback(self, buffer: dict):
        def order_selection(store: Stores, resource):
            # TODO - make order selector for test
            orders: List[ProductionOrder] = store.resource_input[resource].items
            priorities = [
                [order.id, order.released * buffer[order.product]] for order in orders
            ]
            order_id, _ = sorted(priorities, key=lambda x: x[1])[0]
            return order_id

        return order_selection


if __name__ == "__main__":
    resource_path = Path("config/resources.yaml")
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("config/products.yaml")
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

    run_until = 1000
    schedule_interval = 10
    monitor_interval = 10

    sim = Simulation(
        run_until=run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        schedule_interval=schedule_interval,
        monitor_interval=monitor_interval,
    )

    sim.run_simulation()
