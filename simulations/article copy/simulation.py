import random
from pathlib import Path
from typing import List

import simpy
import yaml

from rlsim.engine.control import ProductionOrder
from rlsim.engine.inbound import Inbound
from rlsim.engine.monitor import Monitor
from rlsim.engine.outbound import Outbound
from rlsim.engine.production import Production
from rlsim.stores.article_stores import DBR_stores
from rlsim.scheduler.article_scheduler import ArticleScheduler


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
            cb_start=30.5,
        )
        if self.set_constraint:
            self.stores.contraint_resource = self.set_constraint

        self.monitor = Monitor(
            self.stores,
            self.monitor_interval,
            warmup=self.warmup_monitor,
            show_print=True,
        )
        # callback = self.order_selection_callback()
        self.production = Production(self.stores)

        self.scheduler = ArticleScheduler(
            self.stores, constraint_buffer_size=30.5, shipping_buffer_size=30.5
        )
        self.inboud = Inbound(self.stores, self.products_config)
        self.outbound = Outbound(
            self.stores, self.products_config, delivery_mode="asReady"
        )

    def run_simulation(self):
        print(self.run_until)
        self.env.run(until=self.run_until)

    def order_selection_callback(self):
        def order_selection(store: DBR_stores, resource):
            orders: List[ProductionOrder] = store.resource_input[resource].items
            order = sorted(orders, key=lambda x: x.duedate)[0]
            return order.id

        return order_selection


if __name__ == "__main__":
    resource_path = Path("simulations/article/config/resources.yaml")
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("simulations/article/config/products_original.yaml")
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

    run_until = 200001
    schedule_interval = 72
    monitor_interval = 50000
    warmup = 100000
    warmup_monitor = 0

    sim = Simulation(
        run_until=run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        schedule_interval=schedule_interval,
        monitor_interval=monitor_interval,
        warmup=warmup,
    )

    sim.run_simulation()
