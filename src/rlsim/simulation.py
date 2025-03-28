from production import Production
from control import Stores
from scheduler import Scheduler
from monitor import Monitor


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
        self.production = Production(self.stores, warmup=0)
        self.scheduler = Scheduler(self.stores, self.schedule_interval)

    def run_simulation(self):
        print(self.run_until)
        self.env.run(until=self.run_until)


if __name__ == "__main__":
    resource_path = Path("config/resources.yaml")
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("config/products.yaml")
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

    run_until = 1000
    schedule_interval = 72
    monitor_interval = 36

    sim = Simulation(
        run_until=run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        schedule_interval=schedule_interval,
        monitor_interval=monitor_interval,
    )

    sim.run_simulation()
