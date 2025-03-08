from production import Production
from control import Info


import simpy
import random
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from time import sleep


class Simulation():
    def __init__(
        self,
        run_until: int,
        resources_cfg: dict,
        products_cfg: dict,
        schedule_interval: int,
        set_constraint: int = None,
        monitor_interval: int = None,
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


        info = Info(self.env, self.resources_config, self.products_config)
        production = Production(self.env, info, self.warmup)
    
    def run_simulation(self):
        self.env.run(run_until)


if __name__ == "__main__":
    
    resource_path = Path("../config/resources.yaml")
    with open(resource_path, 'r') as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("../config/products.yaml")
    with open(products_path, 'r') as file:
        products_cfg = yaml.safe_load(file)

    run_until = 1000
    schedule_interval = 72
    monitor_interval = 722
   
    sim = Simulation(
        run_until=run_until, 
        resources_cfg=resources_cfg, 
        products_cfg=products_cfg, 
        schedule_interval=schedule_interval,
    )

    sim.run_simulation()
    