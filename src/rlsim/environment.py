from pathlib import Path
import random
from typing import Type, Union
import simpy
import yaml

from rlsim.engine.control import Stores
from rlsim.engine.inbound import Inbound
from rlsim.engine.monitor import Monitor
from rlsim.engine.outbound import Outbound
from rlsim.engine.production import Production
from rlsim.engine.scheduler import Scheduler


class Environment:
    def __init__(
        self,
        run_until: int,
        resources_cfg: dict,
        products_cfg: dict,
        warmup: int = 0,
        monitor_warmup: int = 0,
        monitor_interval: int = 0,
        log_interval: int = 0,
        training: bool = False,
        stores: Type[Stores] = Stores,
        monitor: Type[Monitor] = Monitor,
        production: Type[Production] = Production,
        scheduler: Type[Scheduler] = Scheduler,
        inbound: Type[Inbound] = Inbound,
        outbound: Type[Outbound] = Outbound,
        stores_kwargs: dict = None,
        monitor_kwargs: dict = None,
        production_kwargs: dict = None,
        scheduler_kwargs: dict = None,
        inbound_kwargs: dict = None,
        outbound_kwargs: dict = None,
        seed: int = None,
    ):
        super().__init__()

        self.stores_kwargs = stores_kwargs or {}
        self.monitor_kwargs = monitor_kwargs or {}
        self.production_kwargs = production_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.inbound_kwargs = inbound_kwargs or {}
        self.outbound_kwargs = outbound_kwargs or {}

        self.env = simpy.Environment()

        # Parameters
        self.run_until = run_until
        self.resources_config = resources_cfg
        self.products_config = products_cfg
        self.warmup = warmup
        self.monitor_warmup = monitor_warmup
        self.monitor_interval = monitor_interval
        self.log_interval = log_interval
        self.training = training
        self.seed = seed

        # Engine
        self.stores = stores(
            env=self.env,
            resources=self.resources_config,
            products=self.products_config,
            warmup=self.warmup,
            log_interval=self.log_interval,
            seed=self.seed,
            training=self.training,
            **self.stores_kwargs,
        )
        self.monitor = monitor(
            stores=self.stores,
            interval=self.monitor_interval,
            warmup=self.monitor_warmup,
            **self.monitor_kwargs,
        )
        self.production = production(self.stores, **self.production_kwargs)
        self.scheduler = scheduler(self.stores, **self.scheduler_kwargs)
        self.inbound = inbound(self.stores, **self.inbound_kwargs)
        self.outbound = outbound(self.stores, **self.outbound_kwargs)

    def run_simulation(self):
        print(self.run_until)
        self.env.run(until=self.run_until)

    def save_parameters(self, save_folder: Union[str, Path]):

        if not isinstance(save_folder, Path):
            save_folder = Path(save_folder)

        with open(save_folder / "config_products.yaml", "w") as file:
            yaml.dump(self.stores.products, file)
        with open(save_folder / "config_resources.yaml", "w") as file:
            yaml.dump(self.stores.resources, file)

        params = {
            "run_until": self.run_until,
            "warmup": self.warmup,
            "monitor_warmup": self.monitor_warmup,
            "monitor_interval": self.monitor_interval,
            "log_interval": self.log_interval,
            "training": self.training,
            "seed": self.seed,
            "stores_kwargs": self.stores_kwargs,
            "monitor_kwargs": self.monitor_kwargs,
            "production_kwargs": self.production_kwargs,
            "scheduler_kwargs": self.scheduler_kwargs,
            "inbound_kwargs": self.inbound_kwargs,
            "outbound_kwargs": self.outbound_kwargs,
        }
        with open(save_folder / "params.yaml", "w") as file:
            yaml.dump(params, file)
