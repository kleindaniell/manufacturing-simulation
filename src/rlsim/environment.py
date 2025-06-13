import random
from typing import Type
import simpy

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

        stores_kwargs = stores_kwargs or {}
        monitor_kwargs = monitor_kwargs or {}
        production_kwargs = production_kwargs or {}
        scheduler_kwargs = scheduler_kwargs or {}
        inbound_kwargs = inbound_kwargs or {}
        outbound_kwargs = outbound_kwargs or {}

        self.env = simpy.Environment()

        # Parameters
        self.run_until = run_until
        self.resources_config = resources_cfg
        self.products_config = products_cfg
        self.warmup = warmup
        self.monitor_warmup = monitor_warmup
        self.monitor_interval = monitor_interval
        self.log_interval = log_interval
        self.seed = seed

        # Engine
        self.stores = stores(
            env=self.env,
            resources=self.resources_config,
            products=self.products_config,
            warmup=self.warmup,
            log_interval=self.log_interval,
            seed=self.seed,
            **stores_kwargs,
        )
        self.monitor = monitor(
            stores=self.stores,
            interval=self.monitor_interval,
            warmup=self.monitor_warmup,
            **monitor_kwargs,
        )
        self.production = production(self.stores, **production_kwargs)
        self.scheduler = scheduler(self.stores, **scheduler_kwargs)
        self.inbound = inbound(self.stores, **inbound_kwargs)
        self.outbound = outbound(self.stores, **outbound_kwargs)

    def run_simulation(self):
        print(self.run_until)
        self.env.run(until=self.run_until)
