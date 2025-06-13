import argparse
import random
from pathlib import Path
from typing import Type
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
        random.seed(seed)

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

        # Engine
        self.stores = stores(
            env=self.env,
            resources=self.resources_config,
            products=self.products_config,
            warmup=self.warmup,
            log_interval=self.log_interval,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation environment")

    parser.add_argument(
        "--resources", type=Path, required=True, help="Path to resources config YAML"
    )
    parser.add_argument(
        "--products", type=Path, required=True, help="Path to products config YAML"
    )
    parser.add_argument(
        "--run-until", type=int, default=200001, help="Simulation end time"
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=50000, help="Monitor sampling interval"
    )
    parser.add_argument(
        "--log-interval", type=int, default=48, help="Log sampling interval"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup duration for start logging results",
    )
    parser.add_argument(
        "--monitor-warmup", type=int, default=0, help="Warmup duration for monitor"
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config files
    with open(args.resources, "r") as f:
        resources_cfg = yaml.safe_load(f)

    with open(args.products, "r") as f:
        products_cfg = yaml.safe_load(f)

    # Instantiate and run simulation
    sim = Environment(
        run_until=args.run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        monitor_interval=args.monitor_interval,
        log_interval=args.log_interval,
        monitor_warmup=args.monitor_warmup,
        warmup=args.warmup,
        seed=args.seed,
    )

    sim.run_simulation()
