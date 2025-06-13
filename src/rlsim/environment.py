from pathlib import Path
import random
from typing import Type, Union, Dict, Any
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

        # Store initialization parameters for reset
        self._init_params = {
            "run_until": run_until,
            "resources_cfg": resources_cfg.copy() if resources_cfg else {},
            "products_cfg": products_cfg.copy() if products_cfg else {},
            "warmup": warmup,
            "monitor_warmup": monitor_warmup,
            "monitor_interval": monitor_interval,
            "log_interval": log_interval,
            "training": training,
            "stores": stores,
            "monitor": monitor,
            "production": production,
            "scheduler": scheduler,
            "inbound": inbound,
            "outbound": outbound,
            "stores_kwargs": stores_kwargs.copy() if stores_kwargs else {},
            "monitor_kwargs": monitor_kwargs.copy() if monitor_kwargs else {},
            "production_kwargs": production_kwargs.copy() if production_kwargs else {},
            "scheduler_kwargs": scheduler_kwargs.copy() if scheduler_kwargs else {},
            "inbound_kwargs": inbound_kwargs.copy() if inbound_kwargs else {},
            "outbound_kwargs": outbound_kwargs.copy() if outbound_kwargs else {},
            "seed": seed,
        }

        self.stores_kwargs = stores_kwargs or {}
        self.monitor_kwargs = monitor_kwargs or {}
        self.production_kwargs = production_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.inbound_kwargs = inbound_kwargs or {}
        self.outbound_kwargs = outbound_kwargs or {}

        # Initialize the environment
        self._initialize_environment()

    def _initialize_environment(self):
        """Initialize or reinitialize the simulation environment"""
        self.env = simpy.Environment()

        # Parameters
        self.run_until = self._init_params["run_until"]
        self.resources_config = self._init_params["resources_cfg"]
        self.products_config = self._init_params["products_cfg"]
        self.warmup = self._init_params["warmup"]
        self.monitor_warmup = self._init_params["monitor_warmup"]
        self.monitor_interval = self._init_params["monitor_interval"]
        self.log_interval = self._init_params["log_interval"]
        self.training = self._init_params["training"]
        self.seed = self._init_params["seed"]

        # Engine components
        self.stores = self._init_params["stores"](
            env=self.env,
            resources=self.resources_config,
            products=self.products_config,
            warmup=self.warmup,
            log_interval=self.log_interval,
            seed=self.seed,
            training=self.training,
            **self.stores_kwargs,
        )
        self.monitor = self._init_params["monitor"](
            stores=self.stores,
            interval=self.monitor_interval,
            warmup=self.monitor_warmup,
            **self.monitor_kwargs,
        )
        self.production = self._init_params["production"](
            self.stores, **self.production_kwargs
        )
        self.scheduler = self._init_params["scheduler"](
            self.stores, **self.scheduler_kwargs
        )
        self.inbound = self._init_params["inbound"](self.stores, **self.inbound_kwargs)
        self.outbound = self._init_params["outbound"](
            self.stores, **self.outbound_kwargs
        )

    def reset(self, seed: int = None, **kwargs):
        """
        Reset the simulation environment to its initial state.

        Args:
            seed: Optional new seed for random number generation
            **kwargs: Optional parameters to update for the reset
        """
        # Update parameters if provided
        if seed is not None:
            self._init_params["seed"] = seed
            self.seed = seed

        # Update any other parameters passed via kwargs
        for key, value in kwargs.items():
            if key in self._init_params:
                self._init_params[key] = value
                setattr(self, key, value)

        # Reinitialize the environment
        self._initialize_environment()

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
        with open(save_folder / "config_products.yaml", "w") as file:
            yaml.dump(params, file)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
