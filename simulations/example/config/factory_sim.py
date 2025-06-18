# factory_sim.py - Main simulation interface
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import simpy
import pandas as pd
from abc import ABC, abstractmethod


@dataclass
class SimulationConfig:
    """Centralized configuration for the simulation"""

    run_until: int = 10000
    warmup_time: int = 1000
    log_interval: int = 100
    monitor_interval: int = 500
    seed: Optional[int] = None

    # Delivery modes
    delivery_mode: str = "asReady"  # "asReady", "onDue", "instantly"

    # Enable/disable features
    enable_monitoring: bool = True
    enable_breakdowns: bool = True
    enable_setups: bool = True


class FactorySimulation:
    """Main simulation interface - simplified entry point"""

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.env = simpy.Environment()
        self.metrics = SimulationMetrics()
        self.resources = {}
        self.products = {}
        self._components = {}

    def add_resource(
        self,
        name: str,
        capacity: int = 1,
        setup_time: Union[float, Dict] = 0,
        breakdown_config: Optional[Dict] = None,
    ):
        """Add a resource to the factory"""
        self.resources[name] = {
            "capacity": capacity,
            "setup": self._normalize_distribution(setup_time),
            "breakdown": breakdown_config or {},
        }
        return self

    def add_product(self, name: str, processes: List[Dict], demand_config: Dict):
        """Add a product with its production processes"""
        self.products[name] = {
            "processes": {
                f"{name}_step_{i}": process for i, process in enumerate(processes, 1)
            },
            "demand": demand_config,
        }
        return self

    def load_from_yaml(self, resources_file: str, products_file: str):
        """Load configuration from YAML files"""
        with open(resources_file, "r") as f:
            resources = yaml.safe_load(f)
        with open(products_file, "r") as f:
            products = yaml.safe_load(f)

        self.resources.update(resources)
        self.products.update(products)
        return self

    def set_scheduling_policy(self, policy: Union[str, Callable] = "FIFO"):
        """Set the scheduling policy for order selection"""
        if isinstance(policy, str):
            policies = {
                "FIFO": self._fifo_policy,
                "SPT": self._spt_policy,
                "EDD": self._edd_policy,
                "PRIORITY": self._priority_policy,
            }
            self._scheduling_policy = policies.get(policy, self._fifo_policy)
        else:
            self._scheduling_policy = policy
        return self

    def run(self) -> "SimulationResults":
        """Run the simulation and return results"""
        # Initialize internal components
        self._initialize_simulation()

        # Run simulation
        self.env.run(until=self.config.run_until)

        # Return results
        return SimulationResults(self.metrics, self.config)

    def _initialize_simulation(self):
        """Initialize all simulation components"""
        # Create the core simulation engine
        self._engine = SimulationEngine(
            self.env, self.resources, self.products, self.config, self.metrics
        )

        # Start simulation processes
        self._engine.start()

    def _normalize_distribution(self, value):
        """Convert simple values to distribution format"""
        if isinstance(value, (int, float)):
            return {"dist": "constant", "params": [value]}
        return value

    # Built-in scheduling policies
    def _fifo_policy(self, queue):
        return queue[0].id if queue else None

    def _spt_policy(self, queue):
        return min(queue, key=lambda x: x.processing_time).id if queue else None

    def _edd_policy(self, queue):
        return min(queue, key=lambda x: x.duedate).id if queue else None

    def _priority_policy(self, queue):
        return min(queue, key=lambda x: x.priority).id if queue else None


class SimulationEngine:
    """Internal simulation engine - handles the core simulation logic"""

    def __init__(self, env, resources, products, config, metrics):
        self.env: simpy.Environment = env
        self.resources = resources
        self.products = products
        self.config = config
        self.metrics = metrics

        # Initialize stores and resources
        self._init_stores()
        self._init_resources()

    def _init_stores(self):
        """Initialize all SimPy stores"""
        self.demand_orders = simpy.Store(self.env)
        self.finished_goods = {
            product: simpy.Container(self.env) for product in self.products
        }
        self.work_in_process = {
            product: simpy.Container(self.env) for product in self.products
        }

        # Resource queues
        self.resource_queues = {}
        for resource in self.resources:
            self.resource_queues[resource] = {
                "input": simpy.FilterStore(self.env),
                "processing": simpy.Store(self.env),
                "output": simpy.Store(self.env),
            }

    def _init_resources(self):
        """Initialize SimPy resources"""
        self.simpy_resources = {}
        for name, config in self.resources.items():
            capacity = config.get("capacity", 1)
            self.simpy_resources[name] = simpy.Resource(self.env, capacity)

    def start(self):
        """Start all simulation processes"""
        # Start demand generation
        for product in self.products:
            self.env.process(self._generate_demand(product))

        # Start production processes
        for resource in self.resources:
            self.env.process(self._production_process(resource))

        # Start delivery process
        for product in self.products:
            self.env.process(self._delivery_process(product))

        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self.env.process(self._monitor_process())

    def _generate_demand(self, product):
        """Generate demand orders for a product"""
        demand_config = self.products[product]["demand"]
        dist = DistributionGenerator(self.config.seed)

        while True:
            # Generate inter-arrival time
            freq_config = demand_config["freq"]
            inter_arrival = dist.sample(freq_config["dist"], freq_config["params"])
            yield self.env.timeout(inter_arrival)

            # Generate order quantity
            qty_config = demand_config["quantity"]
            quantity = dist.sample(qty_config["dist"], qty_config["params"])

            # Generate due date
            due_config = demand_config["duedate"]
            due_offset = dist.sample(due_config["dist"], due_config["params"])

            # Create order
            order = ProductionOrder(
                product=product,
                quantity=int(quantity),
                arrival_time=self.env.now,
                due_date=self.env.now + due_offset,
            )

            yield self.demand_orders.put(order)

    def _production_process(self, resource_name):
        """Handle production at a resource"""
        resource = self.simpy_resources[resource_name]
        queue = self.resource_queues[resource_name]

        while True:
            # Get next order
            order = yield queue["input"].get()

            with resource.request() as req:
                yield req

                # Setup time
                setup_time = self._get_setup_time(resource_name, order)
                if setup_time > 0:
                    yield self.env.timeout(setup_time)

                # Processing time
                process_time = self._get_process_time(order)
                start_time = self.env.now
                yield self.env.timeout(process_time)

                # Update metrics
                self.metrics.record_utilization(
                    resource_name, self.env.now - start_time
                )

                # Move to next stage or completion
                yield queue["output"].put(order)

    def _delivery_process(self, product):
        """Handle delivery of finished products"""
        # Implementation depends on delivery mode
        pass

    def _monitor_process(self):
        """Monitor simulation progress"""
        yield self.env.timeout(self.config.warmup_time)

        while True:
            # Collect metrics
            self.metrics.collect_snapshot(
                self.env.now, self.finished_goods, self.work_in_process
            )
            yield self.env.timeout(self.config.monitor_interval)

    def _get_setup_time(self, resource_name, order):
        """Calculate setup time for resource"""
        # Implementation for setup time calculation
        return 0

    def _get_process_time(self, order):
        """Calculate processing time for order"""
        # Implementation for processing time calculation
        return 1


@dataclass
class ProductionOrder:
    """Represents a production order"""

    product: str
    quantity: int
    arrival_time: float
    due_date: float
    priority: int = 0
    current_process: int = 0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    id: int = field(init=False)

    _counter = 0

    def __post_init__(self):
        ProductionOrder._counter += 1
        self.id = ProductionOrder._counter


class SimulationMetrics:
    """Centralized metrics collection"""

    def __init__(self):
        self.data = {
            "throughput": [],
            "utilization": {},
            "wip_levels": [],
            "lead_times": [],
            "tardiness": [],
            "service_level": [],
        }

    def record_utilization(self, resource: str, duration: float):
        """Record resource utilization"""
        if resource not in self.data["utilization"]:
            self.data["utilization"][resource] = []
        self.data["utilization"][resource].append(duration)

    def collect_snapshot(self, time: float, finished_goods: Dict, wip: Dict):
        """Collect system snapshot"""
        total_wip = sum(container.level for container in wip.values())
        self.data["wip_levels"].append((time, total_wip))

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Convert metrics to pandas DataFrames"""
        return {
            "utilization": self._utilization_to_df(),
            "wip": pd.DataFrame(self.data["wip_levels"], columns=["time", "wip"]),
            # Add other metrics...
        }

    def _utilization_to_df(self) -> pd.DataFrame:
        """Convert utilization data to DataFrame"""
        data = []
        for resource, times in self.data["utilization"].items():
            for time_val in times:
                data.append({"resource": resource, "utilization_time": time_val})
        return pd.DataFrame(data)


class SimulationResults:
    """Container for simulation results with analysis methods"""

    def __init__(self, metrics: SimulationMetrics, config: SimulationConfig):
        self.metrics = metrics
        self.config = config
        self._dataframes = None

    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        """Get results as pandas DataFrames"""
        if self._dataframes is None:
            self._dataframes = self.metrics.to_dataframe()
        return self._dataframes

    def summary(self) -> Dict:
        """Get summary statistics"""
        return {
            "avg_utilization": self._avg_utilization(),
            "avg_wip": self._avg_wip(),
            "throughput": self._throughput(),
            # Add more summary stats...
        }

    def _avg_utilization(self) -> Dict[str, float]:
        """Calculate average utilization per resource"""
        util_df = self.dataframes["utilization"]
        return util_df.groupby("resource")["utilization_time"].mean().to_dict()

    def _avg_wip(self) -> float:
        """Calculate average WIP level"""
        wip_df = self.dataframes["wip"]
        return wip_df["wip"].mean() if not wip_df.empty else 0

    def _throughput(self) -> float:
        """Calculate system throughput"""
        # Implementation for throughput calculation
        return 0

    def plot_utilization(self):
        """Plot resource utilization"""
        # Implementation for plotting
        pass

    def plot_wip_levels(self):
        """Plot WIP levels over time"""
        # Implementation for plotting
        pass


class DistributionGenerator:
    """Utility class for generating random numbers from distributions"""

    def __init__(self, seed: Optional[int] = None):
        import random
        import numpy as np

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def sample(self, dist_type: str, params: List[float]) -> float:
        """Sample from a distribution"""
        if dist_type == "constant":
            return params[0]
        elif dist_type == "uniform":
            return self.rng.uniform(params[0], params[1])
        elif dist_type == "normal":
            return self.np_rng.normal(params[0], params[1])
        elif dist_type == "exponential":
            return self.np_rng.exponential(params[0])
        elif dist_type == "gamma":
            return self.np_rng.gamma(params[0], params[1])
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")


# Example usage and simplified interface
class FactoryBuilder:
    """Builder pattern for easy factory construction"""

    def __init__(self):
        self.sim = FactorySimulation()

    def simple_line(self, stations: List[str], processing_times: List[float]):
        """Create a simple production line"""
        # Add resources
        for station in stations:
            self.sim.add_resource(station, capacity=1)

        # Create a simple product
        processes = []
        for i, (station, time) in enumerate(zip(stations, processing_times)):
            processes.append(
                {
                    "resource": station,
                    "processing_time": {"dist": "constant", "params": [time]},
                }
            )

        self.sim.add_product(
            "product",
            processes,
            {
                "freq": {"dist": "exponential", "params": [10]},
                "quantity": {"dist": "constant", "params": [1]},
                "duedate": {"dist": "constant", "params": [100]},
            },
        )

        return self.sim

    def job_shop(self, resources: Dict[str, int], products: Dict[str, List]):
        """Create a job shop configuration"""
        # Add resources with capacities
        for resource, capacity in resources.items():
            self.sim.add_resource(resource, capacity=capacity)

        # Add products with routing
        for product_name, routing in products.items():
            processes = []
            for step in routing:
                processes.append(
                    {
                        "resource": step["resource"],
                        "processing_time": step.get(
                            "time", {"dist": "constant", "params": [1]}
                        ),
                    }
                )

            # Default demand pattern
            self.sim.add_product(
                product_name,
                processes,
                {
                    "freq": {"dist": "exponential", "params": [20]},
                    "quantity": {"dist": "constant", "params": [1]},
                    "duedate": {"dist": "constant", "params": [100]},
                },
            )

        return self.sim


# Simplified usage examples
def example_simple_usage():
    """Example of simple usage"""

    # # Method 1: Builder pattern for common configurations
    # factory = FactoryBuilder().simple_line(
    #     stations=["cutting", "assembly", "packaging"], processing_times=[5, 8, 3]
    # )

    # # Method 2: Fluent interface for custom configuration
    # factory = (
    #     FactorySimulation()
    #     .add_resource("machine1", capacity=2, setup_time=2)
    #     .add_resource("machine2", capacity=1, setup_time=1)
    #     .add_product(
    #         "product_a",
    #         [
    #             {
    #                 "resource": "machine1",
    #                 "processing_time": {"dist": "normal", "params": [5, 1]},
    #             },
    #             {
    #                 "resource": "machine2",
    #                 "processing_time": {"dist": "constant", "params": [3]},
    #             },
    #         ],
    #         {
    #             "freq": {"dist": "exponential", "params": [10]},
    #             "quantity": {"dist": "constant", "params": [1]},
    #             "duedate": {"dist": "constant", "params": [50]},
    #         },
    #     )
    #     .set_scheduling_policy("SPT")
    # )

    # Method 3: YAML configuration (backwards compatible)
    factory = FactorySimulation().load_from_yaml("resources.yaml", "products.yaml")

    # Run simulation
    config = SimulationConfig(run_until=10000, warmup_time=1000)
    factory.config = config
    results = factory.run()

    # Analyze results
    print(results.summary())
    results.plot_utilization()

    # Access raw data
    dfs = results.dataframes
    utilization_df = dfs["utilization"]

    return results
