from abc import ABC
from typing import Any, Callable, Dict, Literal, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
import simpy

from rlsim.engine.cli_config import create_simulation_parser
from rlsim.engine.logs import ProductLogs, ResourceLogs
from rlsim.engine.orders import DemandOrder, ProductionOrder
from rlsim.engine.stores import SimulationStores
from rlsim.engine.utils import DistributionGenerator, load_yaml

from time import time


class DeliveryMode(Enum):
    """Enum for delivery modes to improve type safety"""

    AS_READY = "asReady"
    ON_DUE = "onDue"
    INSTANTLY = "instantly"


class PrintMode(Enum):
    """Enum for print modes to improve type safety"""

    STATUS = "status"
    METRICS = "metrics"
    ALL = "all"
    ANY = "any"


@dataclass
class SimulationConfig:
    """Configuration class for better organization and validation"""

    run_until: float
    warmup: float = 0
    monitor_warmup: Optional[float] = None
    monitor_interval: Optional[int] = None
    delivery_mode: str = "asReady"
    log_interval: int = 72
    save_logs: bool = True

    def __post_init__(self):
        if self.monitor_warmup is None:
            self.monitor_warmup = self.warmup
        if self.monitor_interval is None:
            self.monitor_interval = int(self.run_until / 4)


class FactorySimulation(ABC):
    """
    Discrete event simulation for factory operations with support for
    multiple products, resources, and delivery modes.
    """

    def __init__(
        self,
        config: Union[dict, SimulationConfig],
        resources: Dict[str, dict],
        products: Dict[str, dict],
        save_logs: bool = True,
        print_mode: Literal["status", "metrics", "all"] = "metrics",
        seed: Optional[int] = None,
        queue_order_selection: Optional[Callable] = None,
    ):
        # Convert dict config to SimulationConfig if needed
        if isinstance(config, dict):
            self.config = self._parse_config(config, save_logs)
        else:
            self.config = config

        self.resources_config: Dict[str, dict] = resources
        self.products_config: Dict[str, dict] = products
        self.seed = seed
        self.print_mode = PrintMode(print_mode)
        self.queue_order_selection = queue_order_selection

        # Initialize logger
        self.logger = self._setup_logger()

        # Validation
        self._validate_configuration()

        # Initialize simulation components
        self._initiate_environment()

    def _parse_config(self, config_dict: dict, save_logs: bool) -> SimulationConfig:
        """Parse configuration dictionary into SimulationConfig object"""
        run_until = config_dict.get("run_until")
        if run_until is None:
            raise ValueError("run_until must be specified in configuration")

        return SimulationConfig(
            run_until=run_until,
            warmup=config_dict.get("warmup", 0),
            monitor_warmup=config_dict.get("monitor_warmup"),
            monitor_interval=config_dict.get("monitor_interval"),
            delivery_mode=config_dict.get("delivery_mode", "asReady"),
            log_interval=config_dict.get("log_interval", 72),
            save_logs=config_dict.get("save_logs", save_logs),
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for simulation events"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _validate_configuration(self) -> None:
        """Validate simulation configuration"""
        # Validate delivery mode
        try:
            DeliveryMode(self.config.delivery_mode)
        except ValueError:
            valid_modes = [mode.value for mode in DeliveryMode]
            raise ValueError(f"Invalid delivery_mode. Must be one of: {valid_modes}")

        # Validate products have required structure
        for product_name, product_config in self.products_config.items():
            if "demand" not in product_config:
                raise ValueError(
                    f"Product {product_name} missing 'demand' configuration"
                )
            if "processes" not in product_config:
                raise ValueError(
                    f"Product {product_name} missing 'processes' configuration"
                )

        # Validate resources have required structure
        for resource_name, resource_config in self.resources_config.items():
            if not isinstance(resource_config, dict):
                raise ValueError(
                    f"Resource {resource_name} configuration must be a dictionary"
                )

    def _initiate_environment(self) -> None:
        """Initialize environment classes, processes and stores"""
        self.logger.info("Initializing simulation environment")

        self.env = simpy.Environment()
        self.stores = SimulationStores(
            self.env, self.products_config, self.resources_config
        )

        # Initialize random number generators with better naming
        self._init_random_generators()

        # Initialize logging
        self._init_logging()

        # Start simulation processes
        self._start_resources()
        self._start_products()
        self._run_scheduler()
        self._run_monitor()

    def _init_random_generators(self) -> None:
        """Initialize random number generators for different purposes"""
        self.rnd_product = DistributionGenerator(self.seed)
        self.rnd_process = DistributionGenerator(self.seed)
        self.rnd_breakdown = DistributionGenerator(self.seed)

    def _init_logging(self) -> None:
        """Initialize logging components"""
        self.log_product = ProductLogs(self.products_config.keys())
        self.log_resource = ResourceLogs(self.resources_config.keys())

        if self.config.save_logs:
            self._register_log()

    def _register_log(self) -> None:
        """Register logging process"""

        def register_product_log():
            try:
                yield self.env.timeout(self.config.warmup)
                while True:
                    self._log_current_state()
                    yield self.env.timeout(self.config.log_interval)
            except Exception as e:
                self.logger.error(f"Error in product logging: {e}")

        self.env.process(register_product_log())

    def _log_current_state(self) -> None:
        """Log current state of products and WIP"""
        for product in self.products_config.keys():
            # Log finished goods
            fg_level = self.stores.finished_goods[product].level
            self.log_product.fg_log[product].append((self.env.now, fg_level))

            # Log WIP
            wip_level = self.stores.wip[product].level
            self.log_product.wip_log[product].append((self.env.now, wip_level))

    def _start_products(self) -> None:
        """Start product-related processes"""
        for product in self.products_config.keys():
            self.env.process(self._generate_demand_orders(product))
            self.env.process(self._delivery_orders(product))

    def _generate_demand_orders(self, product: str):
        """Generate demand orders"""
        try:
            product_config = self.products_config[product]
            demand_config = product_config["demand"]

            # Extract distribution parameters
            freq_params = self._extract_distribution_params(demand_config["freq"])
            quantity_params = self._extract_distribution_params(
                demand_config["quantity"]
            )
            due_params = self._extract_distribution_params(demand_config["duedate"])

            while True:
                # Generate order parameters
                frequency = np.float32(
                    self.rnd_product.random_number(
                        freq_params["dist"], freq_params["params"]
                    )
                )
                quantity = round(
                    self.rnd_product.random_number(
                        quantity_params["dist"], quantity_params["params"]
                    ),
                    0,
                )
                duedate_offset = np.float32(
                    self.rnd_product.random_number(
                        due_params["dist"], due_params["params"]
                    )
                )

                yield self.env.timeout(frequency)

                demand_order = self._create_demand_order(
                    product, quantity, duedate_offset
                )
                yield self.stores.inbound_demand_orders.put(demand_order)

        except Exception as e:
            self.logger.error(f"Error generating demand orders for {product}: {e}")

    def _extract_distribution_params(self, config: dict) -> dict:
        """Extract distribution parameters from configuration"""
        return {"dist": config.get("dist"), "params": config.get("params")}

    def _create_demand_order(
        self, product: str, quantity: float, duedate_offset: float
    ) -> DemandOrder:
        """Create a demand order"""
        duedate = self.env.now + duedate_offset

        return DemandOrder(
            product=product,
            quantity=quantity,
            duedate=duedate,
            arived=self.env.now,
            delivery_mode=self.config.delivery_mode,
        )

    def _run_scheduler(self) -> None:
        """Start the scheduler process"""
        self.env.process(self.scheduler())

    def scheduler(self):
        """Main scheduler process - can be overridden in subclasses"""
        while True:
            try:
                demand_order: DemandOrder = yield self.stores.inbound_demand_orders.get()

                production_order = ProductionOrder(
                    product=demand_order.product, quantity=demand_order.quantity
                )
                production_order.schedule = self.env.now
                production_order.duedate = demand_order.duedate
                production_order.priority = 0

                self.env.process(self._release_order(production_order))
                yield self.stores.outbound_demand_orders[demand_order.product].put(
                    demand_order
                )

            except Exception as e:
                self.logger.error(f"Error in scheduler: {e}")

    def _release_order(self, production_order: ProductionOrder):
        """Release production order to the shop floor"""
        try:
            product = production_order.product
            product_config = self.products_config[product]

            # Get process information
            processes = product_config["processes"]
            last_process = len(processes)
            first_process = next(iter(processes))
            first_resource = processes[first_process]["resource"]

            # Set order parameters
            production_order.process_total = last_process
            production_order.process_finished = 0
            production_order.released = self.env.now

            # Add to WIP and route to first resource
            yield self.stores.wip[product].put(production_order.quantity)
            yield self.stores.resource_input[first_resource].put(production_order)

            # Log release
            if self.config.save_logs and self.config.warmup <= self.env.now:
                self.log_product.released[product].append(
                    (self.env.now, production_order.quantity)
                )
        except Exception as e:
            self.logger.error(f"Error releasing order: {e}")

    def _start_resources(self) -> None:
        """Initialize and start all resources"""
        self.machine_down: Dict[str, simpy.Event] = {}
        self.resources: Dict[str, simpy.Resource] = {}

        for resource_name in self.resources_config:
            self._init_single_resource(resource_name)

    def _init_single_resource(self, resource_name: str) -> None:
        """Initialize a single resource with all its processes"""
        resource_config = self.resources_config[resource_name]
        quantity = resource_config.get("quantity", 1)

        # Create SimPy resource
        self.resources[resource_name] = simpy.Resource(self.env, quantity)

        # Initialize breakdown state
        self.machine_down[resource_name] = self.env.event()
        self.machine_down[resource_name].succeed()

        # Start resource processes
        if self._has_breakdown_config(resource_config):
            self.env.process(self._breakdowns(resource_name))

        self.env.process(self._transportation(resource_name))
        self.env.process(self._production_system(resource_name))

    def _has_breakdown_config(self, resource_config: dict) -> bool:
        """Check if resource has breakdown configuration"""
        return (
            resource_config.get("tbf") is not None
            and resource_config.get("ttr") is not None
        )

    def _breakdowns(self, resource: str):
        """Handle resource breakdowns"""
        try:
            resource_config = self.resources_config[resource]

            while True:
                # Get breakdown parameters
                tbf = self._get_breakdown_time(resource_config["tbf"])
                ttr = self._get_breakdown_time(resource_config["ttr"])

                # Wait for breakdown
                yield self.env.timeout(tbf)

                # Start breakdown
                self.machine_down[resource] = self.env.event()
                breakdown_start = self.env.now

                # Repair time
                yield self.env.timeout(ttr)
                self.machine_down[resource].succeed()
                breakdown_end = self.env.now

                # Log breakdown if after warmup
                if self.env.now >= self.config.warmup and self.config.save_logs:
                    duration = round(breakdown_end - breakdown_start, 6)
                    self.log_resource.breakdowns[resource].append(
                        (breakdown_start, duration)
                    )

        except Exception as e:
            self.logger.error(f"Error in breakdown process for {resource}: {e}")

    def _get_breakdown_time(self, breakdown_config: dict) -> float:
        """Get breakdown time from configuration"""
        dist = breakdown_config.get("dist", "constant")
        params = breakdown_config.get("params", [0])
        return self.rnd_breakdown.random_number(dist, params)

    def _transportation(self, resource: str):
        """Handle transportation between resources"""
        while True:
            try:
                production_order: ProductionOrder = yield self.stores.resource_output[
                    resource
                ].get()
                yield self.stores.resource_transport[resource].put(production_order)

                if self._is_order_complete(production_order):
                    yield from self._complete_order(production_order, resource)
                else:
                    yield from self._route_to_next_resource(production_order, resource)

            except Exception as e:
                self.logger.error(f"Error in transportation for {resource}: {e}")

    def _is_order_complete(self, production_order: ProductionOrder) -> bool:
        """Check if production order is complete"""
        return production_order.process_total == production_order.process_finished

    def _complete_order(self, production_order: ProductionOrder, resource: str):
        """Complete the production order"""
        product = production_order.product
        production_order.finished = self.env.now

        yield self.stores.resource_transport[resource].get()
        yield self.stores.finished_goods[product].put(production_order.quantity)
        yield self.stores.wip[product].get(production_order.quantity)

        # Log flow time if enabled
        if self.config.save_logs and self.env.now >= self.config.warmup:
            flow_time = self.env.now - production_order.released
            self.log_product.flow_time[product].append((self.env.now, flow_time))

    def _route_to_next_resource(
        self, production_order: ProductionOrder, current_resource: str
    ):
        """Route production order to next resource"""
        product = production_order.product
        process_id = production_order.process_finished
        next_resource = self.stores.processes_value_list[product][process_id][
            "resource"
        ]

        yield self.stores.resource_transport[current_resource].get()
        yield self.stores.resource_input[next_resource].put(production_order)

    def _production_system(self, resource: str):
        """Main production system for a resource"""
        last_process = None
        last_product = None

        while True:
            try:
                yield self.machine_down[resource]

                # Get order from queue
                production_order: ProductionOrder = yield from self._get_next_order(
                    resource
                )
                yield self.stores.resource_processing[resource].put(production_order)

                product = production_order.product
                process = production_order.process_finished

                # Handle setup
                setup_time = self._calculate_setup_time(
                    resource, product, process, last_product, last_process
                )
                last_process = process
                last_product = product

                # Process the order
                yield from self._process_order(resource, production_order, setup_time)

            except Exception as e:
                self.logger.error(f"Error in production system for {resource}: {e}")

    def _get_next_order(self, resource: str):
        """Get next order from resource queue"""
        queue_len = len(self.stores.resource_input[resource].items)

        if self.queue_order_selection is not None and queue_len > 1:
            order_id = self.queue_order_selection(self.stores, resource)
            production_order = yield self.stores.resource_input[resource].get(
                lambda item: item.id == order_id
            )
        else:
            production_order = yield self.stores.resource_input[resource].get()

        return production_order

    def _calculate_setup_time(
        self,
        resource: str,
        product: str,
        process: int,
        last_product: Optional[str],
        last_process: Optional[int],
    ) -> float:
        """Calculate setup time based on last processed item"""
        if last_product == product and last_process == process:
            return 0

        resource_config = self.resources_config[resource]
        setup_config = resource_config.get("setup", {})
        setup_dist = setup_config.get("dist", "constant")
        setup_params = setup_config.get("params", [0])

        setup_time = self.rnd_process.random_number(setup_dist, setup_params)

        # Log setup if after warmup
        if self.env.now >= self.config.warmup and self.config.save_logs:
            self.log_resource.setups[resource].append((self.env.now, setup_time))

        return setup_time

    def _process_order(
        self, resource: str, production_order: ProductionOrder, setup_time: float
    ):
        """Process the production order"""
        product = production_order.product
        process = production_order.process_finished

        with self.resources[resource].request() as req:
            yield req

            # Setup time
            if setup_time > 0:
                yield self.env.timeout(setup_time)

            # Get processing time parameters
            process_config = self.stores.processes_value_list[product][process][
                "processing_time"
            ]
            process_time_dist = process_config.get("dist")
            process_time_params = process_config.get("params")

            start_time = self.env.now

            # Process each part
            for part in range(int(production_order.quantity)):
                processing_time = self.rnd_process.random_number(
                    process_time_dist, process_time_params
                )
                yield self.env.timeout(processing_time)

            # Complete processing
            production_order.process_finished += 1
            end_time = self.env.now

            # Move order through system
            yield self.stores.resource_processing[resource].get()
            yield self.stores.resource_finished[resource].put(production_order)
            yield self.stores.resource_output[resource].put(production_order)

            # Log utilization
            if self.env.now >= self.config.warmup and self.config.save_logs:
                processing_duration = round(end_time - start_time, 6)
                self.log_resource.utilization[resource].append(
                    (self.env.now, processing_duration)
                )

    def _delivery_orders(self, product: str):
        """Handle delivery of orders for a product"""
        while True:
            try:
                demand_order: DemandOrder = yield self.stores.outbound_demand_orders[
                    product
                ].get()
                delivery_mode = DeliveryMode(demand_order.delivery_mode)

                if delivery_mode == DeliveryMode.AS_READY:
                    self.env.process(self._delivery_as_ready(demand_order))
                elif delivery_mode == DeliveryMode.ON_DUE:
                    self.env.process(self._delivery_on_duedate(demand_order))
                elif delivery_mode == DeliveryMode.INSTANTLY:
                    self.env.process(self._delivery_instantly(demand_order))

            except Exception as e:
                self.logger.error(f"Error in delivery process for {product}: {e}")

    def _delivery_instantly(self, demand_order: DemandOrder):
        """Handle instantly delivery mode"""
        quantity = demand_order.quantity
        product = demand_order.product

        if self.stores.finished_goods[product].level >= quantity:
            yield self.stores.finished_goods[product].get(quantity)
            if self.config.save_logs and self.env.now > self.config.warmup:
                self.log_product.delivered_ontime[product].append(
                    (self.env.now, quantity)
                )
        elif self.env.now >= self.config.warmup and self.config.save_logs:
            self.log_product.lost_sales[product].append((self.env.now, quantity))

    def _delivery_as_ready(self, demand_order: DemandOrder):
        """Handle as-ready delivery mode"""
        quantity = demand_order.quantity
        product = demand_order.product

        # Remove from finished goods
        yield self.stores.finished_goods[product].get(quantity)

        # Update delivery time and log
        demand_order.delivered = self.env.now

        if self.config.save_logs and self.env.now >= self.config.warmup:
            self._log_delivery_performance(demand_order)
            self._log_lead_time(demand_order)

    def _delivery_on_duedate(self, demand_order: DemandOrder):
        """Handle on-duedate delivery mode"""

        def _delivery_order(demand_order: DemandOrder):
            quantity = demand_order.quantity
            duedate = demand_order.duedate
            product = demand_order.product

            # Wait for due date
            delay = max(0, duedate - self.env.now)
            if delay > 0:
                yield self.env.timeout(delay)

            # Remove from finished goods
            yield self.stores.finished_goods[product].get(quantity)

            # Update delivery time and log
            demand_order.delivered = self.env.now

            if self.config.save_logs and self.env.now >= self.config.warmup:
                self._log_delivery_performance(demand_order)
                self._log_lead_time(demand_order)

        self.env.process(_delivery_order(demand_order))

    def _log_delivery_performance(self, demand_order: DemandOrder) -> None:
        """Log delivery performance metrics"""
        product = demand_order.product
        quantity = demand_order.quantity

        if demand_order.delivered <= demand_order.duedate:
            self.log_product.delivered_ontime[product].append((self.env.now, quantity))
            earliness = demand_order.duedate - self.env.now
            self.log_product.earliness[product].append((self.env.now, earliness))
        else:
            self.log_product.delivered_late[product].append((self.env.now, quantity))
            tardiness = self.env.now - demand_order.duedate
            self.log_product.tardiness[product].append((self.env.now, tardiness))

    def _log_lead_time(self, demand_order: DemandOrder) -> None:
        """Log lead time for the order"""
        product = demand_order.product
        lead_time = self.env.now - demand_order.arived
        self.log_product.lead_time[product].append((self.env.now, lead_time))

    def _run_monitor(self) -> None:
        """Start the monitoring process"""
        if self.print_mode != PrintMode.ANY:
            self.env.process(self._monitor())

    def _monitor(self):
        """Monitor simulation progress and print metrics"""
        start_time = time()
        yield self.env.timeout(self.config.monitor_warmup)

        while True:
            try:
                end_time = time()
                elapsed_time = end_time - start_time

                print("\n" + "=" * 50)
                print(f"Simulation Time: {self.env.now}")
                print(f"Elapsed Real Time: {elapsed_time:.4f} seconds")
                print("=" * 50)

                if self.print_mode == PrintMode.ALL:
                    self._print_all_metrics()
                elif self.print_mode == PrintMode.METRICS:
                    self._print_metrics_only()
                elif self.print_mode == PrintMode.STATUS:
                    self._print_status_only()

                yield self.env.timeout(self.config.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitor: {e}")

    def _print_all_metrics(self) -> None:
        """Print all available metrics and status"""
        snapshot = self.stores.simulation_snapshot()
        print("SYSTEM STATUS:")
        print(snapshot)
        print("\n")

        self._print_metrics_only()

    def _print_metrics_only(self) -> None:
        """Print only performance metrics"""
        products = self.log_product.calculate_metrics()
        print("PRODUCT METRICS:")
        print(products)
        print("\n")

        resources = self.log_resource.calculate_metrics()
        if not resources.empty:
            resources.loc[:, "utilization"] = (
                resources.loc[:, "utilization"] / self.env.now
            )
            print("RESOURCE METRICS:")
            print(resources)

    def _print_status_only(self) -> None:
        """Print only system status"""
        snapshot = self.stores.simulation_snapshot()
        print("SYSTEM STATUS:")
        print(snapshot)
        print("\n")

    def reset_simulation(self, seed: Optional[int] = None) -> None:
        """Reset simulation with optional new seed"""
        if seed is not None:
            self.seed = seed
        self.logger.info("Resetting simulation")
        self._initiate_environment()

    def run_simulation(self) -> None:
        """Run the simulation until completion"""
        self.logger.info(f"Starting simulation - running until {self.config.run_until}")
        start_time = time()

        try:
            self.env.run(until=self.config.run_until)
            end_time = time()
            self.logger.info(
                f"Simulation completed in {end_time - start_time:.2f} seconds"
            )
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise

    def get_simulation_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results"""
        return {
            "product_metrics": self.log_product.calculate_metrics(),
            "resource_metrics": self.log_resource.calculate_metrics(),
            "system_snapshot": self.stores.simulation_snapshot(),
            "simulation_time": self.env.now,
            "configuration": {
                "run_until": self.config.run_until,
                "warmup": self.config.warmup,
                "delivery_mode": self.config.delivery_mode,
                "seed": self.seed,
            },
        }


if __name__ == "__main__":
    from pathlib import Path

    parser = create_simulation_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    # Use provided paths or defaults
    config_path = (
        args_dict["config"]
        if args_dict["config"] is not None
        else Path("src/rlsim/config/config.yaml")
    )

    product_path = (
        args_dict["products"]
        if args_dict["products"] is not None
        else Path("src/rlsim/config/products.yaml")
    )

    resource_path = (
        args_dict["resources"]
        if args_dict["resources"] is not None
        else Path("src/rlsim/config/resources.yaml")
    )

    # Check if files exist
    if config_path.exists() and product_path.exists() and resource_path.exists():
        print("Loading configuration from files...")
        config = load_yaml(config_path)
        products = load_yaml(product_path)
        resources = load_yaml(resource_path)

        sim = FactorySimulation(
            config,
            resources,
            products,
            save_logs=args_dict.get("save_logs", True),
            print_mode=args_dict.get("print_mode", "metrics"),
            seed=args_dict.get("seed", 123),
        )

        print("Running simulation from config files...")
        print(f"- Config: {config_path}")
        print(f"- Products: {product_path}")
        print(f"- Resources: {resource_path}")

        sim.run_simulation()

    else:
        missing_files = []
        if not config_path.exists():
            missing_files.append(str(config_path))
        if not product_path.exists():
            missing_files.append(str(product_path))
        if not resource_path.exists():
            missing_files.append(str(resource_path))

        print(f"Warning: Configuration files not found: {missing_files}")
        print("Simulation aborted...")
