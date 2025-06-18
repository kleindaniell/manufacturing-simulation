from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal

import numpy as np
import simpy
from pathlib import Path
import json
import yaml

from rlsim.engine.cli_config import create_simulation_parser
from rlsim.engine.logs import ProductLogs, ResourceLogs, GeneralLogs
from rlsim.engine.orders import DemandOrder, ProductionOrder
from rlsim.engine.stores import SimulationStores
from rlsim.engine.utils import DistributionGenerator, load_yaml

from time import time


class FactorySimulation(ABC):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        save_logs: bool = True,
        print_mode: Literal["status", "metrics", "all"] = "metrics",
        seed: int = None,
        custom_logs: dict = None,
        queue_order_selection: Callable = None,
    ):
        self.config: Dict[str, Any] = config
        self.resources_config: Dict[str, dict] = resources
        self.products_config: Dict[str, dict] = products
        self.seed = seed
        self.print_mode = print_mode
        self.custom_logs = custom_logs
        self.queue_order_selection = queue_order_selection

        self.run_until = self.config.get("run_until", None)
        if self.run_until is None:
            raise ValueError("run_until must be specified")
        self.warmup = self.config.get("warmup", 0)
        self.monitor_warmup = self.config.get("monitor_warmup", self.warmup)
        self.monitor_interval = self.config.get(
            "monitor_interval", int(self.run_until / 4)
        )
        self.delivery_mode = self.config.get("delivery_mode", "asReady")
        self.log_interval = self.config.get("log_interval", 72)
        self.save_logs = self.config.get("save_logs", save_logs)

        self._initiate_environment()

    def _initiate_environment(self):
        """Start environment classes, processes and stores"""

        self.env = simpy.Environment()

        self.stores = SimulationStores(
            self.env, self.products_config, self.resources_config
        )

        # Init logging components
        self._init_loggint()

        # Init random numbers generator
        self._init_random_generators()

        # Start Simulation processes
        self._start_resources()
        self._start_products()
        self._run_scheduler()
        self._run_monitor()
        self._start_custom_process()

    def _init_random_generators(self) -> None:
        """Initialize random number generators for different purposes"""
        self.rnd_product = DistributionGenerator(self.seed)
        self.rnd_process = DistributionGenerator(self.seed)
        self.rnd_breakdown = DistributionGenerator(self.seed)

    def _init_loggint(self) -> None:
        """Initialize logging components"""
        custom_logs = self._create_custom_logs()

        product_custom_logs = custom_logs.get("products", {})
        self.log_product = ProductLogs(
            self.products_config.keys(), **product_custom_logs
        )
        resource_custom_logs = custom_logs.get("resources", {})
        self.log_resource = ResourceLogs(
            self.resources_config.keys(), **resource_custom_logs
        )
        general_custom_logs = custom_logs.get("general", {})
        self.log_general = GeneralLogs(**general_custom_logs)

        if self.save_logs:
            self._register_log()
            self._register_custom_logs()

    def _register_log(self) -> None:
        def register_product_log():
            yield self.env.timeout(self.warmup)
            while True:
                for product in self.products_config.keys():
                    # Log finished goods level
                    fg_level = self.stores.finished_goods[product].level
                    self.log_product.fg_log[product].append((self.env.now, fg_level))
                    # Log WIP
                    wip = self.stores.wip[product].level
                    self.log_product.wip_log[product].append((self.env.now, wip))

                yield self.env.timeout(self.log_interval)

        self.env.process(register_product_log())

    @abstractmethod
    def _create_custom_logs(self):
        pass

    @abstractmethod
    def _register_custom_logs(self):
        pass

    def _start_products(self):
        for product in self.products_config.keys():
            self.env.process(self._generate_demand_orders(product))
            self.env.process(self._delivery_orders(product))

    def _generate_demand_orders(self, product):
        product_config = self.products_config[product]

        freq_dist = product_config["demand"]["freq"].get("dist")
        freq_params = product_config["demand"]["freq"].get("params")
        quantity_dist = product_config["demand"]["quantity"].get("dist")
        quantity_params = product_config["demand"]["quantity"].get("params")
        due_dist = product_config["demand"]["duedate"].get("dist")
        due_params = product_config["demand"]["duedate"].get("params")

        while True:
            frequency = np.float32(
                self.rnd_product.random_number(freq_dist, freq_params)
            )

            quantity = round(
                self.rnd_product.random_number(quantity_dist, quantity_params), 0
            )
            duedate = np.float32(self.rnd_product.random_number(due_dist, due_params))

            yield self.env.timeout(frequency)

            duedate += self.env.now

            demandOrder = DemandOrder(
                product=product,
                quantity=quantity,
                duedate=duedate,
                arived=self.env.now,
                delivery_mode=self.delivery_mode,
            )
            # print(f"{self.env.now} - {demandOrder}")
            yield self.stores.inbound_demand_orders.put(demandOrder)

    def _run_scheduler(self):
        self.env.process(self.scheduler())

    def scheduler(self):
        print("====== running scheduler ========")
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            quantity = demandOrder.quantity

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = self.env.now
            productionOrder.duedate = demandOrder.duedate
            productionOrder.priority = 0

            self.env.process(self._release_order(productionOrder))

            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def _release_order(self, productionOrder: ProductionOrder):
        product = productionOrder.product
        product_config = self.products_config[product]

        # Get processes info
        processes = product_config["processes"]
        last_process = len(processes)
        first_process = next(iter(processes))
        first_resource = processes[first_process]["resource"]

        # Set order parameters
        productionOrder.process_total = last_process
        productionOrder.process_finished = 0
        productionOrder.released = self.env.now

        # Add order to firts product and increment WIP
        yield self.stores.resource_input[first_resource].put(productionOrder)
        yield self.stores.wip[product].put(productionOrder.quantity)

        # Log release products
        if self.warmup <= self.env.now and self.save_logs:
            self.log_product.released[product].append(
                (self.env.now, productionOrder.quantity)
            )

    def _start_resources(self) -> None:
        self.machine_down: Dict[str, simpy.Event] = {}
        self.resources: Dict[str, simpy.Resource] = {}

        for resource in self.resources_config:
            resource_config: dict = self.resources_config.get(resource)
            quantity = resource_config.get("quantity", 1)

            # Create Resource
            self.resources[resource] = simpy.Resource(self.env, quantity)

            # Initialize breakdown state
            self.machine_down[resource] = self.env.event()
            self.machine_down[resource].succeed()

            # Start resource process
            if self.resources_config[resource].get(
                "tbf", None
            ) and self.resources_config[resource].get("ttr", None):
                self.env.process(self._breakdowns(resource))

            self.env.process(self._transportation(resource))
            self.env.process(self._production_system(resource))

    def _breakdowns(self, resource):
        """Handle resources breakdowns"""
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
            # Log breakdown
            if self.env.now >= self.warmup and self.save_logs:
                self.log_resource.breakdowns[resource].append(
                    (breakdown_start, round(breakdown_end - breakdown_start, 6))
                )

    def _get_breakdown_time(self, breakdown_config: dict) -> float:
        """Get breakdown time from configuration"""
        dist = breakdown_config.get("dist", "constant")
        params = breakdown_config.get("params", [0])
        return self.rnd_breakdown.random_number(dist, params)

    def _transportation(self, resource):
        while True:
            # Get order from output
            productionOrder: ProductionOrder = yield self.stores.resource_output[
                resource
            ].get()
            # Put order on transportation
            yield self.stores.resource_transport[resource].put(productionOrder)

            product = productionOrder.product
            # Order finished
            if productionOrder.process_total == productionOrder.process_finished:
                productionOrder.finished = self.env.now
                yield self.stores.resource_transport[resource].get()
                yield self.stores.finished_goods[product].put(productionOrder.quantity)
                yield self.stores.wip[product].get(productionOrder.quantity)

                # Log flow time
                if self.save_logs and self.warmup <= self.env.now:
                    self.log_product.flow_time[product].append(
                        (self.env.now, self.env.now - productionOrder.released)
                    )
            # Order to next resource
            else:
                process_id = productionOrder.process_finished
                next_resource = self.stores.processes_value_list[product][process_id][
                    "resource"
                ]

                yield self.stores.resource_transport[resource].get()
                yield self.stores.resource_input[next_resource].put(productionOrder)

    def _production_system(self, resource):
        last_process = None
        last_product = None

        while True:
            yield self.machine_down[resource]

            # Get order from queue
            queue_len = len(self.stores.resource_input[resource].items)
            if self.queue_order_selection is not None and queue_len > 1:
                productionOrderId = self.queue_order_selection(self, resource)
                productionOrder: ProductionOrder = yield self.stores.resource_input[
                    resource
                ].get(lambda item: item.id == productionOrderId)
            else:
                productionOrder: ProductionOrder = yield self.stores.resource_input[
                    resource
                ].get()

            yield self.stores.resource_processing[resource].put(productionOrder)

            product = productionOrder.product
            process = productionOrder.process_finished

            # Check setup
            if last_product == product and last_process == process:
                setup_time = 0
            else:
                setup_dist = self.resources_config[resource]["setup"].get(
                    "dist", "constant"
                )
                setup_params = self.resources_config[resource]["setup"].get(
                    "params", [0]
                )
                setup_time = self.rnd_process.random_number(setup_dist, setup_params)
                if self.env.now >= self.warmup and self.save_logs:
                    self.log_resource.setups[resource].append(
                        (self.env.now, setup_time)
                    )

            last_process = process

            with self.resources[resource].request() as req:
                yield req

                yield self.env.timeout(setup_time)

                process_time_dist = self.stores.processes_value_list[product][process][
                    "processing_time"
                ].get("dist")
                process_time_params = self.stores.processes_value_list[product][
                    process
                ]["processing_time"].get("params")

                order_quantity = productionOrder.quantity

                start_time = self.env.now

                for part in range(int(order_quantity)):
                    processing_time = self.rnd_process.random_number(
                        process_time_dist, process_time_params
                    )

                    yield self.env.timeout(processing_time)

                # Register data in order
                productionOrder.process_finished += 1

                end_time = self.env.now
                yield self.stores.resource_processing[resource].get()
                yield self.stores.resource_finished[resource].put(productionOrder)
                yield self.stores.resource_output[resource].put(productionOrder)
                if self.env.now >= self.warmup and self.save_logs:
                    self.log_resource.utilization[resource].append(
                        (self.env.now, round(end_time - start_time, 6))
                    )

    def _delivery_orders(self, product):
        while True:
            demandOrder: DemandOrder = yield self.stores.outbound_demand_orders[
                product
            ].get()

            delivery_mode = demandOrder.delivery_mode

            if delivery_mode == "asReady":
                self.env.process(self._delivery_as_ready(demandOrder))
            elif delivery_mode == "onDue":
                self.env.process(self._delivery_on_duedate(demandOrder))
            elif delivery_mode == "instantly":
                self.env.process(self._delivery_instantly(demandOrder))

    def _delivery_instantly(self, demand_order: DemandOrder):
        """Handle instantly delivery mode"""
        quantity = demand_order.quantity
        product = demand_order.product

        if self.stores.finished_goods[product].level >= quantity:
            yield self.stores.finished_goods[product].get(quantity)
            if self.save_logs and self.env.now > self.warmup:
                self.log_product.delivered_ontime[product].append(
                    (self.env.now, quantity)
                )
        elif self.env.now >= self.warmup and self.save_logs:
            self.log_product.lost_sales[product].append((self.env.now, quantity))

    def _delivery_as_ready(self, demand_order: DemandOrder):
        """Handle as-ready delivery mode"""
        quantity = demand_order.quantity
        product = demand_order.product

        # Remove from finished goods
        yield self.stores.finished_goods[product].get(quantity)

        # Update delivery time and log
        demand_order.delivered = self.env.now

        if self.save_logs and self.env.now >= self.warmup:
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

            if self.save_logs and self.env.now >= self.warmup:
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

    def _run_monitor(self):
        if self.print_mode in ["all", "metrics", "status"]:
            self.env.process(self._monitor())

    def _monitor(self):
        """Monitor simulation progress and print metrics"""
        start_time = time()
        yield self.env.timeout(self.monitor_warmup)

        while True:
            end_time = time()
            elapsed_time = end_time - start_time

            print("\n" + "=" * 50)
            print(f"Simulation Time: {self.env.now}")
            print(f"Elapsed Real Time: {elapsed_time:.4f} seconds")
            print("=" * 50)

            if self.print_mode == "all":
                self._print_all_metrics()
            elif self.print_mode == "metrics":
                self._print_metrics_only()
            elif self.print_mode == "status":
                self._print_status_only()

            yield self.env.timeout(self.monitor_interval)

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
            sim_elapsed_time = self.env.now - self.warmup
            resources.loc[:, "utilization"] = (
                resources.loc[:, "utilization"] / sim_elapsed_time
            )
            print("RESOURCE METRICS:")
            print(resources)

    def _print_status_only(self) -> None:
        """Print only system status"""
        snapshot = self.stores.simulation_snapshot()
        print("SYSTEM STATUS:")
        print(snapshot)
        print("\n")

    def save_history_logs(self, save_path: Path) -> None:
        """Save logs to folder"""
        save_path.mkdir(exist_ok=True, parents=True)
        products_log = self.log_product.to_dataframe()
        products_log.to_csv(save_path / "products_log.csv", index=False)
        resources_log = self.log_resource.to_dataframe()
        resources_log.to_csv(save_path / "resources_log.csv", index=False)
        general_log = self.log_general.to_dataframe()
        general_log.to_csv(save_path / "general_log.csv", index=False)

    def save_params(self, save_folder_path: Path) -> None:
        """Save params to folder"""
        save_folder_path.mkdir(exist_ok=True, parents=True)
        # Save config
        config_path = save_folder_path / "config.yaml"
        self._save_yaml(self.config, config_path)
        # Save products
        products_path = save_folder_path / "products.yaml"
        self._save_yaml(self.products_config, products_path)
        # Save resources
        resources_path = save_folder_path / "resources.yaml"
        self._save_yaml(self.resources_config, resources_path)

    def _save_json(self, data: dict, save_path: Path) -> None:
        with open(save_path, "w") as file:
            json.dump(data, file, indent=4)

    def _save_yaml(self, data: dict, save_path: Path) -> None:
        with open(save_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    def reset_simulation(self, seed) -> None:
        """Reset simulation"""
        self.seed = seed
        self._initiate_environment()

    def run_simulation(self) -> float:
        start_time = time()
        self.env.run(until=self.run_until)
        end_time = time()
        return end_time - start_time

    @abstractmethod
    def _start_custom_process(self):
        pass


if __name__ == "__main__":
    from pathlib import Path

    parser = create_simulation_parser()
    args = parser.parse_args()
    args_dict = vars(args)

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

    config = load_yaml(config_path)
    products = load_yaml(product_path)
    resources = load_yaml(resource_path)
    sim = FactorySimulation(
        config,
        resources,
        products,
        save_logs=args_dict["save_logs"],
        print_mode="metrics",
        seed=123,
    )
    elapsed_time = sim.run_simulation()
