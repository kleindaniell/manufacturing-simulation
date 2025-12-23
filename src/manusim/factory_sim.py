from abc import ABC
from pathlib import Path
from time import time
from typing import Any, Dict, Literal
import pandas as pd

import numpy as np
import simpy
from manusim.engine.logs import Logger
from manusim.engine.orders import DemandOrder, ProductionOrder
from manusim.engine.stores import SimulationStores
from manusim.engine.utils import DistributionGenerator
from manusim.metrics import MetricGeneral, MetricProducts, MetricResources


class FactorySimulation(ABC):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        print_mode: Literal["status", "metrics", "all"] = "metrics",
        seed: int = None,
    ):
        self.config: Dict[str, Any] = config
        self.resources_config: Dict[str, dict] = resources
        self.products_config: Dict[str, dict] = products
        self.seed = seed
        self.print_mode = print_mode
        self.log_save_path = self.config.get("log_save_path", None)
        self.memory_size = self.config.get("memory_size", 10000)

        # Load config
        self.run_until = self.config.get("run_until", None)
        if self.run_until is None:
            raise ValueError("run_until must be specified")

        self.warmup = self.config.get("warmup", 0)

        self.monitor_warmup = self.config.get("monitor_warmup", self.warmup)
        self.monitor_interval = self.config.get(
            "monitor_interval", int(self.run_until / 4)
        )

        self.delivery_mode = self.config.get("delivery_mode", "asReady")

        self.log_queues = self.config.get("log_queues", False)

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
        self._warmup_period()
        self._start_resources()
        self._start_products()
        self._run_monitor()
        self._start_custom_process()
        self._run_scheduler()

        self._initiate_custom_env()

    def _initiate_custom_env(self):
        pass

    def _warmup_period(self):
        self.warmup_finished = False

        def warmup():
            yield self.env.timeout(self.warmup)
            self.warmup_finished = True

        self.env.process(warmup())

    def _init_random_generators(self) -> None:
        """Initialize random number generators for different purposes"""
        self.rnd_product = DistributionGenerator(self.seed)
        self.rnd_process = DistributionGenerator(self.seed)
        self.rnd_breakdown = DistributionGenerator(self.seed)

    def _init_loggint(self) -> None:
        """Initialize logging components"""

        self.logs = Logger(logs_save_path=self.log_save_path, mem_size=self.memory_size)
        for product_metric in MetricProducts:
            self.logs.create_log(product_metric.name, self.products_config.keys())

        for resource_metric in MetricResources:
            self.logs.create_log(resource_metric.name, self.resources_config.keys())

        for general_metric in MetricGeneral:
            self.logs.create_log(general_metric.name, ["general"])

        self._create_custom_logs()

        self._register_custom_logs()

    def _create_custom_logs(self):
        """Create custom logs if needed"""
        return

    def _register_custom_logs(self):
        """Register custom logs created"""
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
            yield self.stores.inbound_demand_orders.put(demandOrder)

    def _run_scheduler(self):
        self.env.process(self.scheduler())

    def scheduler(self):
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
        if self.warmup_finished:
            self.logs.log(
                variable=MetricProducts.released.name,
                key=product,
                value=(self.env.now, productionOrder.quantity),
            )
            # Log Wip
            wip = self.stores.wip[product].level
            self.logs.log(
                variable=MetricProducts.wip.name,
                key=product,
                value=(self.env.now, wip),
            )
            total_wip = sum(store.level for store in self.stores.wip.values())
            self.logs.log(
                variable=MetricGeneral.wip_general.name,
                key="general",
                value=(self.env.now, total_wip),
            )
            # Log resource queue
            if self.log_queues:
                current_log = self.logs.get_log(
                    MetricResources.queue.name, first_resource
                )
                if len(current_log) == 0:
                    queue_qnt = sum(
                        [
                            x.quantity
                            for x in self.stores.resource_input[first_resource].items
                        ]
                    )
                else:
                    queue_qnt = current_log[-1][1]
                    queue_qnt += productionOrder.quantity
                # log
                self.logs.log(
                    variable=MetricResources.queue.name,
                    key=first_resource,
                    value=(self.env.now, queue_qnt),
                )

    def _start_resources(self) -> None:
        self.resources: Dict[str, simpy.Resource] = {}
        self.tbf: Dict[str, float] = {}
        self.utilization_to_fail: Dict[str, float] = {}
        for resource in self.resources_config:
            resource_config: dict = self.resources_config.get(resource)
            quantity = resource_config.get("quantity", 1)

            # Create Resource
            self.resources[resource] = simpy.Resource(self.env, quantity)

            # Initialize breakdown state
            resource_config = self.resources_config[resource]
            tbf_config = resource_config.get("tbf", None)
            self.tbf[resource] = (
                self._get_breakdown_time(tbf_config) if tbf_config else float("inf")
            )
            self.utilization_to_fail[resource] = 0

            # Start resource process
            self.env.process(self._transportation(resource))
            self.env.process(self._production_system(resource))

    def _breakdowns(self, resource):
        """Handle resources breakdowns"""
        resource_config = self.resources_config[resource]

        # Start breakdown
        breakdown_start = self.env.now

        # Repair time
        ttr = self._get_breakdown_time(resource_config["ttr"])
        yield self.env.timeout(ttr)
        breakdown_end = self.env.now

        # Log breakdown
        if self.warmup_finished:
            self.logs.log(
                variable=MetricResources.breakdown.name,
                key=resource,
                value=(breakdown_start, round(breakdown_end - breakdown_start, 6)),
            )

        # Set new time between fail and reset utilization
        self.tbf[resource] = self._get_breakdown_time(resource_config["tbf"])
        self.utilization_to_fail[resource] = 0

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

                if self.warmup_finished:
                    # Log flow time
                    self.logs.log(
                        variable=MetricProducts.flowTime.name,
                        key=product,
                        value=(self.env.now, self.env.now - productionOrder.released),
                    )
                    
                    # Log Wip
                    wip = self.stores.wip[product].level
                    self.logs.log(
                        variable=MetricProducts.wip.name,
                        key=product,
                        value=(self.env.now, wip),
                    )
                    total_wip = sum(store.level for store in self.stores.wip.values())
                    self.logs.log(
                        variable=MetricGeneral.wip_general.name,
                        key="general",
                        value=(self.env.now, total_wip),
                    )

                    # Log FG
                    fg_level = self.stores.finished_goods[product].level
                    self.logs.log(
                        variable=MetricProducts.finishedGoods.name,
                        key=product,
                        value=(self.env.now, fg_level),
                    )
                    total_fg = sum(
                        store.level for store in self.stores.finished_goods.values()
                    )
                    self.logs.log(
                        variable=MetricGeneral.finishedGoods_general.name,
                        key="general",
                        value=(self.env.now, total_fg),
                    )

            # Order to next resource
            else:
                process_id = productionOrder.process_finished
                next_resource = self.stores.processes_value_list[product][process_id][
                    "resource"
                ]

                yield self.stores.resource_transport[resource].get()
                yield self.stores.resource_input[next_resource].put(productionOrder)
                self._custom_order_in_resource_input(productionOrder, next_resource)

                # Log queues
                if self.warmup_finished and self.log_queues:
                    current_queue = self.logs.get_log(
                        MetricResources.queue.name, next_resource
                    )
                    if len(current_queue) == 0:
                        queue_qnt = sum(
                            [
                                x.quantity
                                for x in self.stores.resource_input[next_resource].items
                            ]
                        )
                    else:
                        queue_qnt = current_queue[-1][1]
                        queue_qnt += productionOrder.quantity
                    # Log queue
                    self.logs.log(
                        variable=MetricResources.queue.name,
                        key=next_resource,
                        value=(self.env.now, queue_qnt),
                    )

    def _custom_order_in_resource_input(self, productionOrder, resource):
        pass

    def _custom_order_out_resource_input(self, productionOrder, resource):
        pass

    def _custom_part_processed(self, product, resource, process):
        pass

    def _custom_order_processed(self, productionOrder, resource):
        pass

    def _production_system(self, resource):
        last_process = None
        last_product = None
        while True:
            if self.utilization_to_fail[resource] >= self.tbf[resource]:
                yield from self._breakdowns(resource)

            # Get order from queue
            productionOrder: ProductionOrder = yield from self.order_selection(resource)
            self._custom_order_out_resource_input(productionOrder, resource)

            if self.warmup_finished and self.log_queues:
                current_queue = self.logs.get_log(MetricResources.queue.name, resource)
                if len(current_queue) == 0:
                    queue_qnt = sum(
                        [x.quantity for x in self.stores.resource_input[resource].items]
                    )
                else:
                    queue_qnt = current_queue[-1][1]
                    queue_qnt += productionOrder.quantity
                # Log queue
                self.logs.log(
                    variable=MetricResources.queue.name,
                    key=resource,
                    value=(self.env.now, queue_qnt),
                )

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
                if self.warmup_finished:
                    self.logs.log(
                        variable=MetricResources.setup.name,
                        key=resource,
                        value=(self.env.now, setup_time),
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

                for _ in range(int(order_quantity)):
                    processing_time = self.rnd_process.random_number(
                        process_time_dist, process_time_params
                    )

                    yield self.env.timeout(processing_time)
                    self._custom_part_processed(product, resource, process)

                end_time = self.env.now
                utilization = round(end_time - start_time, 6)

                # Update utilization to fail
                self.utilization_to_fail[resource] += utilization

                # Register data in order
                productionOrder.process_finished += 1

                yield self.stores.resource_processing[resource].get()
                yield self.stores.resource_output[resource].put(productionOrder)

                self._custom_order_processed(productionOrder, resource)

                if self.warmup_finished:
                    self.logs.log(
                        variable=MetricResources.utilization.name,
                        key=resource,
                        value=(self.env.now, utilization),
                    )

    def order_selection(self, resource):
        productionOrder = yield self.stores.resource_input[resource].get()
        return productionOrder

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
            demand_order.delivered = self.env.now

            # Placeholder for custom action when fg is reduced
            self._custom_fg_reduced(product)

        # Log orders
        if self.warmup_finished:
            fg_level = self.stores.finished_goods[product].level
            self.logs.log(
                variable=MetricProducts.finishedGoods.name,
                key=product,
                value=(self.env.now, fg_level),
            )
            total_fg = sum(
                store.level for store in self.stores.finished_goods.values()
            )
            self.logs.log(
                variable=MetricGeneral.finishedGoods_general.name,
                key="general",
                value=(self.env.now, total_fg),
            )

            self._log_delivery_performance(demand_order)
            self._log_lead_time(demand_order)

    def _delivery_as_ready(self, demand_order: DemandOrder):
        """Handle as-ready delivery mode"""
        quantity = demand_order.quantity
        product = demand_order.product

        # Remove from finished goods
        yield self.stores.finished_goods[product].get(quantity)

        # Placeholder for custom action when fg is reduced
        self._custom_fg_reduced(product)

        # Update delivery time and log
        demand_order.delivered = self.env.now

        if self.warmup_finished:
            fg_level = self.stores.finished_goods[product].level
            self.logs.log(
                variable=MetricProducts.finishedGoods.name,
                key=product,
                value=(self.env.now, fg_level),
            )
            total_fg = sum(
                store.level for store in self.stores.finished_goods.values()
            )
            self.logs.log(
                variable=MetricGeneral.finishedGoods_general.name,
                key="general",
                value=(self.env.now, total_fg),
            )

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

            # Placeholder for custom action when fg is reduced
            self._custom_fg_reduced(product)

            # Update delivery time and log
            demand_order.delivered = self.env.now

            if self.warmup_finished:
                fg_level = self.stores.finished_goods[product].level
                self.logs.log(
                    variable=MetricProducts.finishedGoods.name,
                    key=product,
                    value=(self.env.now, fg_level),
                )
                total_fg = sum(
                    store.level for store in self.stores.finished_goods.values()
                )
                self.logs.log(
                    variable=MetricGeneral.finishedGoods_general.name,
                    key="general",
                    value=(self.env.now, total_fg),
                )

                self._log_delivery_performance(demand_order)
                self._log_lead_time(demand_order)

        self.env.process(_delivery_order(demand_order))

    def _custom_fg_reduced(self, product):
        pass

    def _log_delivery_performance(self, demand_order: DemandOrder) -> None:
        """Log delivery performance metrics"""
        product = demand_order.product
        quantity = demand_order.quantity
        delivered = demand_order.delivered
        duedate = demand_order.duedate
        # Lost Sales
        if not delivered:
            self.logs.log(
                variable=MetricProducts.lostSales.name,
                key=product,
                value=(self.env.now, quantity),
            )
        # Delivered ontime
        elif delivered <= duedate:
            self.logs.log(
                variable=MetricProducts.deliveredOntime.name,
                key=product,
                value=(self.env.now, quantity),
            )
            earliness = demand_order.duedate - self.env.now
            self.logs.log(
                variable=MetricProducts.earliness.name,
                key=product,
                value=(self.env.now, earliness),
            )
        # Delivered late
        elif delivered > duedate:
            self.logs.log(
                variable=MetricProducts.deliveredLate.name,
                key=product,
                value=(self.env.now, quantity),
            )
            tardiness = self.env.now - demand_order.duedate
            self.logs.log(
                variable=MetricProducts.tardiness.name,
                key=product,
                value=(self.env.now, tardiness),
            )

    def _log_lead_time(self, demand_order: DemandOrder) -> None:
        """Log lead time for the order"""
        product = demand_order.product
        lead_time = self.env.now - demand_order.arived
        self.logs.log(
            variable=MetricProducts.leadTime.name,
            key=product,
            value=(self.env.now, lead_time),
        )

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

            print("\n" + "-" * 50)
            print("-" * 50)
            print(f"Simulation Time: {self.env.now}")
            print(f"Elapsed Real Time: {elapsed_time:.4f} seconds")
            print("-" * 50)

            if self.print_mode == "all":
                self._print_all_metrics()
            elif self.print_mode == "metrics":
                self._print_metrics()
            elif self.print_mode == "status":
                self._print_status()

            self.print_custom_metrics()

            yield self.env.timeout(self.monitor_interval)

    def _print_all_metrics(self) -> None:
        """Print all available metrics and status"""
        self._print_status()
        self._print_metrics()

    def _print_status(self) -> None:
        """Print only system status"""

        print("SYSTEM STATUS:")
        snapshot = self.stores.simulation_snapshot()
        if not snapshot.empty:
            print(snapshot)
        else:
            print("Empty metrics")
        print("-" * 50)

    def _print_metrics(self) -> None:
        """Print only performance metrics"""

        print("PRODUCT METRICS:")
        products = self.products_metrics()
        if not products.empty:
            print(products)
        else:
            print("Empty products metrics")

        print("-" * 50)
        print("RESOURCE METRICS:")
        resources = self.resources_metrics()
        if not resources.empty:

            print(resources)
        else:
            print("Empty resource metrics")
        print("-" * 50)

        print("GENERAL METRICS:")
        general = self.general_metrics()
        if not general.empty:

            print(general)
        else:
            print("Empty general metrics")
        print("-" * 50)

    def print_custom_metrics(self):
        pass

    def products_metrics(self, saved_logs=False) -> pd.DataFrame:
        df_list = []
        sum_metrics = [
            MetricProducts.deliveredOntime.name,
            MetricProducts.deliveredLate.name,
            MetricProducts.lostSales.name,
        ]

        for metric in MetricProducts:

            metric_df = self.logs.get_variable_logs(
                variable=metric.name, saved_logs=saved_logs
            )

            if metric.name in sum_metrics:
                metric_df = metric_df.pivot_table(
                    values="value", index="key", columns="variable", aggfunc="sum"
                )
            else:
                metric_df = metric_df.pivot_table(
                    values="value", index="key", columns="variable", aggfunc="mean"
                )

            df_list.append(metric_df)
        return pd.concat(df_list, axis=1)

    def resources_metrics(self, saved_logs=False) -> pd.DataFrame:
        df_list = []
        for metric in MetricResources:
            metric_df = self.logs.get_variable_logs(
                variable=metric.name, saved_logs=saved_logs
            )

            if metric.name == "utilization":
                metric_df = metric_df.pivot_table(
                    values="value", index="key", columns="variable", aggfunc="sum"
                )
                metric_df = metric_df / (self.env.now - self.warmup)
            else:
                metric_df = metric_df.pivot_table(
                    values="value", index="key", columns="variable", aggfunc="mean"
                )
                metric_df = metric_df / (self.env.now - self.warmup)

            df_list.append(metric_df)
        return pd.concat(df_list, axis=1)

    def general_metrics(self, saved_logs=False) -> pd.DataFrame:
        df_list = []
        for metric in MetricGeneral:
            metric_df = self.logs.get_variable_logs(
                variable=metric.name, saved_logs=saved_logs
            )

            metric_df = metric_df.pivot_table(
                values="value", index="key", columns="variable", aggfunc="mean"
            )

            df_list.append(metric_df)
        return pd.concat(df_list, axis=1)

    def save_metrics(self, save_path: Path, saved_logs=False) -> None:
        products_df = self.products_metrics(saved_logs=saved_logs)
        resources_df = self.resources_metrics(saved_logs=saved_logs)
        general_df = self.general_metrics(saved_logs=saved_logs)

        save_path.mkdir(exist_ok=True, parents=True)
        products_df.to_csv(save_path / "metrics_products.csv")
        resources_df.to_csv(save_path / "metrics_resources.csv")
        general_df.to_csv(save_path / "metrics_general.csv")

    def save_custom_metrics(self, save_path: Path, saved_logs=False) -> None:
        pass

    def reset_simulation(self, seed, log_save_path) -> None:
        """Reset simulation"""
        self.seed = seed
        self.log_save_path = log_save_path
        self._initiate_environment()

    def run_simulation(self) -> float:
        start_time = time()
        self.env.run(until=self.run_until)
        end_time = time()
        return end_time - start_time

    def _start_custom_process(self):
        pass
