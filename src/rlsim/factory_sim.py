from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import simpy

from rlsim.engine.cli_config import create_simulation_parser
from rlsim.engine.logs import ProductLogs, ResourceLogs
from rlsim.engine.orders import DemandOrder, ProductionOrder
from rlsim.engine.stores import SimulationStores
from rlsim.engine.utils import DistributionGenerator, load_yaml


class FactorySimulation(ABC):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        save_logs: bool = True,
        seed: int = None,
        queue_order_selection: Callable = None,
    ):
        self.env = simpy.Environment()
        self.config: Dict[str, Any] = config
        self.resources_config: Dict[str, dict] = resources
        self.products_config: Dict[str, dict] = products
        self.save_logs = save_logs
        self.seed = seed
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

        self.stores = SimulationStores(
            self.env, self.products_config, self.resources_config
        )

        # self._print_vars()
        self._start_resources()
        self._start_products()
        self._run_scheduler()
        self._run_monitor()

        self.dist = DistributionGenerator(self.seed)

        self.log_product = ProductLogs(self.products_config.keys())
        self.log_resource = ResourceLogs(self.resources_config.keys())

        if self.save_logs:
            self._register_log()

    def _print_vars(self):
        for key in self.config:
            print(f"{key} - {self.config[key]}")

    def _register_log(self) -> None:
        def register_product_log():
            yield self.env.timeout(self.warmup)
            while True:
                total_wip = 0
                for product in self.products_config.keys():
                    self.log_product.fg_log[product].append(
                        (self.env.now, self.stores.finished_goods[product].level)
                    )

                    wip = self.stores.wip[product].level
                    self.log_product.wip_log[product].append((self.env.now, wip))
                    total_wip += wip

                yield self.env.timeout(self.log_interval)

        self.env.process(register_product_log())

    def _generate_demand_orders(self, product):
        product_config = self.products_config[product]

        freq_dist = product_config["demand"]["freq"].get("dist")
        freq_params = product_config["demand"]["freq"].get("params")
        quantity_dist = product_config["demand"]["quantity"].get("dist")
        quantity_params = product_config["demand"]["quantity"].get("params")
        due_dist = product_config["demand"]["duedate"].get("dist")
        due_params = product_config["demand"]["duedate"].get("params")

        while True:
            frequency = np.float32(self.dist.random_number(freq_dist, freq_params))

            quantity = round(self.dist.random_number(quantity_dist, quantity_params), 0)
            duedate = np.float32(self.dist.random_number(due_dist, due_params))

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

        last_process = len(self.products_config[product]["processes"])
        first_process = next(iter(self.products_config[product]["processes"]))
        first_resource = self.products_config[product]["processes"][first_process][
            "resource"
        ]

        productionOrder.process_total = last_process
        productionOrder.process_finished = 0

        productionOrder.released = self.env.now

        yield self.stores.wip[product].put(productionOrder.quantity)
        # Add productionOrder to first resource input
        yield self.stores.resource_input[first_resource].put(productionOrder)

        self.log_product.released[product].append(
            (self.env.now, productionOrder.quantity)
        )

    def _start_resources(self) -> None:
        self.machine_down: Dict[str, simpy.Event] = {}
        self.resources: Dict[str, simpy.Resource] = {}
        for resource in self.resources_config:

            resource_config: dict = self.resources_config.get(resource)
            quantity = resource_config.get("quantity", 1)

            self.resources[resource] = simpy.Resource(self.env, quantity)

            self.machine_down[resource] = self.env.event()
            self.machine_down[resource].succeed()
            if self.resources_config[resource].get(
                "tbf", None
            ) and self.resources_config[resource].get("ttr", None):
                self.env.process(self._breakdowns(resource))

            self.env.process(self._transportation(resource))
            self.env.process(self._production_system(resource))

    def _breakdowns(self, resource):
        while True:
            tbf_dist = self.resources_config[resource]["tbf"].get("dist", "constant")
            tbf_params = self.resources_config[resource]["tbf"].get("params", [0])
            tbf = self.dist.random_number(tbf_dist, tbf_params)

            ttr_dist = self.resources_config[resource]["ttr"].get("dist", "constant")
            ttr_params = self.resources_config[resource]["ttr"].get("params", [0])
            ttr = self.dist.random_number(ttr_dist, ttr_params)

            yield self.env.timeout(tbf)
            self.machine_down[resource] = self.env.event()
            breakdown_start = self.env.now
            yield self.env.timeout(ttr)
            self.machine_down[resource].succeed()
            breakdown_end = self.env.now

            if self.env.now >= self.warmup:
                self.log_resource.breakdowns[resource].append(
                    (breakdown_start, round(breakdown_end - breakdown_start, 6))
                )

    def _transportation(self, resource):
        while True:
            productionOrder: ProductionOrder = yield self.stores.resource_output[
                resource
            ].get()
            yield self.stores.resource_transport[resource].put(productionOrder)

            product = productionOrder.product
            if productionOrder.process_total == productionOrder.process_finished:
                productionOrder.finished = self.env.now
                yield self.stores.resource_transport[resource].get()
                yield self.stores.finished_goods[product].put(productionOrder.quantity)

                self.log_product.flow_time[product].append(
                    (self.env.now, self.env.now - productionOrder.released)
                )
                yield self.stores.wip[product].get(productionOrder.quantity)

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
                productionOrderId = self.queue_order_selection(self.stores, resource)
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
                setup_time = self.dist.random_number(setup_dist, setup_params)
                if self.env.now >= self.warmup:
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
                    processing_time = self.dist.random_number(
                        process_time_dist, process_time_params
                    )

                    yield self.env.timeout(processing_time)

                # Register data in order
                productionOrder.process_finished += 1

                end_time = self.env.now
                yield self.stores.resource_processing[resource].get()
                yield self.stores.resource_finished[resource].put(productionOrder)
                yield self.stores.resource_output[resource].put(productionOrder)
                if self.env.now >= self.warmup:
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

    def _delivery_instantly(self, demandOrder: DemandOrder):
        quantity = demandOrder.quantity
        product = demandOrder.product
        if self.stores.finished_goods[product].level >= quantity:
            yield self.stores.finished_goods[product].get(quantity)
            if self.save_logs and self.warmup < self.env.now:
                self.log_product.delivered_ontime[product].append(
                    (self.env.now, quantity)
                )

        elif self.warmup < self.env.now:
            self.log_product.lost_sales[product].append((self.env.now, quantity))

    def _delivery_as_ready(self, demandOrder: DemandOrder):
        quantity = demandOrder.quantity
        duedate = demandOrder.duedate
        product = demandOrder.product

        # remove from finished goods
        yield self.stores.finished_goods[product].get(quantity)
        # check ontime or late
        demandOrder.delivered = self.env.now
        if self.save_logs and self.warmup < self.env.now:
            if demandOrder.delivered <= duedate:
                self.log_product.delivered_ontime[product].append(
                    (self.env.now, quantity)
                )
                self.log_product.earliness[product].append(
                    (self.env.now, demandOrder.duedate - self.env.now)
                )
            else:
                self.log_product.delivered_late[product].append(
                    (self.env.now, quantity)
                )
                self.log_product.tardiness[product].append(
                    (self.env.now, self.env.now - duedate)
                )

        self.log_product.lead_time[product].append(
            (self.env.now, self.env.now - demandOrder.arived)
        )

    def _delivery_on_duedate(self, demandOrder: DemandOrder):
        def _delivey_order(demandOrder: DemandOrder):
            quantity = demandOrder.quantity
            duedate = demandOrder.duedate
            product = demandOrder.product

            # Whait for duedate
            delay = duedate - self.env.now
            self.env.timeout(delay)

            # Remove from finished goods
            yield self.stores.finished_goods[product].get(quantity)

            # Check ontime or late
            demandOrder.delivered = self.env.now
            if self.save_logs and self.warmup < self.env.now:
                if demandOrder.delivered <= duedate:
                    self.log_product.delivered_ontime[product].append(
                        (self.env.now, quantity)
                    )
                    self.log_product.earliness[product].append(
                        (self.env.now, duedate - self.env.now)
                    )
                else:
                    self.log_product.delivered_late[product].append(
                        (self.env.now, quantity)
                    )
                    self.log_product.tardiness[product].append(
                        (self.env.now, self.env.now - duedate)
                    )

                self.log_product.lead_time[product].append(
                    (self.env.now, self.env.now - demandOrder.arived)
                )

        self.env.process(_delivey_order(demandOrder))

    def _run_monitor(self):
        self.env.process(self._monitor())

    def _monitor(self):
        yield self.env.timeout(self.monitor_warmup)
        while True:
            snapshot = self.stores.simulation_snapshot()
            print(self.env.now)
            print(snapshot)
            print("\n")
            products = self.log_product.calculate_metrics()
            print(products)
            resources = self.log_resource.calculate_metrics()
            print(resources)

            yield self.env.timeout(self.monitor_interval)

    def _start_products(self):
        for product in self.products_config.keys():
            self.env.process(self._generate_demand_orders(product))
            self.env.process(self._delivery_orders(product))

    def run_simulation(self):
        self.env.run(until=self.run_until)


if __name__ == "__main__":

    config = load_yaml("src/rlsim/config/config.yaml")
    products = load_yaml("src/rlsim/config/products.yaml")
    resources = load_yaml("src/rlsim/config/resources.yaml")

    sim = FactorySimulation(config, resources, products, True, 123)
    sim.run_simulation()
