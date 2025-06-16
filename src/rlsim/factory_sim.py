from abc import ABC
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import simpy

from rlsim.engine.metrics import ProductMetrics, ResourceMetrics
from rlsim.engine.orders import DemandOrder, ProductionOrder
from rlsim.engine.stores import SimulationStores
from rlsim.engine.utils import DistributionGenerator


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
        self.resources: Dict[str, dict] = resources
        self.products: Dict[str, dict] = products
        self.save_logs = save_logs
        self.seed = seed
        self.queue_order_selection = queue_order_selection

        self.stores = SimulationStores(self.env, self.products, self.resources)

        self._start_resources()
        self._start_products()
        self._run_scheduler()

        self.dist = DistributionGenerator(self.seed)

        self.log_products = ProductMetrics(self.products.keys())
        self.log_resources = ResourceMetrics(self.resources.keys())

        self.total_wip_log: List[Tuple[float, float]] = []

        if self.save_logs:
            self._register_log()

    def _register_log(self) -> None:
        def register_product_log():
            yield self.env.timeout(self.warmup)
            while True:
                total_wip = 0
                for product in self.products.keys():
                    self.log_products.fg_log[product].append(
                        (self.env.now, self.stores.finished_goods[product].level)
                    )

                    wip = self.stores.wip[product].level
                    self.log_products.wip_log[product].append((self.env.now, wip))
                    total_wip += wip

                self.total_wip_log.append((self.env.now, total_wip))

                yield self.env.timeout(self.log_interval)

        self.env.process(register_product_log())

    def _generate_demand_orders(self, product):
        product_config = self.products[product]

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

        last_process = len(self.products[product]["processes"])
        first_process = next(iter(self.products[product]["processes"]))
        first_resource = self.products[product]["processes"][first_process]["resource"]

        productionOrder.process_total = last_process
        productionOrder.process_finished = 0

        productionOrder.released = self.env.now

        yield self.stores.wip[product].put(productionOrder.quantity)
        # Add productionOrder to first resource input
        yield self.stores.resource_input[first_resource].put(productionOrder)

        self.log_products.released[product].append(
            (self.env.now, productionOrder.quantity)
        )

    def _start_resources(self) -> None:
        self.resources: Dict[str, simpy.Resource] = {}
        self.machine_down: Dict[str, simpy.Event] = {}

        for resource in self.resources:
            resource_config: dict = self.resources.get(resource)
            quantity = resource_config.get("quantity", 1)

            self.resources[resource] = simpy.Resource(self.env, quantity)

            self.machine_down[resource] = self.env.event()
            self.machine_down[resource].succeed()

            if self.resources[resource].get("tbf", None) and self.resources[
                resource
            ].get("ttr", None):
                self.env.process(self._breakdowns(resource))

            self.env.process(self._transportation(resource))
            self.env.process(self._production_system(resource))

    def _breakdowns(self, resource):
        try:
            while True:
                tbf_dist = self.resources[resource]["tbf"].get("dist", "constant")
                tbf_params = self.resources[resource]["tbf"].get("params", [0])
                tbf = self.dist.random_number(tbf_dist, tbf_params)

                ttr_dist = self.resources[resource]["ttr"].get("dist", "constant")
                ttr_params = self.resources[resource]["ttr"].get("params", [0])
                ttr = self.dist.random_number(ttr_dist, ttr_params)

                yield self.env.timeout(tbf)
                self.machine_down[resource] = self.env.event()
                breakdown_start = self.env.now
                yield self.env.timeout(ttr)
                self.machine_down[resource].succeed()
                breakdown_end = self.env.now

                if self.env.now >= self.warmup:
                    self.log_resources.breakdowns[resource].append(
                        (breakdown_start, round(breakdown_end - breakdown_start, 6))
                    )

        except ValueError:
            pass

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

                self.log_products.flow_time[product].append(
                    (self.env.now, self.env.now - productionOrder.released)
                )
                yield self.wip[product].get(productionOrder.quantity)

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
                setup_dist = self.resources[resource]["setup"].get("dist", "constant")
                setup_params = self.resources[resource]["setup"].get("params", [0])
                setup_time = self.dist.random_number(setup_dist, setup_params)
                if self.env.now >= self.warmup:
                    self.log_resources.setups[resource].append(
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
                    self.log_resources.utilization[resource].append(
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
            if not self.stores.training and self.stores.warmup < self.env.now:
                self.log_products.delivered_ontime[product].append(
                    (self.env.now, quantity)
                )

        elif self.stores.warmup < self.env.now:
            self.log_products.lost_sales[product].append((self.env.now, quantity))

    def _delivery_as_ready(self, demandOrder: DemandOrder):
        quantity = demandOrder.quantity
        duedate = demandOrder.duedate
        product = demandOrder.product

        # remove from finished goods
        yield self.stores.finished_goods[product].get(quantity)
        # check ontime or late
        demandOrder.delivered = self.env.now
        if not self.stores.training and self.stores.warmup < self.env.now:
            if demandOrder.delivered <= duedate:
                self.log_products.delivered_ontime[product].append(
                    (self.env.now, quantity)
                )
                self.log_products.earliness[product].append(
                    (self.env.now, demandOrder.duedate - self.env.now)
                )
            else:
                self.log_products.delivered_late[product].append(
                    (self.env.now, quantity)
                )
                self.log_products.tardiness[product].append(
                    (self.env.now, self.env.now - duedate)
                )

        self.log_products.lead_time[product].append(
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
            if not self.stores.training and self.stores.warmup < self.env.now:
                if demandOrder.delivered <= duedate:
                    self.log_products.delivered_ontime[product].append(
                        (self.env.now, quantity)
                    )
                    self.log_products.earliness[product].append(
                        (self.env.now, duedate - self.env.now)
                    )
                else:
                    self.log_products.delivered_late[product].append(
                        (self.env.now, quantity)
                    )
                    self.log_products.tardiness[product].append(
                        (self.env.now, self.env.now - duedate)
                    )

                self.log_products.lead_time[product].append(
                    (self.env.now, self.env.now - demandOrder.arived)
                )

        self.env.process(_delivey_order(demandOrder))

    def _start_products(self):
        for product in self.products.keys():
            self.env.process(self._generate_demand_orders(product))
            self.env.process(self._delivery_orders(product))

    def run_simulation(self):
        print(self.run_until)
        self.env.run(until=self.run_until)
