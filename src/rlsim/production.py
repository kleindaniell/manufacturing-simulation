import simpy

from rlsim.control import ProductionOrder, Stores
from rlsim.utils import random_number


class Production:
    def __init__(self, stores: Stores, warmup: int = 0, order_selection_fn=None):
        self.stores: Stores = stores
        self.env: simpy.Environment = stores.env
        self.warmup = warmup
        self.order_selection_fn = order_selection_fn
        self._create_resources()

    def _create_resources(self) -> None:
        self.resources = {}
        self.machine_down = {}
        self.wait_queue = {}

        for resource in self.stores.resources:
            resource_config = self.stores.resources.get(resource)
            quantity = resource_config.get("quantity", 1)

            self.resources[resource] = simpy.Resource(self.env, quantity)

            self.machine_down[resource] = self.env.event()
            self.machine_down[resource].succeed()

            # self.wait_queue[resource] = self.env.event()
            # self.wait_queue[resource].succeed()

            self.env.process(self._breakdowns(resource))
            self.env.process(self._transportation(resource))
            self.env.process(self._production_system(resource))

    def _breakdowns(self, resource):
        try:
            while True:
                tbf_dist = self.stores.resources[resource]["tbf"].get(
                    "dist", "constant"
                )
                tbf_params = self.stores.resources[resource]["tbf"].get("params", [0])
                tbf = random_number(tbf_dist, tbf_params)

                ttr_dist = self.stores.resources[resource]["ttr"].get(
                    "dist", "constant"
                )
                ttr_params = self.stores.resources[resource]["ttr"].get("params", [0])
                ttr = random_number(ttr_dist, ttr_params)

                yield self.env.timeout(tbf)
                self.machine_down[resource] = self.env.event()
                breakdown_start = self.env.now
                yield self.env.timeout(ttr)
                self.machine_down[resource].succeed()
                breakdown_end = self.env.now

                if self.env.now >= self.warmup:
                    self.stores.resource_breakdowns[resource].append(
                        {"start": breakdown_start, "end": breakdown_end}
                    )

        except:
            pass

    def _transportation(self, resource):
        while True:
            productionOrder = yield self.stores.resource_output[resource].get()
            yield self.stores.resource_transport[resource].put(productionOrder)

            product = productionOrder.product
            if productionOrder.process_total == productionOrder.process_finished:
                productionOrder.finished = self.env.now
                yield self.stores.resource_transport[resource].get()
                yield self.stores.finished_goods[product].put(productionOrder.quantity)

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

            productionOrder: ProductionOrder = yield self.stores.resource_input[
                resource
            ].get()

            yield self.stores.resource_input[resource].put(productionOrder)

            # Get order from queue
            if self.order_selection_fn is not None:
                productionOrderId = self.order_selection_fn(self.stores, resource)
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
                setup_dist = self.stores.resources[resource]["setup"].get(
                    "dist", "constant"
                )
                setup_params = self.stores.resources[resource]["setup"].get(
                    "params", [0]
                )
                setup_time = random_number(setup_dist, setup_params)
                if self.env.now >= self.warmup:
                    self.setups_cout[resource] += 1
                    self.setups_time[resource] += setup_time

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

                for part in range(order_quantity):
                    processing_time = random_number(
                        process_time_dist, process_time_params
                    )

                    yield self.env.timeout(processing_time)

                # Register data in order
                productionOrder.process_finished += 1

                end_time = self.env.now
                yield self.stores.resource_processing[resource].get()
                yield self.stores.resource_output[resource].put(productionOrder)
                if self.env.now > self.warmup:
                    self.stores.resource_utilization[resource] += round(
                        end_time - start_time, 8
                    )
