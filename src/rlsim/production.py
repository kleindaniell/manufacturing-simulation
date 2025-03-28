import simpy
from rlsim.control import Stores, ProductionOrder
from rlsim.utils import random_number


class Production:
    def __init__(self, stores: Stores, warmup: int = 0):
        self.stores: Stores = stores
        self.env: simpy.Environment = stores.env
        self.warmup = warmup
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
                        [breakdown_start, breakdown_end]
                    )

        except:
            pass

    def _transportation(self, resource):
        while True:
            order = yield self.stores.resource_output[resource].get()
            yield self.stores.resource_transport[resource].put(order)

            product = order["product"]
            if order["process_total"] == order["process_finished"]:
                order["finished"] = self.env.now
                yield self.stores.resource_transport[resource].get()
                yield self.stores.finished_orders[product].put(order)

            else:
                process_id = order["process_finished"]
                next_resource = self.stores.processes_value_list[product][process_id][
                    "resource"
                ]
                print(
                    f"transport: {self.env.now:.3f} {product} {resource}->{next_resource}"
                )
                yield self.stores.resource_transport[resource].get()
                yield self.stores.resource_input[next_resource].put(order)

    def _production_system(self, resource):
        last_process = None
        last_product = None

        while True:
            yield self.machine_down[resource]

            # Get order from queue
            # order = yield self._get_order_resource_queue(resource, "fifo")
            order = yield self.stores.resource_input[resource].get(lambda item: True)
            yield self.stores.resource_processing[resource].put(order)

            product = order["product"]
            process = order["process_finished"]

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
                # if self.env.now >= self.warmup:
                # self.setups_cout[resource] += 1
                # self.setups_time[resource] += setup_time

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

                order_quantity = order.get("quantity")

                start_time = self.env.now

                for part in range(order_quantity):
                    processing_time = random_number(
                        process_time_dist, process_time_params
                    )

                    yield self.env.timeout(processing_time)

                # Register data in order
                order["process_finished"] += 1

                end_time = self.env.now
                yield self.stores.resource_processing[resource].get()
                yield self.stores.resource_output[resource].put(order)
                # if self.env.now > self.warmup:
                #     self.utilization[resource] += round(end_time - start_time, 8)

    def _get_order_resource_queue(self, resource, method):
        self.wait_queue[resource] = self.env.event()
        match method:
            case "fifo":
                order = yield self.stores.resource_input[resource].get()

                print(order)
            case "toc_penetration":
                # TODO: implement filter for toc penetration method
                order = yield self.stores.resource_input[resource].get()

        # order = yield self.stores.resource_input[resource].get()

        return order
