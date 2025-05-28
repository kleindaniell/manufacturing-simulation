from abc import ABC, abstractmethod

import simpy

from rlsim.control import ProductionOrder, Stores


class Scheduler(ABC):
    def __init__(self, store: Stores, interval: int):
        self.stores = store
        self.env: simpy.Environment = store.env
        self.interval = interval

    def release_order(self, productionOrder: ProductionOrder):
        product = productionOrder.product

        last_process = len(self.stores.products[product]["processes"])
        first_process = next(iter(self.stores.products[product]["processes"]))
        first_resource = self.stores.products[product]["processes"][first_process][
            "resource"
        ]

        productionOrder.process_total = last_process
        productionOrder.process_finished = 0

        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)

        productionOrder.released = self.env.now

        # Add productionOrder to first resource input
        yield self.stores.resource_input[first_resource].put(productionOrder)

    @abstractmethod
    def _scheduler(self):
        pass

    @abstractmethod
    def run_scheduler(self):
        pass
