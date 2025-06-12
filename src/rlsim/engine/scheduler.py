from abc import ABC, abstractmethod

import simpy

from rlsim.engine.control import ProductionOrder, Stores


class Scheduler(ABC):
    def __init__(self, store: Stores):
        self.stores = store
        self.env: simpy.Environment = store.env

    def release_order(self, productionOrder: ProductionOrder):
        product = productionOrder.product

        last_process = len(self.stores.products[product]["processes"])
        first_process = next(iter(self.stores.products[product]["processes"]))
        first_resource = self.stores.products[product]["processes"][first_process][
            "resource"
        ]

        productionOrder.process_total = last_process
        productionOrder.process_finished = 0

        productionOrder.released = self.env.now

        yield self.stores.wip[product].put(productionOrder.quantity)
        # Add productionOrder to first resource input
        yield self.stores.resource_input[first_resource].put(productionOrder)

        self.stores.log_products.released[product].append(
            (self.env.now, productionOrder.quantity)
        )

    @abstractmethod
    def _scheduler(self):
        pass

    @abstractmethod
    def run_scheduler(self):
        pass
