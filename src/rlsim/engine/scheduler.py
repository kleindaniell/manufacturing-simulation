from abc import ABC

import simpy

from rlsim.engine.control import ProductionOrder, Stores, DemandOrder


class Scheduler(ABC):
    def __init__(self, store: Stores, run_scheduler: bool = True):
        self.stores = store
        self.env: simpy.Environment = store.env

        if run_scheduler:
            self.run_scheduler()

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

    def scheduler(self, product):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()

            quantity = demandOrder.quantity

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = self.env.now
            productionOrder.duedate = demandOrder.duedate
            productionOrder.priority = 0

            self.env.process(self.release_order(productionOrder))

            yield self.stores.outbound_demand_orders[product].put(demandOrder)
            # print(productionOrder)

    def run_scheduler(self):
        for product in self.stores.products.keys():
            self.env.process(self.scheduler(product))
