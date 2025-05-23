import simpy

from rlsim.control import ProductionOrder, Stores, DemandOrder
from rlsim.scheduler import Scheduler


class SimpleScheduler(Scheduler):
    def __init__(self, store, interval):
        super().__init__(store, interval)
        self.run_scheduler()

    # def _scheduler(self):
    #     while True:
    #         for product in self.stores.products.keys():
    #             productionOrder = ProductionOrder(product=product, quantity=1)
    #             self.env.process(self.release_order(productionOrder))

    #         yield self.env.timeout(self.interval)

    def _scheduler(self, product):

        while True:

            demandOrder: DemandOrder = yield self.stores.demand_orders[product].get()

            quantity = demandOrder.quantity

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = self.env.now
            productionOrder.priority = 0
            self.env.process(self.release_order(productionOrder))

    def run_scheduler(self):

        for product in self.stores.products.keys():
            self.env.process(self._scheduler(product))
