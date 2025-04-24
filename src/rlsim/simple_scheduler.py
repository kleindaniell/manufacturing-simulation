import simpy

from rlsim.control import ProductionOrder, Stores
from rlsim.scheduler import Scheduler


class SimpleScheduler(Scheduler):
    def __init__(self, store, interval):
        super().__init__(store, interval)
        self.run_scheduler()

    def _scheduler(self):
        while True:
            for product in self.stores.products.keys():
                productionOrder = ProductionOrder(product=product, quantity=1)
                self.env.process(self.release_order(productionOrder))

            yield self.env.timeout(self.interval)
