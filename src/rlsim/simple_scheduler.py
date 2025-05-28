from rlsim.control import DemandOrder, ProductionOrder
from rlsim.scheduler import Scheduler


class SimpleScheduler(Scheduler):
    def __init__(self, store, interval):
        super().__init__(store, interval)
        self.run_scheduler()

    def _scheduler(self, product):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders[
                product
            ].get()

            quantity = demandOrder.quantity

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = self.env.now
            productionOrder.duedate = demandOrder.duedate
            productionOrder.priority = 0

            self.env.process(self.release_order(productionOrder))

            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def run_scheduler(self):
        for product in self.stores.products.keys():
            self.env.process(self._scheduler(product))
