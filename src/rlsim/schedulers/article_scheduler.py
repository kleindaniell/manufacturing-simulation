from rlsim.engine.control import DemandOrder, ProductionOrder
from rlsim.engine.scheduler import Scheduler
from rlsim.stores.dbr_stores import DBR_stores


class ArticleScheduler(Scheduler):
    def __init__(self, stores: DBR_stores):
        super().__init__(stores)
        self.stores = stores

        self.run_scheduler()

    def _scheduler(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            quantity = demandOrder.quantity

            cb_level = self.stores.constraint_buffer_level

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = self.env.now
            productionOrder.duedate = demandOrder.duedate
            productionOrder.priority = 0

            self.env.process(self.release_order(productionOrder))

            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def run_scheduler(self):
        self.env.process(self._scheduler())
