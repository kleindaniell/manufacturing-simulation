from rlsim.engine.control import DemandOrder, ProductionOrder
from rlsim.engine.scheduler import Scheduler
from rlsim.stores.article_stores import DBR_stores


class ArticleScheduler(Scheduler):
    def __init__(
        self, stores: DBR_stores, constraint_buffer_size, shipping_buffer_size
    ):
        super().__init__(stores)
        self.stores = stores
        self.stores.constraint_buffer = constraint_buffer_size
        self.stores.shipping_buffer = shipping_buffer_size

        self.run_scheduler()

    def _scheduler(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            quantity = demandOrder.quantity

            ccr_processing_time = sum(
                [
                    process["processing_time"]["params"][0]
                    for process in self.stores.processes_value_list[product]
                    if process["resource"] == self.stores.contraint_resource
                ]
            )

            if ccr_processing_time > 0:
                # schedule = (
                #     duedate
                #     - (self.stores.shipping_buffer + ccr_processing_time)
                #     - self.stores.constraint_buffer
                # )

                buffer_diff = (
                    self.stores.constraint_buffer_level - self.stores.constraint_buffer
                )
                schedule = (
                    self.env.now + buffer_diff if buffer_diff > 0 else self.env.now
                )

            else:
                schedule = self.env.now

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = schedule
            productionOrder.duedate = demandOrder.duedate
            productionOrder.priority = 0

            self.env.process(self.process_order(productionOrder, ccr_processing_time))

            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def process_order(
        self, productionOrder: ProductionOrder, ccr_processing_time: float
    ):

        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)

        self.stores.constraint_buffer_level += ccr_processing_time

        self.env.process(self.release_order(productionOrder))

    def run_scheduler(self):
        self.env.process(self._scheduler())
