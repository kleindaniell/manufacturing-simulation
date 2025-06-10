from typing import Tuple, List

from rlsim.engine.control import DemandOrder, ProductionOrder
from rlsim.engine.scheduler import Scheduler
from rlsim.stores.dbr_mta_store import DBR_stores


class DBR_MTA(Scheduler):
    def __init__(
        self,
        stores: DBR_stores,
        schedule_interval,
        constraint_buffer_size,
    ):
        super().__init__(stores)
        self.stores = stores
        self.stores.constraint_buffer = constraint_buffer_size
        self.schedule_interval = schedule_interval

        self.run_scheduler()
        self.env.process(self._process_demandOders())

    def run_scheduler(self):
        self.env.process(self._scheduler(self.schedule_interval))

    def _scheduler(self, interval):
        ccr_setup_time_params = self.stores.resources[
            self.stores.contraint_resource
        ].get("setup", {"params": None})
        ccr_setup_time = ccr_setup_time_params.get("params", [0])[0]

        last_sold = {}
        for product in self.stores.products.keys():
            last_sold[product] = 0

        while True:

            orders: List[Tuple[ProductionOrder, float, float]] = []
            for product in self.stores.products.keys():

                replenishment = self.stores.sold_product[product] - last_sold[product]
                last_sold[product] = self.stores.sold_product[product]

                ccr_processing_time = sum(
                    [
                        process["processing_time"]["params"][0]
                        for process in self.stores.processes_value_list[product]
                        if process["resource"] == self.stores.contraint_resource
                    ]
                )

                penetration = self.calculate_penetration(product)
                print(f"{product} - {ccr_processing_time} - {replenishment}")
                orders.append(
                    (
                        # Production order
                        ProductionOrder(
                            product=product,
                            quantity=replenishment,
                            priority=round(
                                penetration / self.stores.shipping_buffer[product], 3
                            ),
                        ),
                        # ccr processin time
                        ccr_processing_time,
                        # Release priority
                        round(replenishment / self.stores.shipping_buffer[product], 3),
                    )
                )

            # Ordenate by priority
            orders = list(sorted(orders, key=lambda x: x[-1], reverse=True))
            # Release orders based on priority

            print(f"\n {self.env.now} \n")
            print(f"{self.stores.contraint_resource}")
            print(f"{self.stores.constraint_buffer}")
            print(f"{self.stores.constraint_buffer_level}")
            print(orders)
            if self.stores.constraint_buffer_level < self.stores.constraint_buffer:
                ccr_safe_load = (
                    self.stores.constraint_buffer - self.stores.constraint_buffer_level
                )

                print(f"safe load: {ccr_safe_load}")

                for productionOrder, ccr_time, _ in orders:
                    release = True
                    product = productionOrder.product
                    quantity = productionOrder.quantity

                    if ccr_time > 0:
                        ccr_time = (quantity * ccr_time) + ccr_setup_time
                        productionOrder.schedule = self.env.now + ccr_time
                    else:
                        productionOrder.schedule = self.env.now

                    if quantity > 0:

                        self.env.process(self.process_order(productionOrder, ccr_time))
                        ccr_safe_load -= ccr_time
                        print(f"safe load updated: {ccr_safe_load}")
                    if ccr_safe_load <= 0:
                        break

            yield self.env.timeout(interval)

    def process_order(self, productionOrder: ProductionOrder, ccr_add: float):

        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)

        self.stores.constraint_buffer_level += ccr_add

        self.env.process(self.release_order(productionOrder))

    def calculate_penetration(self, product):

        finished_goods = self.stores.finished_goods[product].level
        target_level = self.stores.shipping_buffer[product]

        penetration = target_level - finished_goods

        return penetration

    def _process_demandOders(self):

        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            # print(f"{product}: {demandOrder}")
            yield self.stores.outbound_demand_orders[product].put(demandOrder)
