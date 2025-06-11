from typing import Literal

import simpy

from rlsim.engine.control import DemandOrder, Stores


class Outbound:
    def __init__(
        self,
        stores: Stores,
        products_cfg: dict,
        delivery_mode: Literal["asReady", "onDue", "instantly"],
        training: bool = False,
    ):
        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.products = products_cfg
        self.delivery_mode = delivery_mode
        self.training = training

        if self.delivery_mode == "asReady":
            for product in self.products.keys():
                self.env.process(self._delivery_as_ready(product))

        elif self.delivery_mode == "onDue":
            for product in self.products.keys():
                self.env.process(self._delivery_on_duedate(product))

        elif self.delivery_mode == "instantly":
            for product in self.products.keys():
                print(f"start delivery: {product}")
                self.env.process(self._delivery_instantly(product))

    def _delivery_instantly(self, product):
        while True:

            demandOrder: DemandOrder = yield self.stores.outbound_demand_orders[
                product
            ].get()

            quantity = demandOrder.quantity
            if self.stores.finished_goods[product].level >= quantity:
                yield self.stores.finished_goods[product].get(quantity)
                if not self.training and self.stores.warmup < self.env.now:
                    self.stores.delivered_ontime[product] += quantity

            elif self.stores.warmup < self.env.now:
                self.stores.lost_sales[product] += quantity

    def _delivery_as_ready(self, product):
        while True:
            demandOrder: DemandOrder = yield self.stores.outbound_demand_orders[
                product
            ].get()
            quantity = demandOrder.quantity
            duedate = demandOrder.duedate

            # remove from finished goods
            yield self.stores.finished_goods[product].get(quantity)
            # check ontime or late
            demandOrder.delivered = self.env.now
            if not self.training and self.stores.warmup < self.env.now:
                if demandOrder.delivered <= duedate:
                    self.stores.delivered_ontime[product] += quantity
                    self.stores.earliness[product].append(
                        demandOrder.duedate - self.env.now
                    )
                else:
                    self.stores.delivered_late[product] += quantity
                    self.stores.tardiness[product].append(
                        self.env.now - demandOrder.duedate
                    )

            self.stores.lead_time[product].append(self.env.now - demandOrder.arived)

    def _delivery_on_duedate(self, product):
        def _delivey_order(demandOrder: DemandOrder):
            quantity = demandOrder.quantity
            duedate = demandOrder.duedate

            # Whait for duedate
            delay = duedate - self.env.now
            self.env.timeout(delay)

            # Remove from finished goods
            yield self.stores.finished_goods[product].get(quantity)

            # Check ontime or late
            demandOrder.delivered = self.env.now
            if not self.training and self.stores.warmup < self.env.now:
                if demandOrder.delivered <= duedate:
                    self.stores.delivered_ontime[product] += quantity
                    self.stores.earliness[product].append(duedate - self.env.now)
                else:
                    self.stores.delivered_late[product] += quantity
                    self.stores.tardiness[product].append(self.env.now - duedate)

                self.stores.lead_time[product].append(self.env.now - demandOrder.arived)

        while True:
            demandOrder: DemandOrder = yield self.stores.outbound_demand_orders[
                product
            ].get()
            self.env.process(_delivey_order(demandOrder))
