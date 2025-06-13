from typing import Literal

import simpy

from rlsim.engine.control import DemandOrder, Stores


class Outbound:
    def __init__(
        self,
        stores: Stores,
        delivery_mode: Literal["asReady", "onDue", "instantly"] = "asReady",
    ):
        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.delivery_mode = delivery_mode

        if self.delivery_mode == "asReady":
            for product in self.stores.products.keys():
                self.env.process(self._delivery_as_ready(product))

        elif self.delivery_mode == "onDue":
            for product in self.stores.products.keys():
                self.env.process(self._delivery_on_duedate(product))

        elif self.delivery_mode == "instantly":
            for product in self.stores.products.keys():
                self.env.process(self._delivery_instantly(product))

    def _delivery_instantly(self, product):
        while True:
            demandOrder: DemandOrder = yield self.stores.outbound_demand_orders[
                product
            ].get()

            quantity = demandOrder.quantity
            if self.stores.finished_goods[product].level >= quantity:
                yield self.stores.finished_goods[product].get(quantity)
                if not self.stores.training and self.stores.warmup < self.env.now:
                    self.stores.log_products.delivered_ontime[product].append(
                        (self.env.now, quantity)
                    )

            elif self.stores.warmup < self.env.now:
                self.stores.log_products.lost_sales[product].append(
                    (self.env.now, quantity)
                )

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
            if not self.stores.training and self.stores.warmup < self.env.now:
                if demandOrder.delivered <= duedate:
                    self.stores.log_products.delivered_ontime[product].append(
                        (self.env.now, quantity)
                    )
                    self.stores.log_products.earliness[product].append(
                        (self.env.now, demandOrder.duedate - self.env.now)
                    )
                else:
                    self.stores.log_products.delivered_late[product].append(
                        (self.env.now, quantity)
                    )
                    self.stores.log_products.tardiness[product].append(
                        (self.env.now, self.env.now - duedate)
                    )

            self.stores.log_products.lead_time[product].append(
                (self.env.now, self.env.now - demandOrder.arived)
            )

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
            if not self.stores.training and self.stores.warmup < self.env.now:
                if demandOrder.delivered <= duedate:
                    self.stores.log_products.delivered_ontime[product].append(
                        (self.env.now, quantity)
                    )
                    self.stores.log_products.earliness[product].append(
                        (self.env.now, duedate - self.env.now)
                    )
                else:
                    self.stores.log_products.delivered_late[product].append(
                        (self.env.now, quantity)
                    )
                    self.stores.log_products.tardiness[product].append(
                        (self.env.now, self.env.now - duedate)
                    )

                self.stores.log_products.lead_time[product].append(
                    (self.env.now, self.env.now - demandOrder.arived)
                )

        while True:
            demandOrder: DemandOrder = yield self.stores.outbound_demand_orders[
                product
            ].get()
            self.env.process(_delivey_order(demandOrder))
