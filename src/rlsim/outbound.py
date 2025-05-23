import simpy

from typing import Literal

from rlsim.control import Stores, ProductionOrder, DemandOrder
from rlsim.utils import random_number


class Outbound:
    def __init__(
        self,
        stores: Stores,
        products_cfg: dict,
        operation_mode: str,
        training: bool = False,
    ):

        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.products = products_cfg
        self.operation_mode = operation_mode
        self.training = training

        if self.operation_mode == "mto_instantly":
            for product in self.products.keys():
                self.env.process(self._delivery_demand_orders(product))
        elif self.operation_mode == "mts":
            for product in self.products.keys():
                self.env.process(self._sell_demand_orders(product))

    def _sell_demand_orders(self, product):
        """
        xxx
        """

        while True:

            demandOrder: DemandOrder = yield self.stores.demand_orders[product].get()

            quantity = demandOrder.quantity

            if self.stores.finished_goods[product].level >= quantity:
                yield self.stores.finished_goods[product].get(quantity)
                if not self.training:
                    yield self.stores.delivered_ontime[product].put(demandOrder)
            else:
                self.stores.lost_sales[product].put(quantity)

    def _delivery_demand_orders(self, product, on_due: bool = True) -> None:

        while True:

            demandOrder: DemandOrder = yield self.stores.demand_orders[product].get()
            quantity = demandOrder.quantity
            duedate = demandOrder.duedate

            # remove from finished goods
            yield self.stores.finished_goods[product].get(quantity)
            # check ontime or late
            demandOrder.delivered = self.env.now
            if not self.training:
                if demandOrder.delivered <= duedate:
                    yield self.stores.delivered_ontime[product].put(demandOrder)
                else:
                    yield self.stores.delivered_late[product].put(demandOrder)
