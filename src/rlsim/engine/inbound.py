import simpy
import numpy as np

from rlsim.engine.control import DemandOrder, Stores
from rlsim.engine.utils import Distribution


class Inbound:
    def __init__(
        self,
        stores: Stores,
    ):
        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.dist = Distribution(seed=self.stores.seed)

        for product in self.stores.products.keys():
            self.env.process(self._generate_demand_orders(product))

    def _generate_demand_orders(self, product):
        product_config = self.stores.products[product]

        freq_dist = product_config["demand"]["freq"].get("dist")
        freq_params = product_config["demand"]["freq"].get("params")
        quantity_dist = product_config["demand"]["quantity"].get("dist")
        quantity_params = product_config["demand"]["quantity"].get("params")
        due_dist = product_config["demand"]["duedate"].get("dist")
        due_params = product_config["demand"]["duedate"].get("params")

        while True:
            frequency = np.float32(self.dist.random_number(freq_dist, freq_params))

            quantity = round(self.dist.random_number(quantity_dist, quantity_params), 0)
            duedate = np.float32(self.dist.random_number(due_dist, due_params))

            yield self.env.timeout(frequency)

            duedate += self.env.now

            demandOrder = DemandOrder(
                product=product,
                quantity=quantity,
                duedate=duedate,
                arived=self.env.now,
            )
            # print(f"{self.env.now} - {demandOrder}")
            yield self.stores.inbound_demand_orders.put(demandOrder)
