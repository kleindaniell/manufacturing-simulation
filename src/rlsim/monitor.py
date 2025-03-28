import simpy
from rlsim.control import Stores

import pandas as pd
import numpy as np


class Monitor:
    def __init__(
        self,
        stores: Stores,
        interval: int = 72,
        warmup: int = 0,
        print_type: str = "all",
    ):
        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.interval = interval
        self.warmup = warmup
        self.print_type = print_type

        self.env.process(self.run())

    def run(self):
        yield self.env.timeout(self.warmup)
        while True:
            # orders = self.measure_total_wip()
            # print(f"wip: {self.env.now} {len(orders)}")

            resources_queue = self.measure_resource_queues()
            print(resources_queue)
            yield self.env.timeout(self.interval)
            # print(self.env.now)

    def measure_total_wip(self) -> list:
        orders = []
        for resource in self.stores.resources.keys():
            orders.extend(self.stores.resource_output[resource].items)
            orders.extend(self.stores.resource_input[resource].items)
            orders.extend(self.stores.resource_processing[resource].items)
            orders.extend(self.stores.resource_transport[resource].items)
        return orders

    def measure_resource_queues(self) -> pd.DataFrame:
        resources_list = self.stores.resources.keys()
        products_list = self.stores.products.keys()

        df = pd.DataFrame(
            np.zeros((len(products_list), len(resources_list))),
            index=products_list,
            columns=resources_list,
        )

        for resource in resources_list:
            orders = self.stores.resource_input[resource].items
            for order in orders:
                product = order["product"]

                df.loc[product, resource] += 1

        return df
