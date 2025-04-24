import simpy
from rlsim.control import Stores
from typing import Union, Tuple

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

            wip, queue = self.measure_wip()
            print("WIP")
            print(wip)
            print("Queue")
            print(queue)
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

    def measure_wip(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        resources_list = self.stores.resources.keys()
        products_list = self.stores.products.keys()

        df_wip = pd.DataFrame(
            np.zeros((len(products_list), len(resources_list))),
            index=products_list,
            columns=resources_list,
        )

        df_queue = df_wip.copy()

        for resource in resources_list:
            orders_queue = list(self.stores.resource_input[resource].items)
            orders_queue.extend(self.stores.resource_output[resource].items)
            orders_queue.extend(self.stores.resource_transport[resource].items)
            for order in orders_queue:
                product = order.product
                df_queue.loc[product, resource] += 1

            orders_wip = list(self.stores.resource_processing[resource].items)
            for order in orders_wip:
                product = order.product
                df_wip.loc[product, resource] += 1

        return (df_wip, df_queue)
