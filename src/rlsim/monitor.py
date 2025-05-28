import numpy as np
import pandas as pd
import simpy

from rlsim.control import Stores


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
            df_print = self.measure_wip()
            print(df_print)
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

        df_print = pd.DataFrame(
            np.zeros((len(products_list), len(resources_list))),
            index=products_list,
            columns=resources_list,
        )

        for resource in resources_list:
            orders_queue = list(self.stores.resource_input[resource].items)
            orders_queue.extend(self.stores.resource_output[resource].items)
            orders_queue.extend(self.stores.resource_transport[resource].items)
            for order in orders_queue:
                product = order.product
                df_print.loc[product, resource] += order.quantity

        df_print["wip_total"] = df_print.sum(axis=1)
        df_print["fg"] = 0
        df_print["onTime"] = 0
        df_print["late"] = 0
        for product in products_list:
            df_print.loc[product, "fg"] = self.stores.finished_goods[product].level
            df_print.loc[product, "onTime"] = self.stores.delivered_ontime[
                product
            ].level
            df_print.loc[product, "late"] = self.stores.delivered_late[product].level
            df_print.loc[product, "lost"] = self.stores.lost_sales[product].level

        return df_print
