import numpy as np
import pandas as pd
import simpy

from rlsim.engine.control import Stores


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
            df_status = self.measure_status()
            df_resource = self.measure_resources()
            df_product = self.measure_products()

            print(f"Status - Now: {self.env.now}")
            print(df_status)
            print("Resources")
            print(df_resource)
            print("Product")
            print(df_product)
            yield self.env.timeout(self.interval)

    def measure_status(self) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        df_print.loc["total", :] = df_print.sum(axis=0)
        return df_print

    def measure_resources(self):
        resources_list = self.stores.resources.keys()
        columns = [
            "utilization",
            "breakdowns",
            "breakdowns_mean",
            "setups",
            "setups_mean",
        ]

        df_data = np.zeros(shape=(len(resources_list), len(columns)))
        if self.env.now >= self.stores.warmup:
            for i, resource in enumerate(resources_list):

                try:
                    df_data[i, 0] = self.stores.resource_utilization[resource] / (
                        self.env.now - self.stores.warmup
                    )
                except ZeroDivisionError:
                    df_data[i, 0] = 0

                df_data[i, 1] = len(self.stores.resource_breakdowns[resource])
                if df_data[i, 1] > 0:
                    df_data[i, 2] = np.array(
                        [
                            x["duration"]
                            for x in self.stores.resource_breakdowns[resource]
                        ]
                    ).mean()

                df_data[i, 3] = len(self.stores.resource_setup[resource])
                if df_data[i, 3] > 0:
                    df_data[i, 4] = np.array(
                        [x["duration"] for x in self.stores.resource_setup[resource]]
                    ).mean()

        df_data = df_data.round(3)

        df_resources = pd.DataFrame(df_data, columns=columns, index=resources_list)

        return df_resources

    def measure_products(self):
        products_list = self.stores.products.keys()
        columns = ["flow_time", "lead_time", "tardiness", "earliness", "wip"]

        df_data = np.zeros(shape=(len(products_list), len(columns)))
        flowtime = []
        leadtime = []
        tardiness = []
        earliness = []
        wip_log = []
        if self.env.now >= self.stores.warmup:

            for i, product in enumerate(products_list):

                if len(self.stores.flow_time[product].items) > 0:
                    df_data[i, 0] = np.array(
                        self.stores.flow_time[product].items
                    ).mean()
                    flowtime.extend(self.stores.flow_time[product].items)
                if len(self.stores.lead_time[product].items) > 0:
                    df_data[i, 1] = np.array(
                        self.stores.lead_time[product].items
                    ).mean()
                    leadtime.extend(self.stores.lead_time[product].items)
                if len(self.stores.tardiness[product].items) > 0:
                    df_data[i, 2] = np.array(
                        self.stores.tardiness[product].items
                    ).mean()
                    tardiness.extend(self.stores.tardiness[product].items)
                if len(self.stores.earliness[product].items) > 0:
                    df_data[i, 3] = np.array(
                        self.stores.earliness[product].items
                    ).mean()
                    earliness.extend(self.stores.earliness[product].items)
                if len(self.stores.wip_log[product].items) > 0:
                    print(np.array(self.stores.wip_log[product].items).mean())
                    df_data[i, 4] = np.array(self.stores.wip_log[product].items).mean()
                    wip_log.extend(self.stores.wip_log[product].items)

        df_data = df_data.round(3)

        df_products = pd.DataFrame(df_data, columns=columns, index=products_list)
        df_products.loc["mean", :] = df_products.mean(axis=0)
        df_products.loc["total_mean", :] = [
            np.mean(flowtime),
            np.mean(leadtime),
            np.mean(tardiness),
            np.mean(earliness),
            np.mean(wip_log),
        ]

        return df_products
