import numpy as np
import pandas as pd
from time import time
from typing import Tuple
import simpy


from rlsim.engine.control import Stores


class Monitor:
    def __init__(
        self,
        stores: Stores,
        interval: int = 72,
        warmup: int = 0,
        show_print: bool = False,
    ):
        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.interval = interval
        self.warmup = warmup
        self.show_print = show_print

        if self.show_print:
            self.env.process(self.run())

    def run(self):
        start_time = time()
        yield self.env.timeout(self.warmup)
        while True:
            df_status = self.measure_status()
            df_resource = self.measure_resources()
            df_product = self.measure_products()

            end_time = time()
            elapsed_time = end_time - start_time
            print(f"Monitor Elapsed time: {elapsed_time:.4f} seconds")
            start_time = time()
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
        for product in products_list:
            df_print.loc[product, "fg"] = self.stores.finished_goods[product].level
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
                if len(self.stores.log_resources.resource_utilization[resource]) > 0:
                    try:
                        df_data[i, 0] = np.array(
                            self.stores.log_resources.resource_utilization[resource],
                            dtype=np.float32,
                        )[:, 1].sum() / (self.env.now - self.stores.warmup)
                    except ZeroDivisionError:
                        df_data[i, 0] = 0

                df_data[i, 1] = len(
                    self.stores.log_resources.resource_breakdowns[resource]
                )
                if df_data[i, 1] > 0:
                    df_data[i, 2] = np.array(
                        self.stores.log_resources.resource_breakdowns[resource]
                    )[:, 1].mean()

                df_data[i, 3] = len(self.stores.log_resources.resource_setup[resource])
                if df_data[i, 3] > 0:
                    df_data[i, 4] = np.array(
                        self.stores.log_resources.resource_setup[resource]
                    )[:, 1].mean()

        df_data = df_data.round(3)

        df_resources = pd.DataFrame(df_data, columns=columns, index=resources_list)

        return df_resources

    def measure_products(self):
        products_list = self.stores.products.keys()
        columns = [
            "delivered_ontime",
            "delivered_late",
            "lost_sales",
            "tardiness",
            "earliness",
            "flow_time",
            "lead_time",
            "wip",
            "fg",
        ]

        df_data = np.zeros(shape=(len(products_list), len(columns)))
        delivered_ontime = []
        delivered_late = []
        lost_sales = []
        tardiness = []
        earliness = []
        flowtime = []
        leadtime = []
        wip_log = []
        fg_log = []
        if self.env.now >= self.stores.warmup:
            for i, product in enumerate(products_list):
                # Delivered ontime
                if len(self.stores.log_products.delivered_ontime[product]) > 0:
                    df_data[i, 0] = np.array(
                        self.stores.log_products.delivered_ontime[product]
                    )[:, 1].sum()
                    delivered_ontime.extend(
                        self.stores.log_products.delivered_ontime[product]
                    )
                # Delivered Late
                if len(self.stores.log_products.delivered_late[product]) > 0:
                    df_data[i, 1] = np.array(
                        self.stores.log_products.delivered_late[product]
                    )[:, 1].sum()
                    delivered_late.extend(
                        self.stores.log_products.delivered_late[product]
                    )
                # Lost Sales
                if len(self.stores.log_products.lost_sales[product]) > 0:
                    df_data[i, 2] = np.array(
                        self.stores.log_products.lost_sales[product]
                    )[:, 1].sum()
                    lost_sales.extend(self.stores.log_products.lost_sales[product])
                # Tardiness
                if len(self.stores.log_products.tardiness[product]) > 0:
                    df_data[i, 3] = np.array(
                        self.stores.log_products.tardiness[product]
                    )[:, 1].mean()
                    tardiness.extend(self.stores.log_products.tardiness[product])
                # Earliness
                if len(self.stores.log_products.earliness[product]) > 0:
                    df_data[i, 4] = np.array(
                        self.stores.log_products.earliness[product]
                    )[:, 1].mean()
                    earliness.extend(self.stores.log_products.earliness[product])
                # Flow time
                if len(self.stores.log_products.flow_time[product]) > 0:
                    df_data[i, 5] = np.array(
                        self.stores.log_products.flow_time[product]
                    )[:, 1].mean()
                    flowtime.extend(self.stores.log_products.flow_time[product])
                # Lead time
                if len(self.stores.log_products.lead_time[product]) > 0:
                    df_data[i, 6] = np.array(
                        self.stores.log_products.lead_time[product]
                    )[:, 1].mean()
                    leadtime.extend(self.stores.log_products.lead_time[product])

                # Wip
                if len(self.stores.log_products.wip_log[product]) > 0:
                    df_data[i, 7] = np.array(self.stores.log_products.wip_log[product])[
                        :, 1
                    ].mean()
                    wip_log.extend(self.stores.log_products.wip_log[product])
                # Finished Goods
                if len(self.stores.log_products.fg_log[product]) > 0:
                    df_data[i, 8] = np.array(self.stores.log_products.fg_log[product])[
                        :, 1
                    ].mean()
                    fg_log.extend(self.stores.log_products.fg_log[product])

            df_data = df_data.round(3)

            df_products = pd.DataFrame(df_data, columns=columns, index=products_list)
            df_products.loc["mean", :] = df_products.mean(axis=0)

            ontime_mean = (
                np.array(delivered_ontime)[:, 1].sum()
                if len(delivered_ontime) > 0
                else 0
            )
            late_mean = (
                np.array(delivered_late)[:, 1].sum() if len(delivered_late) > 0 else 0
            )
            lost_mean = np.array(lost_sales)[:, 1].sum() if len(lost_sales) > 0 else 0
            flowtime_mean = np.array(flowtime)[:, 1].mean() if len(flowtime) > 0 else 0
            leadtime_mean = np.array(leadtime)[:, 1].mean() if len(leadtime) > 0 else 0
            tardiness_mean = (
                np.array(tardiness)[:, 1].mean() if len(tardiness) > 0 else 0
            )
            earliness_mean = (
                np.array(earliness)[:, 1].mean() if len(earliness) > 0 else 0
            )
            total_wip_mean = (
                np.array(self.stores.total_wip_log)[:, 1].mean()
                if len(self.stores.total_wip_log) > 0
                else 0
            )
            fg_log_mean = np.array(fg_log)[:, 1].mean() if len(fg_log) > 0 else 0

            df_products.loc["total", :] = [
                ontime_mean,
                late_mean,
                lost_mean,
                tardiness_mean,
                earliness_mean,
                flowtime_mean,
                leadtime_mean,
                total_wip_mean,
                fg_log_mean,
            ]

            return df_products
        else:
            return pd.DataFrame(columns=columns, index=products_list)
