from typing import Tuple
import numpy as np
import pandas as pd

import simpy

from rlsim.engine.control import Stores, ProductionOrder


class DBR_stores(Stores):
    def __init__(
        self,
        env: simpy.Environment,
        resources: dict,
        products: dict,
        warmup: int = 0,
        log_interval: int = 72,
        training: bool = False,
        seed: int = None,
        **kwargs,
    ):
        super().__init__(
            env=kwargs.get("env", env),
            products=kwargs.get("products", products),
            resources=kwargs.get("resources", resources),
            warmup=kwargs.get("warmup", warmup),
            log_interval=kwargs.get("log_interval", log_interval),
            training=kwargs.get("training", training),
            seed=kwargs.get("seed", seed),
        )

        self._create_shipping_buffers()

        self.constraint_buffer = 0
        self.constraint_buffer_level = 0

        self.contraint_resource, self.utilization_df = self.define_constraint()

        self.update_constraint_buffer(self.contraint_resource)

    def _create_shipping_buffers(self):
        # Shipping_buffer
        self.shipping_buffer = {}
        self.shipping_buffer_level = {}
        self.production_orders = {}

        for product in self.products.keys():
            self.shipping_buffer[product] = self.products[product].get(
                "shipping_buffer", 0
            )

            self.shipping_buffer_level[product] = self.shipping_buffer[product]
            self.production_orders[product] = []

            self.finished_goods[product].put(self.shipping_buffer[product])

    def define_constraint(self) -> Tuple[str, pd.DataFrame]:
        df = pd.DataFrame(
            data=np.zeros(
                shape=(len(self.products), len(self.resources)), dtype=np.float32
            ),
            index=self.products.keys(),
            columns=self.resources.keys(),
        )

        for product in self.products.keys():
            product_demand = self.products[product].get("demand")
            mean_arrival_rate = product_demand.get("freq").get("params")[0]
            quantity = product_demand.get("quantity").get("params")[0]

            for process in self.processes_value_list[product]:
                mean_processing_time = process["processing_time"]["params"][0]
                resource = process["resource"]

                df.loc[product, resource] += mean_processing_time

            df.loc[product, :] = df.loc[product, :] * (1 / mean_arrival_rate) * quantity

        utilization_df = df.copy()
        constraint_resource = df.sum().sort_values(ascending=False).index[0]

        return constraint_resource, utilization_df

    def update_constraint_buffer(self, constraint):
        setup_config = self.resources[constraint].get("setup", {"params": None})
        setup_time = setup_config.get("params", [0])[0]

        def _update_buffer():
            while True:
                productionOrder: ProductionOrder = yield self.resource_finished[
                    constraint
                ].get()
                product = productionOrder.product
                quantity = productionOrder.quantity
                actual_process = productionOrder.process_finished - 1
                product_process = self.processes_value_list[product][actual_process]
                product_processing_time = product_process["processing_time"]["params"][
                    0
                ]
                self.constraint_buffer_level -= (
                    quantity * product_processing_time
                ) + setup_time

        self.env.process(_update_buffer())

    def calculate_shipping_buffer(self, product):
        self.shipping_buffer_level[product] = (
            self.wip[product].level + self.finished_goods[product].level
        )

        return self.shipping_buffer_level[product]
