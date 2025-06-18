from typing import Tuple, List, Callable

import pandas as pd
import numpy as np

from rlsim.factory_sim import FactorySimulation
from rlsim.experiment import ExperimentRunner
from rlsim.engine.orders import ProductionOrder, DemandOrder
from rlsim.engine.cli_config import create_experiment_parser
from rlsim.engine.utils import load_yaml


class DBRSimulation(FactorySimulation):
    def __init__(
        self,
        config,
        resources,
        products,
        save_logs=True,
        print_mode="metrics",
        seed=None,
        queue_order_selection=None,
    ):
        super().__init__(
            config,
            resources,
            products,
            save_logs,
            print_mode,
            seed,
            queue_order_selection,
        )
        self.queue_order_selection = self._create_order_selection_fn()
        self._initiate_environment()
        self.ccr_release_limit = self.config.get("ccr_release_limit", float("inf"))
        self.scheduler_interval = self.config.get("scheduler_interval", 72)
        self.cb_target_level = self.config.get("cb_target_level", float("inf"))

        self._create_constraint_buffer()
        self._create_shipping_buffers()

    def _start_custom_process(self):
        self.contraint_resource, self.utilization_df = self.define_constraint()
        self.env.process(self._update_buffer(self.contraint_resource))

    def _create_constraint_buffer(self):

        # Constraint buffers
        self.constraint_buffer = self.config.get("cb_target_level", float("inf"))
        self.constraint_buffer_level = 0

    def _create_shipping_buffers(self):
        # Shipping_buffer

        self.shipping_buffer = {
            p: self.products_config[p].get("shipping_buffer", 0)
            for p in self.products_config.keys()
        }
        self.shipping_buffer_level = {
            p: self.products_config[p].get("shipping_buffer", 0)
            for p in self.products_config.keys()
        }
        for product in self.products_config.keys():
            qnt = self.products_config[product].get("shipping_buffer", 0)
            self.stores.finished_goods[product].put(qnt)

        # for product in self.products_config:
        #     self.shipping_buffer[product] = self.products_config[product].get(
        #         "shipping_buffer", 0
        #     )
        #     self.shipping_buffer_level[product] = 0

    def _create_custom_logs(self):
        custom_logs = {
            "products": {
                "shipping_buffer": {p: [] for p in self.products_config.keys()},
            },
            "general": {"constraint_buffer": []},
        }
        return custom_logs

    def _register_custom_logs(self):
        def register_logs():
            yield self.env.timeout(self.warmup)
            while True:
                self.log_general.constraint_buffer.append(
                    (self.env.now, self.constraint_buffer_level)
                )
                for product in self.products_config:
                    self.log_product.shipping_buffer[product].append(
                        (self.env.now, self.calculate_shipping_buffer(product))
                    )
                yield self.env.timeout(self.log_interval)

        self.env.process(register_logs())

    def define_constraint(self) -> Tuple[str, pd.DataFrame]:
        df = pd.DataFrame(
            data=np.zeros(
                shape=(len(self.products_config), len(self.resources_config)),
                dtype=np.float32,
            ),
            index=self.products_config.keys(),
            columns=self.resources_config.keys(),
        )

        for product in self.products_config.keys():
            product_demand = self.products_config[product].get("demand")
            mean_arrival_rate = product_demand.get("freq").get("params")[0]
            quantity = product_demand.get("quantity").get("params")[0]

            for process in self.stores.processes_value_list[product]:
                mean_processing_time = process["processing_time"]["params"][0]
                resource = process["resource"]

                df.loc[product, resource] += mean_processing_time

            df.loc[product, :] = df.loc[product, :] * (1 / mean_arrival_rate) * quantity

        utilization_df = df.copy()
        constraint_resource = df.sum().sort_values(ascending=False).index[0]
        return constraint_resource, utilization_df

    def _update_buffer(self, constraint):
        while True:
            productionOrder: ProductionOrder = yield self.stores.resource_finished[
                constraint
            ].get()
            product = productionOrder.product
            actual_process = productionOrder.process_finished - 1
            product_process = self.stores.processes_value_list[product][actual_process]
            product_processing_time = product_process["processing_time"]["params"][0]
            self.constraint_buffer_level -= product_processing_time

    def calculate_shipping_buffer(self, product):
        self.shipping_buffer_level[product] = (
            self.stores.wip[product].level + self.stores.finished_goods[product].level
        )

        return self.shipping_buffer_level[product]

    # Start refactor
    def scheduler(self):
        ccr_setup_time_params = self.stores.resources[self.contraint_resource].get(
            "setup", {"params": None}
        )
        ccr_setup_time = ccr_setup_time_params.get("params", [0])[0]

        while True:
            orders: List[Tuple[ProductionOrder, float, float]] = []
            for product in self.stores.products.keys():

                ccr_processing_time = sum(
                    [
                        process["processing_time"]["params"][0]
                        for process in self.stores.processes_value_list[product]
                        if process["resource"] == self.contraint_resource
                    ]
                )

                replenishment, penetration = self.calculate_replenishment(product)

                orders.append(
                    (
                        # Production order
                        ProductionOrder(
                            product=product,
                            quantity=replenishment,
                            priority=round(
                                penetration / self.shipping_buffer[product], 3
                            ),
                        ),
                        # ccr processin time
                        ccr_processing_time,
                        # Release priority
                        round(replenishment / self.shipping_buffer[product], 3),
                    )
                )

            # Ordenate by priority
            orders = list(sorted(orders, key=lambda x: x[-1], reverse=True))

            # Release orders based on priority
            print(self.constraint_buffer_level, self.constraint_buffer)
            if self.constraint_buffer_level < self.constraint_buffer:
                ccr_safe_load = self.constraint_buffer - self.constraint_buffer_level

                ccr_released = 0
                for productionOrder, ccr_time, _ in orders:

                    product = productionOrder.product
                    quantity = productionOrder.quantity
                    release = False
                    if ccr_time > 0:
                        if ccr_safe_load > 0 and ccr_released < self.ccr_release_limit:
                            ccr_time = (quantity * ccr_time) + ccr_setup_time
                            productionOrder.schedule = self.env.now + ccr_time
                            release = True
                            ccr_released += 1
                    else:
                        ccr_time = 0
                        productionOrder.schedule = self.env.now
                        release = True

                    if quantity > 0 and release:
                        self.env.process(self.process_order(productionOrder, ccr_time))
                        ccr_safe_load -= ccr_time

            yield self.env.timeout(self.scheduler_interval)

    def process_order(self, productionOrder: ProductionOrder, ccr_add: float):
        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)

        self.constraint_buffer_level += ccr_add
        self.env.process(self._release_order(productionOrder))

    def _create_order_selection_fn(self) -> Callable:
        """Create the DBR order selection function."""

        def order_selection(self: DBRSimulation, resource) -> int:
            orders: List[ProductionOrder] = self.stores.resource_input[resource].items

            for id, production_order in enumerate(orders):
                # Get all orders ahead in the system
                ahead_orders: List[ProductionOrder] = []
                for resource_ in self.stores.resources.keys():
                    ahead_orders.extend(self.stores.resource_input[resource_].items)
                    ahead_orders.extend(self.stores.resource_output[resource_].items)
                    ahead_orders.extend(self.stores.resource_transport[resource_].items)
                    ahead_orders.extend(
                        self.stores.resource_processing[resource_].items
                    )

                product = production_order.product
                released = production_order.released

                # Calculate quantity of orders ahead for same product
                ahead_quantity = [
                    order.quantity
                    for order in ahead_orders
                    if order.released < released and order.product == product
                ]

                # Calculate priority
                orders[id].priority = (
                    sum(ahead_quantity) + self.stores.finished_goods[product].level
                ) / self.shipping_buffer[product]

            # Return order with lowest priority
            selected_order = min(orders, key=lambda x: x.priority)
            return selected_order.id

        return order_selection

    def calculate_replenishment(self, product):
        finished_goods = self.stores.finished_goods[product].level
        target_level = self.shipping_buffer[product]

        penetration = target_level - finished_goods
        replenishment = max(target_level - self.calculate_shipping_buffer(product), 0)
        print(
            f"{self.env.now} - {product} - {finished_goods} - {target_level} - {replenishment} - {self.calculate_shipping_buffer(product)}"
        )
        return replenishment, penetration

    def _process_demandOders(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            yield self.stores.outbound_demand_orders[product].put(demandOrder)


def main():
    """Main execution function."""
    parser = create_experiment_parser()
    args = parser.parse_args()

    # Determine paths
    if args.save_folder is None:
        raise ValueError("Experiment folder not specified")

    save_folder = args.save_folder
    config_path = args.config
    products_path = args.products
    resources_path = args.resources
    # Load configurations
    try:
        config = load_yaml(config_path)
        resources_cfg = load_yaml(resources_path)
        products_cfg = load_yaml(products_path)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    sim = DBRSimulation(
        config=config,
        resources=resources_cfg,
        products=products_cfg,
        save_logs=True,
        print_mode="metrics",
        seed=args.exp_seed,
    )

    # Create and run experiment
    # try:
    experiment = ExperimentRunner(
        simulation=sim,
        number_of_runs=args.number_of_runs,
        save_folder_path=save_folder,
        run_name=args.name,
        seed=args.exp_seed,
    )
    experiment.run_experiment()


if __name__ == "__main__":
    exit(main())
