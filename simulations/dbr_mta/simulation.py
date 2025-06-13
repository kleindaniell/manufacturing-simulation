import argparse
from pathlib import Path
from time import time
from typing import List

import yaml
from scheduler import DBR_MTA
from stores import DBR_stores

from rlsim.engine.control import ProductionOrder
from rlsim.environment import Environment


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation environment")

    parser.add_argument(
        "--run-until", type=int, default=200001, help="Simulation end time"
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=50000,
        help="Interval for monitor prints",
    )
    parser.add_argument(
        "--log-interval", type=int, default=48, help="Interval between vars log"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100000,
        help="Warmup for start logging results",
    )
    parser.add_argument(
        "--monitor-warmup", type=int, default=0, help="Warmup for monitor prints"
    )

    parser.add_argument(
        "--cb-size", type=int, default=2000, help="Constraint buffer size"
    )
    parser.add_argument(
        "--schedule-interval", type=int, default=72, help="Schedule interval"
    )

    parser.add_argument(
        "--resources", default=None, type=str, help="Resouce config yaml file path"
    )

    parser.add_argument(
        "--products", default=None, type=str, help="Product config yaml file path"
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def order_selection_callback():
    def order_selection(store: DBR_stores, resource):
        orders: List[ProductionOrder] = store.resource_input[resource].items

        for id, productionOrder in enumerate(orders):
            ahead_orders: List[ProductionOrder] = []
            for resource_ in store.resources.keys():
                ahead_orders.extend(store.resource_input[resource_].items)
                ahead_orders.extend(store.resource_output[resource_].items)
                ahead_orders.extend(store.resource_transport[resource_].items)
                ahead_orders.extend(store.resource_processing[resource_].items)

            product = productionOrder.product
            released = productionOrder.released
            ahead_quantity = [
                order.quantity
                for order in ahead_orders
                if order.released < released and order.product == product
            ]
            orders[id].priority = (
                sum(ahead_quantity) + store.finished_goods[product].level
            ) / store.shipping_buffer[product]

        order = sorted(orders, key=lambda x: x.priority)[0]
        return order.id

    return order_selection


if __name__ == "__main__":
    args = parse_args()

    sim_path = Path(__file__).resolve().parent
    # Load YAML config files
    if args.resources:
        resource_path = args.resources
    else:
        resource_path = sim_path / "config/resources.yaml"
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    if args.products:
        resource_path = args.products
    else:
        products_path = sim_path / "config/products.yaml"
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

    start_time = time()

    # Instantiate and run simulation
    sim = Environment(
        run_until=args.run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        monitor_interval=args.monitor_interval,
        log_interval=args.log_interval,
        monitor_warmup=args.monitor_warmup,
        warmup=args.warmup,
        seed=args.seed,
        stores=DBR_stores,
        scheduler=DBR_MTA,
        scheduler_kwargs={
            "schedule_interval": args.schedule_interval,
            "constraint_buffer_size": args.cb_size,
        },
        outbound_kwargs={"delivery_mode": "instantly"},
        production_kwargs={"order_selection_fn": order_selection_callback()},
    )

    sim.run_simulation()

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    df_products = sim.stores.log_products.to_dataframe()
    df_products.to_csv(sim_path / "data/products.csv", index=False)
    df_resources = sim.stores.log_resources.to_dataframe()
    df_resources.to_csv(sim_path / "data/resources.csv", index=False)
    sim.save_parameters(sim_path / "data")
