from pathlib import Path
from time import time
from typing import List, Optional, Dict, Any, Callable

import yaml
from scheduler import DBR_MTA
from stores import DBR_stores

from rlsim.engine.control import ProductionOrder
from rlsim.environment import Environment, load_config
from cli_config import create_simulation_parser, extract_simulation_args


class SimulationDBR:
    """Simple wrapper for Environment class with DBR-specific components."""

    def __init__(
        self,
        run_until: int,
        resources_cfg: Dict[str, Any],
        products_cfg: Dict[str, Any],
        monitor_interval: int = 50000,
        log_interval: int = 48,
        monitor_warmup: int = 0,
        warmup: int = 100000,
        seed: Optional[int] = None,
        schedule_interval: int = 72,
        constraint_buffer_size: float = float("inf"),
        ccr_release_limit: float = float("inf"),
    ):
        self.sim = Environment(
            run_until=run_until,
            resources_cfg=resources_cfg,
            products_cfg=products_cfg,
            monitor_interval=monitor_interval,
            log_interval=log_interval,
            monitor_warmup=monitor_warmup,
            warmup=warmup,
            seed=seed,
            stores=DBR_stores,
            scheduler=DBR_MTA,
            scheduler_kwargs={
                "schedule_interval": schedule_interval,
                "constraint_buffer_size": constraint_buffer_size,
                "ccr_release_limit": ccr_release_limit,
            },
            outbound_kwargs={"delivery_mode": "instantly"},
            production_kwargs={"order_selection_fn": self._create_order_selection_fn()},
        )

    def run_simulation(self) -> float:
        """Run the simulation and return elapsed time."""
        start_time = time()
        self.sim.run_simulation()
        elapsed_time = time() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        return elapsed_time

    def save_logs(self, sim_path: Path):
        """Save simulation logs to CSV files."""
        if not isinstance(sim_path, Path):
            sim_path = Path(sim_path)

        data_path = sim_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving logs to {data_path}")
        df_products = self.sim.stores.log_products.to_dataframe()
        df_products.to_csv(data_path / "products.csv", index=False)

        df_resources = self.sim.stores.log_resources.to_dataframe()
        df_resources.to_csv(data_path / "resources.csv", index=False)

    def save_params(self, sim_path: Path):
        """Save simulation parameters."""
        if not isinstance(sim_path, Path):
            sim_path = Path(sim_path)

        data_path = sim_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving params to {data_path}")
        self.sim.save_parameters(data_path)

    def _create_order_selection_fn(self) -> Callable:
        """Create the DBR order selection function."""

        def order_selection(store: DBR_stores, resource) -> int:
            orders: List[ProductionOrder] = store.resource_input[resource].items

            for id, production_order in enumerate(orders):
                # Get all orders ahead in the system
                ahead_orders: List[ProductionOrder] = []
                for resource_ in store.resources.keys():
                    ahead_orders.extend(store.resource_input[resource_].items)
                    ahead_orders.extend(store.resource_output[resource_].items)
                    ahead_orders.extend(store.resource_transport[resource_].items)
                    ahead_orders.extend(store.resource_processing[resource_].items)

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
                    sum(ahead_quantity) + store.finished_goods[product].level
                ) / store.shipping_buffer[product]

            # Return order with lowest priority
            selected_order = min(orders, key=lambda x: x.priority)
            return selected_order.id

        return order_selection


def main():
    """Main execution function."""
    parser = create_simulation_parser()
    args = parser.parse_args()

    # Determine paths
    sim_path = args.sim_path or Path(__file__).resolve().parent
    resource_path = args.resources or sim_path / "config/resources.yaml"
    products_path = args.products or sim_path / "config/products.yaml"

    # Load configurations
    resources_cfg = load_config(resource_path)
    products_cfg = load_config(products_path)

    # Extract simulation arguments
    simulation_args = extract_simulation_args(args)

    # Create and run simulation
    sim = SimulationDBR(
        resources_cfg=resources_cfg, products_cfg=products_cfg, **simulation_args
    )

    sim.run_simulation()
    sim.save_logs(sim_path)
    sim.save_params(sim_path)


if __name__ == "__main__":
    main()
