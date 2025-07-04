from typing import List

from manusim.factory_sim import FactorySimulation
from manusim.experiment import ExperimentRunner
from manusim.engine.orders import ProductionOrder
from manusim.engine.cli_config import create_experiment_parser
from manusim.engine.utils import load_yaml


class NewSimulation(FactorySimulation):
    def __init__(
        self,
        config,
        resources,
        products,
        save_logs=True,
        print_mode="metrics",
        seed=None,
    ):
        super().__init__(
            config,
            resources,
            products,
            save_logs,
            print_mode,
            seed,
        )

    # New order selection method
    def order_selection(self, resource):
        orders: List[ProductionOrder] = self.stores.resource_input[resource].items

        # Return order with highest priority
        if len(orders) > 0:
            selected_order = max(orders, key=lambda x: x.priority)
            # Get order from queue
            productionOrder = yield self.stores.resource_input[resource].get(
                lambda x: x.id == selected_order.id
            )
        else:
            productionOrder = yield self.stores.resource_input[resource].get()

        return productionOrder


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

    sim = NewSimulation(
        config=config,
        resources=resources_cfg,
        products=products_cfg,
        save_logs=True,
        print_mode="metrics",
        seed=args.exp_seed,
    )

    # Create and run experiment
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
