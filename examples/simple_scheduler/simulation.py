from typing import List

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from manusim.factory_sim import FactorySimulation
from manusim.experiment import ExperimentRunner
from manusim.engine.orders import ProductionOrder
from manusim.metrics import ExperimentMetrics


class NewSimulation(FactorySimulation):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        print_mode="all",
        seed: int = None,
    ):
        super().__init__(
            config,
            resources,
            products,
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


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig):
    """Main execution function."""
    sim = NewSimulation(
        config=cfg.simulation,
        resources=cfg.resources,
        products=cfg.products,
        print_mode=cfg.simulation.print_mode,
    )

    experiment = ExperimentRunner(
        simulation=sim,
        number_of_runs=cfg.experiment.number_of_runs,
        save_logs=cfg.experiment.save_logs,
        run_name=cfg.experiment.name,
        seed=cfg.experiment.exp_seed,
    )
    experiment.run_experiment()

    metrics = ExperimentMetrics(experiment.save_folder_path)

    metrics.read_logs()
    stats_df = metrics.save_stats(0.95, 0.05)
    print("=" *50)
    print("Experiment Stats")
    print("=" *50)
    print(stats_df)
    print("=" *50)

if __name__ == "__main__":
    exit(main())
