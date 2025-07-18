from typing import List

import hydra
from omegaconf import DictConfig

from manusim.factory_sim import FactorySimulation
from manusim.experiment import ExperimentRunner
from manusim.engine.orders import ProductionOrder


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
    # def order_selection(self, resource):
    #     orders: List[ProductionOrder] = self.stores.resource_input[resource].items

    #     # Return order with highest priority
    #     if len(orders) > 0:
    #         selected_order = max(orders, key=lambda x: x.priority)
    #         # Get order from queue
    #         productionOrder = yield self.stores.resource_input[resource].get(
    #             lambda x: x.id == selected_order.id
    #         )
    #     else:
    #         productionOrder = yield self.stores.resource_input[resource].get()

    #     return productionOrder


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main execution function."""
    sim = NewSimulation(
        config=cfg.simulation,
        resources=cfg.resources,
        products=cfg.products,
        save_logs=cfg.simulation.save_logs,
        print_mode=cfg.simulation.print_mode,
    )

    experiment = ExperimentRunner(
        simulation=sim,
        number_of_runs=cfg.experiment.number_of_runs,
        save_folder_path=cfg.experiment.save_folder,
        run_name=cfg.experiment.name,
        seed=cfg.experiment.exp_seed,
    )
    experiment.run_experiment()


if __name__ == "__main__":
    exit(main())
