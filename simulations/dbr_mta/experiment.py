import random
from pathlib import Path
from typing import List
from time import time


import simpy
import yaml

from simulation import SimulationModel


def run_experiment(
    runs: int,
    save_folder: Path,
    run_until: int,
    resources_cfg: dict,
    products_cfg: dict,
    schedule_interval: int,
    set_constraint: int = None,
    monitor_interval: int = 0,
    warmup: int = 0,
    warmup_monitor: int = 0,
    log_interval: int = 72,
    seed: int = None,
):
    sim = SimulationModel(
        run_until=run_until,
        resources_cfg=resources_cfg,
        products_cfg=products_cfg,
        schedule_interval=schedule_interval,
        monitor_interval=monitor_interval,
        warmup=warmup,
        warmup_monitor=warmup_monitor,
    )

    sim.run_simulation()

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    df_products = sim.stores.log_products.to_dataframe()
    df_products.to_csv(Path("simulations/dbr_mta/data/products.csv"), index=False)
    df_resources = sim.stores.log_resources.to_dataframe()
    df_resources.to_csv(Path("simulations/dbr_mta/data/resources.csv"), index=False)

    print(Path(__file__).resolve())


if __name__ == "__main__":
    resource_path = Path("simulations/dbr_mta/config/resources.yaml")
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("simulations/dbr_mta/config/products.yaml")
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

    run_experiment()

#     run_until = 200001
#     schedule_interval = 48
#     monitor_interval = 50000
#     warmup = 100000
#     warmup_monitor = 0

#     start_time = time()

#     sim = Simulation(
#         run_until=run_until,
#         resources_cfg=resources_cfg,
#         products_cfg=products_cfg,
#         schedule_interval=schedule_interval,
#         monitor_interval=monitor_interval,
#         warmup=warmup,
#         warmup_monitor=warmup_monitor,
#     )

#     sim.run_simulation()

#     end_time = time()
#     elapsed_time = end_time - start_time
#     print(f"Elapsed time: {elapsed_time:.4f} seconds")

#     df_products = sim.stores.log_products.to_dataframe()
#     df_products.to_csv(Path("simulations/dbr_mta/data/products.csv"), index=False)
#     df_resources = sim.stores.log_resources.to_dataframe()
#     df_resources.to_csv(Path("simulations/dbr_mta/data/resources.csv"), index=False)
