import argparse
from pathlib import Path

import yaml

from rlsim.environment import Environment


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation environment")

    parser.add_argument(
        "--run-until", type=int, default=200001, help="Simulation end time"
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=50000, help="Monitor sampling interval"
    )
    parser.add_argument(
        "--log-interval", type=int, default=48, help="Log sampling interval"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup duration for start logging results",
    )
    parser.add_argument(
        "--monitor-warmup", type=int, default=0, help="Warmup duration for monitor"
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config files
    resource_path = Path("example/config/resources.yaml")
    with open(resource_path, "r") as file:
        resources_cfg = yaml.safe_load(file)

    products_path = Path("example/config/products.yaml")
    with open(products_path, "r") as file:
        products_cfg = yaml.safe_load(file)

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
    )

    sim.run_simulation()
