import argparse
from pathlib import Path
from typing import Any, Dict, Literal


def add_general_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add general simulation arguments to parser."""

    general = parser.add_argument_group("General Settings")
    general.add_argument(
        "--run-until", type=int, default=200001, help="Simulation end time"
    )

    general.add_argument(
        "--warmup", type=int, default=100000, help="Warmup for start logging results"
    )
    general.add_argument(
        "--log-interval", type=int, default=48, help="Interval between vars log"
    )
    general.add_argument(
        "--save-logs",
        action="store_false",
        help="Flat to register logs on simulation",
    )
    general.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def add_monitoring_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add monitoring arguments to parser."""
    monitoring = parser.add_argument_group("Monitoring")
    monitoring.add_argument(
        "--monitor-interval",
        type=int,
        default=50000,
        help="Interval for monitor prints",
    )
    monitoring.add_argument(
        "--monitor-warmup", type=int, default=0, help="Warmup for monitor prints"
    )
    monitoring.add_argument(
        "--print-mode",
        type=str,
        default=None,
        help="Warmup for monitor prints. Options: 'all', 'metrics', 'status', None",
    )
    return parser


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add configuration file arguments to parser."""
    config = parser.add_argument_group("Configuration Files")
    config.add_argument(
        "--config", type=Path, default=None, help="General config YAML file path"
    )
    config.add_argument(
        "--resources", type=Path, default=None, help="Resource config YAML file path"
    )
    config.add_argument(
        "--products", type=Path, default=None, help="Product config YAML file path"
    )
    return parser


def create_simulation_parser() -> argparse.ArgumentParser:
    """Create parser for single simulation runs."""
    parser = argparse.ArgumentParser(
        description="Run simulation environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = add_general_args(parser)
    parser = add_monitoring_args(parser)
    parser = add_config_args(parser)

    return parser


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add experiment-specific arguments to parser."""
    experiment = parser.add_argument_group("Experiment Settings")
    experiment.add_argument(
        "--number-of-runs", type=int, default=1, help="Number of simulation runs"
    )
    experiment.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment short name",
    )
    experiment.add_argument(
        "--exp-seed",
        type=str,
        default=None,
        help="Experiment seed",
    )
    experiment.add_argument(
        "--save-folder",
        type=Path,
        default=None,
        help="Folder path to save experiment results",
    )
    return parser


def create_experiment_parser() -> argparse.ArgumentParser:
    """Create parser for experiment runs."""
    parser = argparse.ArgumentParser(
        description="Run DBR simulation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = add_monitoring_args(parser)
    parser = add_config_args(parser)
    parser = add_experiment_args(parser)

    return parser


def extract_simulation_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract simulation arguments from parsed args."""
    return {
        "run_until": args.run_until,
        "monitor_interval": args.monitor_interval,
        "log_interval": args.log_interval,
        "monitor_warmup": args.monitor_warmup,
        "warmup": args.warmup,
        "seed": args.seed,
        "schedule_interval": args.schedule_interval,
        "constraint_buffer_size": args.cb_size,
        "ccr_release_limit": args.ccr_release_limit,
    }
