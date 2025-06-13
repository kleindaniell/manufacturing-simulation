import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml

from cli_config import create_experiment_parser, extract_simulation_args
from rlsim.environment import load_config
from simulation import SimulationDBR


class ExperimentRunner:
    """Runner for multiple DBR simulation experiments."""

    def __init__(
        self,
        resources_cfg: Dict[str, Any],
        products_cfg: Dict[str, Any],
        simulation_args: Dict[str, Any],
        number_of_runs: int,
        save_folder_path: Path,
        sim_path: Path = None,
    ):
        self.resources_cfg = resources_cfg
        self.products_cfg = products_cfg
        self.simulation_args = simulation_args
        self.number_of_runs = number_of_runs
        self.save_folder_path = Path(save_folder_path)
        self.sim_path = sim_path or Path.cwd()

        # Create save directory
        self.save_folder_path.mkdir(parents=True, exist_ok=True)

        # Store experiment results
        self.results = []

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Run multiple simulation experiments."""
        print(f"Starting experiment with {self.number_of_runs} runs")
        print(f"Results will be saved to: {self.save_folder_path}")

        start_time = time.time()

        for run_id in range(self.number_of_runs):
            print(f"\n--- Running simulation - {run_id + 1}/{self.number_of_runs} ---")

            # Set unique seed for each run if base seed is provided
            run_args = self.simulation_args.copy()
            if run_args.get("seed") is not None:
                run_args["seed"] = run_args["seed"] + run_id
            else:
                # Generate random seed for reproducibility
                run_args["seed"] = random.randint(0, 999999)

            # Run single simulation
            result = self._run_single_simulation(run_id, run_args)
            self.results.append(result)

            print(f"Run {run_id + 1} completed in {result['elapsed_time']:.4f} seconds")

        total_time = time.time() - start_time
        print(f"\nExperiment completed in {total_time:.4f} seconds")

        # Save experiment summary
        self._save_experiment_summary(total_time)

        return self.results

    def _run_single_simulation(
        self, run_id: int, run_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single simulation and save results."""
        # Create simulation
        sim = SimulationDBR(
            resources_cfg=self.resources_cfg, products_cfg=self.products_cfg, **run_args
        )

        # Run simulation
        elapsed_time = sim.run_simulation()

        # Create run-specific folder
        run_folder = self.save_folder_path / f"run_{run_id:03d}"
        run_folder.mkdir(exist_ok=True)

        # Save logs and parameters
        sim.save_logs(run_folder)
        sim.save_params(run_folder)

        # Save run-specific info
        run_info = {
            "run_id": run_id,
            "seed": run_args["seed"],
            "elapsed_time": elapsed_time,
            "simulation_end_time": sim.sim.env.now,
            "run_folder": str(run_folder),
            **run_args,
        }

        # Save run info
        with open(run_folder / "run_info.yaml", "w") as f:
            yaml.dump(run_info, f)

        return run_info

    def _save_experiment_summary(self, total_time: float):
        """Save experiment summary and aggregated results."""
        # Create summary
        summary = {
            "experiment_info": {
                "number_of_runs": self.number_of_runs,
                "total_experiment_time": total_time,
                "average_run_time": sum(r["elapsed_time"] for r in self.results)
                / len(self.results),
                "save_folder_path": str(self.save_folder_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "simulation_parameters": self.simulation_args,
            "runs_summary": self.results,
        }

        # Save experiment summary
        with open(self.save_folder_path / "experiment_summary.yaml", "w") as f:
            yaml.dump(summary, f)

        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.save_folder_path / "experiment_results.csv", index=False)

        # Print summary statistics
        self._print_summary_stats(results_df)

    def _print_summary_stats(self, results_df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"Number of runs: {len(results_df)}")
        print(
            f"Average elapsed time: {results_df['elapsed_time'].mean():.4f} ± {results_df['elapsed_time'].std():.4f} seconds"
        )
        print(f"Min elapsed time: {results_df['elapsed_time'].min():.4f} seconds")
        print(f"Max elapsed time: {results_df['elapsed_time'].max():.4f} seconds")
        print(f"Results saved to: {self.save_folder_path}")
        print("=" * 50)


def main():
    """Main execution function."""
    parser = create_experiment_parser()
    args = parser.parse_args()

    # Determine paths
    sim_path = args.sim_path or Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    save_folder_path = (
        args.save_folder_path or sim_path / "experiments" / f"{timestamp}_{args.name}"
    )
    resource_path = args.resources or sim_path / "config/resources.yaml"
    products_path = args.products or sim_path / "config/products.yaml"

    # Load configurations
    try:
        resources_cfg = load_config(resource_path)
        products_cfg = load_config(products_path)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Extract simulation arguments
    simulation_args = extract_simulation_args(args)

    # Create and run experiment
    try:
        experiment = ExperimentRunner(
            resources_cfg=resources_cfg,
            products_cfg=products_cfg,
            simulation_args=simulation_args,
            number_of_runs=args.number_of_runs,
            save_folder_path=save_folder_path,
            sim_path=sim_path,
        )

        experiment.run_experiment()
        return 0

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
