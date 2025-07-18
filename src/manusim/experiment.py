import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from manusim.engine.utils import DistributionGenerator
from manusim.factory_sim import FactorySimulation


class ExperimentRunner:
    """Runner for multiple experiments."""

    def __init__(
        self,
        simulation: FactorySimulation,
        number_of_runs: int,
        save_folder_path: Path,
        run_name: str = None,
        seed: int = None,
    ):
        self.sim = simulation
        self.number_of_runs = number_of_runs
        self.save_folder_path = Path(save_folder_path)
        self.run_name = run_name
        self.seed = seed

        self.rng = DistributionGenerator(self.seed)

        # Create save directory
        self._create_experiment_folder()

        # Store experiment results
        self.results = []

    def _create_experiment_folder(self) -> None:
        """Create experiment folder with timestamp"""
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        self.save_folder_path = self.save_folder_path / f"{timestamp}_{self.run_name}"
        self.save_folder_path.mkdir(parents=True, exist_ok=True)

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Run multiple simulation experiments."""
        print("\n")
        print(f"Starting experiment with {self.number_of_runs} runs")
        print(f"Results will be saved to: {self.save_folder_path}")

        start_time = time.time()

        for run_id in range(self.number_of_runs):
            print(f"\n--- Running simulation - {run_id + 1}/{self.number_of_runs} ---")

            # Create new seed for run
            run_seed = self.rng.random_int()
            self.sim.reset_simulation(run_seed)

            # Run single simulation
            result = self._run_single_simulation(run_id)
            self.results.append(result)

            print("\n")
            print(f"Run {run_id + 1} completed in {result['elapsed_time']:.4f} seconds")

        total_time = time.time() - start_time
        print(f"\nExperiment completed in {total_time:.4f} seconds")

        # Save experiment params
        self.sim.save_params(self.save_folder_path)

        # Save experiment summary
        self._save_experiment_summary(total_time)

        return self.results

    def _run_single_simulation(self, run_id: int) -> Dict[str, Any]:
        """Run a single simulation and save results."""

        # Run simulation
        elapsed_time = self.sim.run_simulation()

        # Create run-specific folder
        run_folder = self.save_folder_path / f"run_{run_id:03d}"
        run_folder.mkdir(exist_ok=True)

        # Save logs
        self.sim.save_history_logs(run_folder)

        # Save run-specific info
        run_info = {
            "run_id": run_id,
            "seed": self.sim.seed,
            "elapsed_time": elapsed_time,
            "simulation_end_time": self.sim.env.now,
            "run_folder": str(run_folder),
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
            "runs_summary": self.results,
        }

        # Save experiment summary
        with open(self.save_folder_path / "experiment_summary.yaml", "w") as f:
            yaml.dump(summary, f)

        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(
            self.save_folder_path / "experiment_metadata.csv", index=False
        )

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
