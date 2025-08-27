import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig

from manusim.engine.utils import DistributionGenerator
from manusim.factory_sim import FactorySimulation
from manusim.metrics import ExperimentMetrics


class ExperimentRunner:
    """Runner for multiple experiments."""

    def __init__(
        self,
        simulation: FactorySimulation,
        number_of_runs: int,
        save_folder_path: Path = None,
        run_name: str = None,
        save_logs: bool = True,
        seed: int = None,
    ):
        self.sim = simulation
        self.number_of_runs = number_of_runs
        self.run_name = run_name
        self.save_logs = save_logs
        self.seed = seed

        if save_folder_path:
            self.save_folder_path = Path(save_folder_path)
        else:
            hdcfg = HydraConfig.get()
            save_path = hdcfg.runtime.output_dir
            self.save_folder_path = Path(save_path)

        self.rng = DistributionGenerator(self.seed)

        # Create save directory
        self._create_experiment_folder()

        # Store experiment metadata
        self.run_metas = []

    def _create_experiment_folder(self) -> None:
        """Create experiment folder with timestamp"""
        # timestamp = datetime.now().strftime("%y%m%d%H%M")
        self.save_folder_path.mkdir(parents=True, exist_ok=True)

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Run multiple simulation experiments."""
        print("\n")
        print(f"Starting experiment with {self.number_of_runs} runs")
        print(f"Results will be saved to: {self.save_folder_path}")

        # Utilize experiment seed as first run seed
        run_seed = self.seed

        start_time = time.time()
        for run_id in range(1, self.number_of_runs + 1):
            print("\n" + "=" * 50)
            print(f"--- Running simulation - {run_id}/{self.number_of_runs} ---")
            print("=" * 50)

            # Create new seed for run
            run_seed = self.rng.random_int()

            # Run single simulation
            run_meta = self._run_single_simulation(run_id, run_seed)
            self.run_metas.append(run_meta)

            print("\n" + "=" * 50)
            print(f"Run {run_id} completed in {run_meta['elapsed_time']:.4f} seconds")
            print("=" * 50)

        total_time = time.time() - start_time
        print(f"\nExperiment completed in {total_time:.4f} seconds")

        # Save experiment summary
        self._save_experiment_summary(total_time)

        return self.run_metas

    def _run_single_simulation(self, run_id: int, seed: int) -> Dict[str, Any]:
        """Run a single simulation and save results."""

        # Create run-specific folder
        run_folder = self.save_folder_path / f"run_{run_id:03d}"
        run_folder.mkdir(exist_ok=True)

        log_save_path = run_folder / "logs"
        self.sim.reset_simulation(seed, log_save_path)

        # Run simulation
        elapsed_time = self.sim.run_simulation()

        # Save metrics

        self.sim.save_metrics(run_folder)
        self.sim.save_custom_metrics(run_folder)

        # Save logs
        if self.save_logs:
            self.sim.logs.save_all_logs()

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
                "average_run_time": sum(r["elapsed_time"] for r in self.run_metas)
                / len(self.run_metas),
                "save_folder_path": str(self.save_folder_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "runs_summary": self.run_metas,
        }

        # Save experiment summary
        with open(self.save_folder_path / "experiment_summary.yaml", "w") as f:
            yaml.dump(summary, f)

        # Create results DataFrame
        results_df = pd.DataFrame(self.run_metas)
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
