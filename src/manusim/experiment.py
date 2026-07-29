import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from manusim.engine.utils import DistributionGenerator
from manusim.factory_sim import FactorySimulation


def _resolve_n_jobs(n_jobs: int) -> int:
    """Force sequential runs when Hydra multirun handles outer parallelism."""
    if n_jobs <= 1:
        return n_jobs
    if HydraConfig.initialized() and HydraConfig.get().mode == RunMode.MULTIRUN:
        print(
            "Hydra multirun is active; forcing ExperimentRunner n_jobs to 1 "
            f"(requested {n_jobs}). Use hydra.launcher.n_jobs for sweep parallelism."
        )
        return 1
    return n_jobs


def _instantiate_simulation(
    sim_cls: type[FactorySimulation], kwargs: Dict[str, Any]
) -> FactorySimulation:
    return sim_cls(**kwargs)


def make_simulation_factory(
    sim_cls: type[FactorySimulation], **kwargs: Any
) -> Callable[[], FactorySimulation]:
    """Build a picklable factory using functools.partial and plain dict kwargs."""
    return partial(_instantiate_simulation, sim_cls, kwargs)


def _run_single_simulation(
    sim: FactorySimulation,
    run_id: int,
    seed: int,
    save_folder_path: Path,
    save_logs: bool,
) -> Dict[str, Any]:
    """Run a single simulation and save results."""
    save_folder_path = Path(save_folder_path)
    run_folder = save_folder_path / f"run_{run_id:03d}"
    run_folder.mkdir(exist_ok=True)

    log_save_path = run_folder / "logs"
    sim.reset_simulation(seed, log_save_path)

    elapsed_time = sim.run_simulation()

    sim.save_metrics(save_path=run_folder, saved_logs=True)
    sim.save_custom_metrics(save_path=run_folder, saved_logs=True)

    if save_logs:
        sim.logs.save_all_logs()

    run_info = {
        "run_id": run_id,
        "seed": sim.seed,
        "elapsed_time": elapsed_time,
        "simulation_end_time": sim.env.now,
        "run_folder": str(run_folder),
    }

    with open(run_folder / "run_info.yaml", "w") as f:
        yaml.dump(run_info, f)

    return run_info


def _run_single_simulation_worker(
    task: Tuple[int, int, Path, bool, Callable[[], FactorySimulation]],
) -> Dict[str, Any]:
    run_id, seed, save_folder_path, save_logs, simulation_factory = task
    sim = simulation_factory()
    return _run_single_simulation(
        sim, run_id, seed, save_folder_path, save_logs
    )


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
        n_jobs: int = 1,
        simulation_factory: Callable[[], FactorySimulation] | None = None,
    ):
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")

        n_jobs = _resolve_n_jobs(n_jobs)

        if n_jobs > 1 and simulation_factory is None:
            raise ValueError(
                "simulation_factory is required when n_jobs > 1"
            )

        self.sim = simulation
        self.number_of_runs = number_of_runs
        self.run_name = run_name
        self.save_logs = save_logs
        self.seed = seed
        self.n_jobs = n_jobs
        self.simulation_factory = simulation_factory

        if save_folder_path:
            self.save_folder_path = Path(save_folder_path)
        else:
            hdcfg = HydraConfig.get()
            save_path = hdcfg.runtime.output_dir
            self.save_folder_path = Path(save_path)

        self.rng = DistributionGenerator(self.seed)

        self._create_experiment_folder()

        self.run_metas = []

    def _create_experiment_folder(self) -> None:
        """Create experiment folder with timestamp"""
        self.save_folder_path.mkdir(parents=True, exist_ok=True)

    def _generate_run_seeds(self) -> List[int]:
        return [self.rng.random_int() for _ in range(self.number_of_runs)]

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Run multiple simulation experiments."""
        print("\n")
        print(f"Starting experiment with {self.number_of_runs} runs")
        if self.n_jobs > 1:
            print(f"Running in parallel with n_jobs={self.n_jobs}")
        print(f"Results will be saved to: {self.save_folder_path}")

        run_seeds = self._generate_run_seeds()
        start_time = time.time()

        if self.n_jobs <= 1:
            self.run_metas = self._run_sequential(run_seeds)
        else:
            self.run_metas = self._run_parallel(run_seeds)

        total_time = time.time() - start_time
        print(f"\nExperiment completed in {total_time:.4f} seconds")

        self._save_experiment_summary(total_time)

        return self.run_metas

    def _run_sequential(self, run_seeds: List[int]) -> List[Dict[str, Any]]:
        run_metas = []
        for run_id, run_seed in enumerate(run_seeds, start=1):
            print("\n" + "=" * 50)
            print(f"--- Running simulation - {run_id}/{self.number_of_runs} ---")
            print("=" * 50)

            run_meta = _run_single_simulation(
                self.sim,
                run_id,
                run_seed,
                self.save_folder_path,
                self.save_logs,
            )
            run_metas.append(run_meta)

            print("\n" + "=" * 50)
            print(f"Run {run_id} completed in {run_meta['elapsed_time']:.4f} seconds")
            print("=" * 50)

        return run_metas

    def _run_parallel(self, run_seeds: List[int]) -> List[Dict[str, Any]]:
        tasks = [
            (
                run_id,
                run_seed,
                self.save_folder_path,
                self.save_logs,
                self.simulation_factory,
            )
            for run_id, run_seed in enumerate(run_seeds, start=1)
        ]
        max_workers = min(self.n_jobs, self.number_of_runs)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            run_metas = list(pool.map(_run_single_simulation_worker, tasks))

        run_metas.sort(key=lambda r: r["run_id"])

        for run_meta in run_metas:
            print(
                f"Run {run_meta['run_id']} completed in "
                f"{run_meta['elapsed_time']:.4f} seconds"
            )

        return run_metas

    def _save_experiment_summary(self, total_time: float):
        """Save experiment summary and aggregated results."""
        summary = {
            "experiment_info": {
                "number_of_runs": self.number_of_runs,
                "n_jobs": self.n_jobs,
                "total_experiment_time": total_time,
                "average_run_time": sum(r["elapsed_time"] for r in self.run_metas)
                / len(self.run_metas),
                "save_folder_path": str(self.save_folder_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "runs_summary": self.run_metas,
        }

        with open(self.save_folder_path / "experiment_summary.yaml", "w") as f:
            yaml.dump(summary, f)

        results_df = pd.DataFrame(self.run_metas)
        results_df.to_csv(
            self.save_folder_path / "experiment_metadata.csv", index=False
        )

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
